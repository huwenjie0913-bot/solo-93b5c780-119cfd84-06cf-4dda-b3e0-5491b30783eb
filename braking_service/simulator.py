"""仿真编排：把请求转换为内部单位制，逐工况 × 制动模式积分并汇总结果。

提交 brake_groups 时启用分组制动传播：各车辆组按自身延迟/建立时间出力，
并以现有统一延迟模型为基线输出停车距离、最晚制动位置与停车余量对比。
"""

from __future__ import annotations

import uuid
from bisect import bisect_right
from dataclasses import replace
from typing import Optional

from .models import SimulationRequest
from .physics import (GRAVITY, BrakeGroupSpec, ForceModel, Trajectory,
                      backward_brake_position, downsample, integrate,
                      scan_violations)
from .supervision import build_envelope, compare_envelopes

MODES = ("service", "emergency")

LOW_MU_EPS = 1e-12          # 判定"低于标量基线"的黏着容差
ZONE_MERGE_EPS = 1e-6       # 相邻低黏着区段合并里程容差


class Track:
    """分段恒定的坡度/限速查询（内部单位：坡度 ratio，限速 m/s）。"""

    def __init__(self, req: SimulationRequest):
        gscale = {"permille": 1e-3, "percent": 1e-2, "ratio": 1.0}[req.grade_unit]
        vscale = 1.0 / 3.6 if req.speed_unit == "km/h" else 1.0
        self._g_starts = [s.start_m for s in req.grades]
        self._g_vals = [s.value * gscale for s in req.grades]
        self._l_starts = [s.start_m for s in req.limits]
        self._l_ends = [s.end_m for s in req.limits]
        self._l_vals = [s.limit * vscale for s in req.limits]
        self.start_m = req.grades[0].start_m
        self.end_m = req.grades[-1].end_m

    def grade_ratio(self, s: float) -> float:
        """里程 s 处坡度（ratio）；界外取最近端点值（外延）。"""
        i = bisect_right(self._g_starts, s) - 1
        i = min(max(i, 0), len(self._g_vals) - 1)
        return self._g_vals[i]

    def limit_mps(self, s: float) -> Optional[float]:
        """里程 s 处限速 (m/s)；无覆盖返回 None。"""
        i = bisect_right(self._l_starts, s) - 1
        if i < 0 or s >= self._l_ends[i]:
            return None
        return self._l_vals[i]


def _scalar_adhesion(req: SimulationRequest, scenario) -> float:
    """工况生效的标量黏着：工况级覆盖优先，否则用请求级。"""
    return scenario.adhesion if scenario.adhesion is not None else req.adhesion


def _make_force_model(req: SimulationRequest, track: Track, scenario,
                      curve) -> ForceModel:
    mass_kg = req.vehicle.mass_t * 1000.0
    vscale = 1.0 / 3.6 if req.speed_unit == "km/h" else 1.0
    segs = scenario.adhesion_segments or []
    return ForceModel(
        mass_kg=mass_kg,
        eff_mass_kg=mass_kg * req.vehicle.rotary_inertia_factor,
        curve_speeds=[p.speed * vscale for p in curve.points],
        curve_forces=[p.force_kn * 1000.0 for p in curve.points],
        adhesion_base=_scalar_adhesion(req, scenario),
        force_factor=scenario.brake_force_factor,
        delay_s=scenario.delay_s if scenario.delay_s is not None else req.brake_delay_s,
        buildup_s=req.brake_buildup_s,
        rr_a=req.rolling_resistance.a_n,
        rr_b=req.rolling_resistance.b_n_per_mps,
        rr_c=req.rolling_resistance.c_n_per_mps2,
        grade_at=track.grade_ratio,
        adhesion_starts=[s.start_m for s in segs],
        adhesion_ends=[s.end_m for s in segs],
        adhesion_values=[s.adhesion for s in segs],
    )


def _make_grouped_force_model(req: SimulationRequest, track: Track, scenario,
                              curve_attr: str) -> ForceModel:
    """分组制动传播模型：各组按自身曲线/延迟/建立时间出力。

    工况级 delay_s（若提交）作为附加的统一指令延迟叠加到各组传播延迟上；
    顶层统一曲线/延迟字段保留在模型中，仅用于展示与等效延迟回落。
    """
    base = _make_force_model(req, track, scenario, getattr(req, curve_attr))
    offset = scenario.delay_s if scenario.delay_s is not None else 0.0
    vscale = 1.0 / 3.6 if req.speed_unit == "km/h" else 1.0
    groups: list[BrakeGroupSpec] = []
    for g in req.brake_groups or []:
        curve = getattr(g, curve_attr)
        groups.append(BrakeGroupSpec(
            name=g.name,
            mass_kg=g.mass_t * 1000.0,
            curve_speeds=[p.speed * vscale for p in curve.points],
            curve_forces=[p.force_kn * 1000.0 for p in curve.points],
            delay_s=g.delay_s + offset,
            buildup_s=g.buildup_s,
        ))
    return replace(base, groups=groups)


def _low_mu_zones(model: ForceModel) -> list[dict]:
    """把分段表中黏着低于标量基线的相邻区段合并为连续低黏着区。

    返回 [{start_m, end_m, min_adhesion}]，按里程升序；无分段时为空。
    """
    base = model.adhesion_base
    zones: list[dict] = []
    for st, en, mu in zip(model.adhesion_starts, model.adhesion_ends,
                          model.adhesion_values):
        if mu < base - LOW_MU_EPS:
            if zones and st <= zones[-1]["end_m"] + ZONE_MERGE_EPS:
                zones[-1]["end_m"] = max(zones[-1]["end_m"], en)
                zones[-1]["min_adhesion"] = min(zones[-1]["min_adhesion"], mu)
            else:
                zones.append({"start_m": st, "end_m": en, "min_adhesion": mu})
    return zones


def _traj_speed_at(traj_s: list[float], traj_v: list[float],
                   s: float) -> Optional[float]:
    """轨迹在里程 s 处的速度（线性插值；s 超出轨迹里程返回 None）。"""
    if not traj_s or s < traj_s[0] or s > traj_s[-1]:
        return None
    i = bisect_right(traj_s, s) - 1
    if i >= len(traj_s) - 1:
        return traj_v[-1]
    s0, s1 = traj_s[i], traj_s[i + 1]
    if s1 <= s0:
        return traj_v[i]
    frac = (s - s0) / (s1 - s0)
    return traj_v[i] + frac * (traj_v[i + 1] - traj_v[i])


def _low_adhesion_summary(model: ForceModel, traj: Trajectory,
                          zones: Optional[list[dict]] = None) -> dict:
    """按制动模式汇总低黏着区：进出速度、区内最低减速度、受影响限速点。"""
    if zones is None:
        zones = _low_mu_zones(model)
    if not zones:
        return {
            "active": False,
            "baseline_adhesion": model.adhesion_base,
            "zones": [],
        }

    traj_s = [p["s_m"] for p in traj.points]
    traj_v = [p["v_mps"] for p in traj.points]

    out_zones: list[dict] = []
    for z in zones:
        st, en = z["start_m"], z["end_m"]
        in_pts = [p for p in traj.points
                  if p["s_m"] >= st - 1e-9 and p["s_m"] < en - 1e-9]
        v_entry = _traj_speed_at(traj_s, traj_v, st)
        v_exit = _traj_speed_at(traj_s, traj_v, en)
        entered = v_entry is not None
        exited = v_exit is not None
        info: dict = {
            "start_m": round(st, 2),
            "end_m": round(en, 2),
            "min_adhesion": round(z["min_adhesion"], 4),
            "traversed": entered,
            "entry_speed_kmh": (round(v_entry * 3.6, 3)
                                if v_entry is not None else None),
            "exit_speed_kmh": (round(v_exit * 3.6, 3)
                               if v_exit is not None else None),
            "stopped_inside": False,
            "min_deceleration_mps2": None,
            "min_deceleration_position_m": None,
            "affected_limit_points_m": [],
        }
        # 区间内最低制动减速度：只计实际在制动（a<0）的采样点，
        # 排除制动延迟/建立期与平坡滑行的零减速度。
        brake_pts = [p for p in in_pts if p["a_mps2"] < -1e-9]
        if brake_pts:
            pmin = min(brake_pts, key=lambda p: p["a_mps2"])
            info["min_deceleration_mps2"] = round(-pmin["a_mps2"], 4)
            info["min_deceleration_position_m"] = round(pmin["s_m"], 2)
        # 轨迹在区内终止（停车）则无出口速度
        last = traj.last
        if entered and not exited and st - 1e-9 <= last["s_m"] < en + 1e-9:
            info["stopped_inside"] = traj.stopped
        out_zones.append(info)

    return {
        "active": True,
        "baseline_adhesion": model.adhesion_base,
        "zones": out_zones,
    }


def _group_milestones(model: ForceModel, traj: Trajectory) -> list[dict]:
    """各组开始响应/达到全力的时刻与里程（里程按轨迹 t→s 线性插值）。

    里程按常用制动轨迹计算；若列车停车或积分终止早于某时刻，对应里程为 None。
    """
    ts = [p["t_s"] for p in traj.points]
    ss = [p["s_m"] for p in traj.points]

    def pos_at(t: float) -> Optional[float]:
        if not ts or t < ts[0] or t > ts[-1]:
            return None
        i = bisect_right(ts, t) - 1
        if i >= len(ts) - 1:
            return ss[-1]
        t0, t1 = ts[i], ts[i + 1]
        if t1 <= t0:
            return ss[i]
        frac = (t - t0) / (t1 - t0)
        return ss[i] + frac * (ss[i + 1] - ss[i])

    t_fulls = [g.delay_s + g.buildup_s for g in model.groups or []]
    t_last = max(t_fulls)
    # 并列时取更靠后的车辆组（尾部建立最晚是主要关切）
    last_idx = max(i for i, tf in enumerate(t_fulls) if tf >= t_last - 1e-9)

    out: list[dict] = []
    for i, g in enumerate(model.groups or []):
        t_resp = g.delay_s
        t_full = t_fulls[i]
        s_resp = pos_at(t_resp)
        s_full = pos_at(t_full)
        entry: dict = {
            "name": g.name,
            "mass_t": round(g.mass_kg / 1000.0, 3),
            "delay_s": round(g.delay_s, 3),
            "buildup_s": round(g.buildup_s, 3),
            "response_time_s": round(t_resp, 3),
            "response_position_m": (round(s_resp, 2)
                                    if s_resp is not None else None),
            "full_time_s": round(t_full, 3),
            "full_position_m": (round(s_full, 2)
                                if s_full is not None else None),
            "is_last_to_full": i == last_idx,
        }
        if s_resp is None or s_full is None:
            entry["note"] = "列车停车或积分终止早于该时刻，对应里程不可得"
        out.append(entry)
    return out


def _brake_force_series(model: ForceModel, traj_out: list[dict]) -> list[dict]:
    """全列制动力—时间序列（含各组分解），与输出轨迹点对齐。"""
    names = [g.name for g in model.groups or []]
    series: list[dict] = []
    for p in traj_out:
        total, per = model.brake_force_breakdown(
            p["v_mps"], p["t_s"], p["s_m"])
        series.append({
            "t_s": p["t_s"],
            "s_m": p["s_m"],
            "v_kmh": p["v_kmh"],
            "total_force_kn": round(total / 1000.0, 2),
            "group_forces_kn": {n: round(f / 1000.0, 2)
                                for n, f in zip(names, per)},
        })
    return series


def _limit_points(req: SimulationRequest, track: Track, model: ForceModel,
                  traj: Trajectory, v0_mps: float, zones: list[dict]
                  ) -> list[dict]:
    """对每个限速收紧点，反推最晚制动位置并核对轨迹是否满足限速。

    反推同时使用分段黏着模型与标量基线模型，给出低黏着导致的最晚制动
    位置前移量，并标记反推制动路径穿过的低黏着区。
    """
    out: list[dict] = []
    s0 = req.initial_position_m
    limits = req.limits
    vscale_in = 1.0 / 3.6 if req.speed_unit == "km/h" else 1.0
    vscale_out = 3.6  # 输出统一 km/h
    bw_min = track.start_m - 500.0

    # 标量基线模型：同一工况但全程使用标量黏着
    scalar_model = replace(model, adhesion_starts=[], adhesion_ends=[],
                           adhesion_values=[])

    # 轨迹在任意里程的速度（按里程单调，可二分）
    traj_s = [p["s_m"] for p in traj.points]
    traj_v = [p["v_mps"] for p in traj.points]

    def traj_speed_at(s: float) -> Optional[float]:
        return _traj_speed_at(traj_s, traj_v, s)

    for i, seg in enumerate(limits):
        if seg.start_m <= s0 + 1e-9:
            continue  # 初始位置之前的限速点无制动意义
        lim_mps = seg.limit * vscale_in
        prev_lim = limits[i - 1].limit * vscale_in if i > 0 else v0_mps
        v_app = min(prev_lim, v0_mps)  # 接近速度：前区段限速与初速度的较小者

        s_full, note, bw_min_reached = backward_brake_position(
            model, seg.start_m, lim_mps, v_app, bw_min)
        s_full_base, _, _ = backward_brake_position(
            scalar_model, seg.start_m, lim_mps, v_app, bw_min)

        delay_dist = v_app * model.equivalent_delay_s(v_app)
        latest = None
        latest_base = None
        feasible = False
        if s_full is not None:
            latest = s_full - delay_dist  # 延迟与建立期折算为走行距离
            feasible = latest >= track.start_m
        if s_full_base is not None:
            latest_base = s_full_base - delay_dist

        # 反推制动路径 [bw_min_reached, seg.start_m] 穿过的低黏着区
        hit_zones = [
            z for z in zones
            if z["end_m"] > bw_min_reached and z["start_m"] < seg.start_m
        ]
        # 前移量 = 标量基线最晚制动点 - 分段最晚制动点（>0 表示必须提前制动）
        advance = (round(latest_base - latest, 1)
                   if latest is not None and latest_base is not None else None)

        v_traj = traj_speed_at(seg.start_m)
        out.append({
            "at_m": seg.start_m,
            "limit_kmh": round(lim_mps * vscale_out, 2),
            "approach_speed_kmh": round(v_app * vscale_out, 2),
            "trajectory_speed_kmh": (round(v_traj * vscale_out, 2)
                                     if v_traj is not None else None),
            "latest_brake_m": round(latest, 1) if latest is not None else None,
            "latest_brake_m_scalar_baseline": (round(latest_base, 1)
                                               if latest_base is not None else None),
            "latest_brake_advance_m": advance,
            "delay_distance_m": round(delay_dist, 1),
            "feasible": feasible,
            "low_adhesion_on_braking_path": (
                bool(hit_zones) and note != "no_braking_required"),
            "low_adhesion_zones_m": [
                {"start_m": round(z["start_m"], 2),
                 "end_m": round(z["end_m"], 2)} for z in hit_zones],
            "respected_in_trajectory": (
                None if v_traj is None
                else bool(v_traj <= lim_mps + 0.5 / 3.6)),
            "note": note,
        })
    return out


def _mode_result(req: SimulationRequest, track: Track, model: ForceModel,
                 curve_name: str, v0_mps: float, target_stop: float,
                 s_max: float) -> dict:
    traj = integrate(model, req.initial_position_m, v0_mps, s_max)
    last = traj.last

    # 最大减速度
    decels = [(p["a_mps2"], p["s_m"]) for p in traj.points]
    a_min, s_at_amin = min(decels, key=lambda x: x[0])
    max_decel = max(0.0, -a_min)

    # 超速区间
    limit_kmh_at = lambda s: (None if track.limit_mps(s) is None
                              else track.limit_mps(s) * 3.6)
    violations = scan_violations(traj.points, limit_kmh_at)

    # 停车目标区间
    stop_pos = last["s_m"] if traj.stopped else None
    if stop_pos is not None and stop_pos > target_stop + 1e-6:
        violations.append({
            "kind": "stop_target",
            "start_m": round(target_stop, 1),
            "end_m": round(stop_pos, 1),
            "max_overspeed_kmh": None,
            "limit_kmh": None,
            "note": f"越过目标停车点 {stop_pos - target_stop:.1f} m",
        })
    elif stop_pos is None:
        violations.append({
            "kind": "stop_target",
            "start_m": round(target_stop, 1),
            "end_m": round(last["s_m"], 1),
            "max_overspeed_kmh": None,
            "limit_kmh": None,
            "note": f"未能在计算范围内停车（{traj.reason}），停车目标不可达",
        })

    # 低黏着区段（基于分段黏着与标量基线）
    zones = _low_mu_zones(model)
    low_adhesion = _low_adhesion_summary(model, traj, zones)

    limit_pts = _limit_points(req, track, model, traj, v0_mps, zones)

    # 把受影响限速点（含最晚制动前移量）回填到对应低黏着区
    zone_index = {(z["start_m"], z["end_m"]): z for z in low_adhesion["zones"]}
    for lp in limit_pts:
        for zp in lp["low_adhesion_zones_m"]:
            key = (round(zp["start_m"], 2), round(zp["end_m"], 2))
            zinfo = zone_index.get(key)
            if zinfo is None:
                continue
            zinfo["affected_limit_points_m"].append({
                "at_m": lp["at_m"],
                "latest_brake_m": lp["latest_brake_m"],
                "latest_brake_m_scalar_baseline":
                    lp["latest_brake_m_scalar_baseline"],
                "latest_brake_advance_m": lp["latest_brake_advance_m"],
            })

    traj_out = downsample(traj.points, req.max_trajectory_points)
    for p in traj_out:
        p["v_kmh"] = round(p["v_mps"] * 3.6, 3)
        p["grade_permille"] = round(track.grade_ratio(p["s_m"]) * 1000.0, 2)
        lim = track.limit_mps(p["s_m"])
        p["limit_kmh"] = round(lim * 3.6, 1) if lim is not None else None
        p["adhesion"] = round(p["adhesion"], 4)
        p["t_s"] = round(p["t_s"], 3)
        p["s_m"] = round(p["s_m"], 2)
        p["v_mps"] = round(p["v_mps"], 4)
        p["a_mps2"] = round(p["a_mps2"], 4)

    out = {
        "mode": curve_name,
        "status": traj.status,
        "termination": None if traj.status == "ok" else {
            "position_m": round(last["s_m"], 2),
            "time_s": round(last["t_s"], 2),
            "reason": traj.reason,
        },
        "stopped": traj.stopped,
        "stop_position_m": round(stop_pos, 2) if stop_pos is not None else None,
        "stop_time_s": round(last["t_s"], 2) if traj.stopped else None,
        "stop_distance_m": (round(stop_pos - req.initial_position_m, 2)
                            if stop_pos is not None else None),
        "max_deceleration_mps2": round(max_decel, 4),
        "max_deceleration_position_m": round(s_at_amin, 1),
        "limit_points": limit_pts,
        "low_adhesion": low_adhesion,
        "violations": violations,
        "trajectory": traj_out,
    }
    if model.groups:
        # 分组制动传播：里程碑（由编排层提升到工况级）与全列制动力—时间序列
        out["group_milestones"] = _group_milestones(model, traj)
        out["brake_force_series"] = _brake_force_series(model, traj_out)
    return out


def _brake_propagation(req: SimulationRequest, track: Track, scenario,
                       modes: dict, group_milestones: list[dict],
                       v0_mps: float, target_stop: float, s_max: float) -> dict:
    """分组传播汇总，并以现有统一延迟模型为基线对比停车距离/最晚制动位置/余量。

    统一基线 = 顶层制动力曲线 + 统一 brake_delay_s（工况 delay_s 覆盖），
    即未提交 brake_groups 时的既有行为。
    """
    curves = {"service": req.service_brake, "emergency": req.emergency_brake}
    unified_delay = (scenario.delay_s if scenario.delay_s is not None
                     else req.brake_delay_s)

    vs_unified: dict = {}
    for mode_name, curve in curves.items():
        g = modes[mode_name]
        g_model = _make_grouped_force_model(req, track, scenario,
                                            f"{mode_name}_brake")
        u_model = _make_force_model(req, track, scenario, curve)
        u_traj = integrate(u_model, req.initial_position_m, v0_mps, s_max)
        u_stop_pos = u_traj.last["s_m"] if u_traj.stopped else None
        u_stop_dist = (round(u_stop_pos - req.initial_position_m, 2)
                       if u_stop_pos is not None else None)
        u_margin = (round(target_stop - u_stop_pos, 2)
                    if u_stop_pos is not None else None)
        g_stop_dist = g["stop_distance_m"]
        g_margin = (round(target_stop - g["stop_position_m"], 2)
                    if g["stop_position_m"] is not None else None)

        delta_d = (round(g_stop_dist - u_stop_dist, 2)
                   if g_stop_dist is not None and u_stop_dist is not None
                   else None)
        delta_pct = (round(delta_d / u_stop_dist * 100.0, 2)
                     if delta_d is not None and u_stop_dist else None)
        delta_m = (round(g_margin - u_margin, 2)
                   if g_margin is not None and u_margin is not None else None)

        # 限速点最晚制动位置对比（同一限速点：分组 vs 统一）
        u_zones = _low_mu_zones(u_model)
        u_lps = _limit_points(req, track, u_model, u_traj, v0_mps, u_zones)
        u_by_at = {p["at_m"]: p for p in u_lps}
        lp_cmp: list[dict] = []
        for p in g["limit_points"]:
            u = u_by_at.get(p["at_m"], {})
            lg, lu = p["latest_brake_m"], u.get("latest_brake_m")
            lp_cmp.append({
                "at_m": p["at_m"],
                "latest_brake_grouped_m": lg,
                "latest_brake_unified_m": lu,
                "latest_brake_delta_m": (round(lg - lu, 1)
                                         if lg is not None and lu is not None
                                         else None),
            })

        vs_unified[mode_name] = {
            "stop_distance_grouped_m": g_stop_dist,
            "stop_distance_unified_m": u_stop_dist,
            "stop_distance_delta_m": delta_d,
            "stop_distance_delta_pct": delta_pct,
            "stopping_margin_grouped_m": g_margin,
            "stopping_margin_unified_m": u_margin,
            "stopping_margin_delta_m": delta_m,
            "equivalent_delay_grouped_s": round(
                g_model.equivalent_delay_s(v0_mps), 3),
            "equivalent_delay_unified_s": round(
                u_model.equivalent_delay_s(v0_mps), 3),
            "limit_points": lp_cmp,
        }

    last_group = next((g["name"] for g in group_milestones
                       if g["is_last_to_full"]), None)
    return {
        "groups": group_milestones,
        "last_to_full_group": last_group,
        "max_full_time_s": max(g["full_time_s"] for g in group_milestones),
        "unified_model": {"delay_s": unified_delay,
                          "buildup_s": req.brake_buildup_s},
        "vs_unified": vs_unified,
    }


def run_simulation(req: SimulationRequest, warnings: list[dict]) -> dict:
    track = Track(req)
    vscale = 1.0 / 3.6 if req.speed_unit == "km/h" else 1.0
    v0_mps = req.initial_speed * vscale
    target_stop = req.target_stop_m if req.target_stop_m is not None else track.end_m
    # 计算域：覆盖终点后再留缓冲，保证能判定"无法停车"
    s_max = max(track.end_m, target_stop) + max(2000.0, 0.5 * (track.end_m - track.start_m))

    baseline = next((s for s in req.scenarios if s.is_baseline), req.scenarios[0])

    scenario_results = []
    for sc in req.scenarios:
        # 两种制动模式的合力模型（监督包络也复用这两个模型反推触发曲线）
        if req.brake_groups:
            model_svc = _make_grouped_force_model(
                req, track, sc, "service_brake")
            model_emg = _make_grouped_force_model(
                req, track, sc, "emergency_brake")
        else:
            model_svc = _make_force_model(req, track, sc, req.service_brake)
            model_emg = _make_force_model(req, track, sc, req.emergency_brake)

        modes = {
            "service": _mode_result(
                req, track, model_svc, "service", v0_mps, target_stop, s_max),
            "emergency": _mode_result(
                req, track, model_emg, "emergency", v0_mps, target_stop, s_max),
        }

        # 分组制动传播：里程碑提升到工况级，并生成与统一模型的对比
        propagation = None
        if req.brake_groups:
            svc_ms = modes["service"].pop("group_milestones")
            modes["emergency"].pop("group_milestones", None)
            propagation = _brake_propagation(
                req, track, sc, modes, svc_ms, v0_mps, target_stop, s_max)

        svc = modes["service"]
        margin = (round(target_stop - svc["stop_position_m"], 2)
                  if svc["stop_position_m"] is not None else None)
        parameters: dict = {
            "adhesion": sc.adhesion if sc.adhesion is not None else req.adhesion,
            "adhesion_segments": (
                [{"start_m": seg.start_m, "end_m": seg.end_m,
                  "adhesion": seg.adhesion}
                 for seg in sc.adhesion_segments]
                if sc.adhesion_segments else None),
            "brake_force_factor": sc.brake_force_factor,
            "delay_s": (sc.delay_s if sc.delay_s is not None
                        else req.brake_delay_s),
        }
        if req.brake_groups:
            parameters["brake_groups"] = [
                {"name": g.name, "mass_t": g.mass_t, "delay_s": g.delay_s,
                 "buildup_s": g.buildup_s} for g in req.brake_groups]
            parameters["group_delay_offset_s"] = (
                sc.delay_s if sc.delay_s is not None else 0.0)
        entry = {
            "name": sc.name,
            "is_baseline": sc is baseline,
            "parameters": parameters,
            "stopping_margin_m": margin,
            "modes": modes,
        }
        if propagation is not None:
            entry["brake_propagation"] = propagation

        # 限速监督包络：未提交 supervision 时不输出任何监督字段
        if req.supervision is not None:
            envelope = build_envelope(
                req, track, req.supervision, model_svc, model_emg)
            entry["supervision"] = envelope
            if req.alternative_supervision is not None:
                alt_envelope = build_envelope(
                    req, track, req.alternative_supervision,
                    model_svc, model_emg)
                entry["supervision_comparison"] = compare_envelopes(
                    envelope, alt_envelope)

        scenario_results.append(entry)

    # 相对基准的制动距离变化
    base_modes = next(r for r in scenario_results if r["is_baseline"])["modes"]
    for r in scenario_results:
        delta = {}
        for mode_name in MODES:
            d0 = base_modes[mode_name]["stop_distance_m"]
            d1 = r["modes"][mode_name]["stop_distance_m"]
            if d0 is not None and d1 is not None:
                delta[mode_name] = {
                    "stop_distance_delta_m": round(d1 - d0, 2),
                    "stop_distance_delta_pct": round((d1 - d0) / d0 * 100.0, 2)
                    if d0 else None,
                }
            else:
                delta[mode_name] = {
                    "stop_distance_delta_m": None,
                    "stop_distance_delta_pct": None,
                }
        r["vs_baseline"] = delta

    # 按停车余量升序（余量缺失=无法停车，排最前）
    scenario_results.sort(
        key=lambda r: (r["stopping_margin_m"] is not None,
                       r["stopping_margin_m"]))
    for i, r in enumerate(scenario_results, 1):
        r["rank"] = i

    return {
        "request_id": uuid.uuid4().hex[:12],
        "status": "ok",
        "validation": {"errors": [], "warnings": warnings},
        "track": {"start_m": track.start_m, "end_m": track.end_m,
                  "target_stop_m": target_stop},
        "baseline_scenario": baseline.name,
        "sorting": "scenarios 按常用制动停车余量升序；余量为 null 表示无法停车，排最前",
        "csv_download": "POST /api/simulate/csv（相同请求体，返回 CSV 文件）",
        "scenarios": scenario_results,
    }
