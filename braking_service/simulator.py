"""仿真编排：把请求转换为内部单位制，逐工况 × 制动模式积分并汇总结果。"""

from __future__ import annotations

import uuid
from bisect import bisect_right
from dataclasses import replace
from typing import Optional

from .models import SimulationRequest
from .physics import (GRAVITY, ForceModel, Trajectory, backward_brake_position,
                      downsample, integrate, scan_violations)

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

        delay_dist = v_app * (model.delay_s + model.buildup_s / 2.0)
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

    return {
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


def run_simulation(req: SimulationRequest, warnings: list[dict]) -> dict:
    track = Track(req)
    vscale = 1.0 / 3.6 if req.speed_unit == "km/h" else 1.0
    v0_mps = req.initial_speed * vscale
    target_stop = req.target_stop_m if req.target_stop_m is not None else track.end_m
    # 计算域：覆盖终点后再留缓冲，保证能判定"无法停车"
    s_max = max(track.end_m, target_stop) + max(2000.0, 0.5 * (track.end_m - track.start_m))

    curves = {"service": req.service_brake, "emergency": req.emergency_brake}

    baseline = next((s for s in req.scenarios if s.is_baseline), req.scenarios[0])

    scenario_results = []
    for sc in req.scenarios:
        modes = {}
        for mode_name, curve in curves.items():
            model = _make_force_model(req, track, sc, curve)
            modes[mode_name] = _mode_result(
                req, track, model, mode_name, v0_mps, target_stop, s_max)

        svc = modes["service"]
        margin = (round(target_stop - svc["stop_position_m"], 2)
                  if svc["stop_position_m"] is not None else None)
        scenario_results.append({
            "name": sc.name,
            "is_baseline": sc is baseline,
            "parameters": {
                "adhesion": sc.adhesion if sc.adhesion is not None else req.adhesion,
                "adhesion_segments": (
                    [{"start_m": seg.start_m, "end_m": seg.end_m,
                      "adhesion": seg.adhesion}
                     for seg in sc.adhesion_segments]
                    if sc.adhesion_segments else None),
                "brake_force_factor": sc.brake_force_factor,
                "delay_s": (sc.delay_s if sc.delay_s is not None
                            else req.brake_delay_s),
            },
            "stopping_margin_m": margin,
            "modes": modes,
        })

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
