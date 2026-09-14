"""限速监督包络：告警/常用/紧急三级速度—里程触发曲线。

沿每个限速收紧点调用 physics.backward_brake_curve（与现有最晚制动位置
反推相同的 RK4 反向步进），生成三条触发曲线：

- 告警（warning）：常用全制动反推曲线，额外前置告警触发延迟；
- 常用（service）：常用全制动反推曲线；
- 紧急（emergency）：紧急全制动反推曲线。

名义触发曲线在性能曲线上再扣减触发延迟的空走距离
``s_trigger(v) = s_perf(v) - v·(t_eq(v) + trigger_delay)``（t_eq 为车辆
制动延迟/建立的等效延迟，分组模型按组力加权，与最晚制动位置口径一致）。

测量误差按不利方向计入（使系统最晚感知、最晚动作）：

- 速度测量误差按"测速偏低"：实际速度 v 时显示 v - Δv，触发边界
  v - Δv = v_nom(s) 故有效触发曲线 ``s_eff(v) = s_nom(v - Δv)``
  （同一实际速度对应的触发位置向限速点方向后移）；
- 里程测量误差按"显示里程偏小"：实际位置 s 时显示 s - Δs（列车实际
  更靠近限速点），故 ``s_eff(v) = s_nom(v) + Δs``；
- 触发延迟按接近速度折算为空走距离，已含在 s_trigger 中。

校核：曲线次序（告警应最靠外、紧急最靠内）、曲线交叉、相邻曲线间的
最小接管距离/时间裕量，以及紧急曲线到限速点的余量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from .models import SimulationRequest, SupervisionConfig
from .physics import ForceModel, backward_brake_curve

LEVELS = ("warning", "service", "emergency")
# 相邻曲线对：(外级, 内级)；emergency 的内级为限速点锚线
PAIRS = (("warning", "service"), ("service", "emergency"),
         ("emergency", "limit_point"))

EPS_M = 1e-6         # 次序/交叉判定里程容差 (m)
ANCHOR = "limit_point"


# ------------------------------------------------------------------- 曲线 ---

@dataclass
class _Curve:
    """单级触发曲线（内部表示，按 v 升序排列的 (v_mps, s_m) 点列）。"""

    level: str
    at_m: float                 # 关联限速收紧点里程
    limit_mps: float
    v_app_mps: float
    # 性能反推曲线（全制动，不含监督延迟/误差），v 升序、s 降序
    perf_vs: list[tuple[float, float]]
    note: str
    min_reached_m: float
    extra_delay_s: float        # 该级触发延迟（warning 为相对常用的额外前置量）
    speed_error_mps: float
    position_error_m: float
    teq_at: Callable[[float], float]  # v -> 车辆等效制动延迟 (s)
    eff_vs: list[tuple[float, float]] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.note == "ok" and self.trigger_at(self.v_app_mps) is not None

    def _perf_s(self, v: float) -> float:
        """性能曲线在速度 v 处的里程（线性插值；界外取端点值）。"""
        vs = self.perf_vs
        if v <= vs[0][0]:
            return vs[0][1]
        if v >= vs[-1][0]:
            return vs[-1][1]
        for i in range(len(vs) - 1):
            v0, s0 = vs[i]
            v1, s1 = vs[i + 1]
            if v0 <= v <= v1:
                if v1 <= v0:
                    return s0
                return s0 + (s1 - s0) * (v - v0) / (v1 - v0)
        return vs[-1][1]

    def nominal_trigger_s(self, v: float) -> float:
        """名义触发位置：性能曲线扣减车辆等效延迟 + 本级触发延迟空走距离。"""
        return self._perf_s(v) - v * (self.teq_at(v) + self.extra_delay_s)

    def trigger_s(self, v: float) -> float:
        """计入不利方向误差后的有效触发位置。

        系统按测量量判断：测速偏低 Δv（v_测 = v - Δv）、里程显示偏小 Δs
        （s_测 = s - Δs，列车实际更靠近限速点）时动作最晚。触发边界
        v - Δv = v_nom(s - Δs) 反解为 s_eff(v) = s_nom(v - Δv) + Δs，
        位置向限速点方向后移（更晚触发）。
        """
        return self.nominal_trigger_s(v - self.speed_error_mps) \
            + self.position_error_m

    def trigger_at(self, v: float,
                   track_start: Optional[float] = None) -> Optional[float]:
        """接近速度处的有效触发位置；不在性能曲线覆盖速度域内返回 None。"""
        if v > self.perf_vs[-1][0] + 1e-9:
            return None
        s = self.trigger_s(v)
        if track_start is not None and s < track_start - EPS_M:
            return None
        return s


def _build_curve(level: str, model: ForceModel, at_m: float, lim_mps: float,
                 v_app_mps: float, bw_min: float, thr,
                 extra_delay_s: float, vscale_in: float) -> _Curve:
    """反推单级性能曲线并构造触发曲线对象。"""
    pts, note, min_s = backward_brake_curve(
        model, at_m, lim_mps, v_app_mps, bw_min)
    # backward 返回 (里程降序, 速度升序)：按 v 升序重排
    pts.sort(key=lambda p: p[1])
    return _Curve(
        level=level, at_m=at_m, limit_mps=lim_mps, v_app_mps=v_app_mps,
        perf_vs=[(v, s) for s, v in pts], note=note, min_reached_m=min_s,
        extra_delay_s=extra_delay_s,
        speed_error_mps=thr.speed_error * vscale_in,
        position_error_m=thr.position_error_m,
        teq_at=model.equivalent_delay_s,
    )


def _sample_curve(c: _Curve, max_points: int) -> list[dict]:
    """输出有效触发曲线采样点（按里程升序，即由远及近接近限速点）。"""
    grid = [v for v, _ in c.perf_vs]
    # 锚点附近与接近速度端点必须保留
    if c.v_app_mps not in grid and c.v_app_mps <= c.perf_vs[-1][0] + 1e-9:
        grid.append(min(c.v_app_mps, c.perf_vs[-1][0]))
    pts = [(c.trigger_s(v), v) for v in grid if v >= c.limit_mps - 1e-9]
    pts.sort(key=lambda p: p[0])  # s 升序
    if len(pts) > max_points:
        stride = math.ceil(len(pts) / max_points)
        sampled = pts[::stride]
        # 末点必须保留；已达上限时替换切片最后一点而不是再追加
        if sampled[-1] != pts[-1]:
            if len(sampled) >= max_points:
                sampled[-1] = pts[-1]
            else:
                sampled.append(pts[-1])
        pts = sampled
    return [{"s_m": round(s, 2), "v_mps": round(v, 4),
             "v_kmh": round(v * 3.6, 3)} for s, v in pts]


# ------------------------------------------------------------------- 校核 ---

def _interp_s(c: Optional[_Curve], v: float, anchor_m: Optional[float] = None
              ) -> float:
    """有效触发曲线在速度 v 处的里程；anchor_m 非空表示限速点锚线。"""
    if anchor_m is not None:
        return anchor_m
    assert c is not None
    return c.trigger_s(v)


def _pair_gap(outer: Optional[_Curve], inner: Optional[_Curve],
              anchor_m: Optional[float], thr) -> dict:
    """扫描一对曲线间的接管裕量、交叉点与最小距离/时间裕量。

    扫描速度域取内级覆盖域（内级在更高速度处不可达即由 infeasible 另行
    报告），网格取两级采样速度的并集。时间裕量 = 距离裕量 / 当前速度。
    """
    lim = (inner.limit_mps if inner is not None else
           (outer.limit_mps if outer is not None else 0.0))
    # 扫描速度网格（m/s，升序）
    v_inner_max = anchor_m if anchor_m is not None else None
    if v_inner_max is None:
        v_max = inner.perf_vs[-1][0]
        grid = sorted({v for c in (outer, inner) for v, _ in c.perf_vs
                       if lim - 1e-9 <= v <= v_max + 1e-9})
    else:
        # 紧急 vs 锚线：用紧急曲线自身速度域
        assert outer is not None
        v_max = outer.perf_vs[-1][0]
        grid = sorted({v for v, _ in outer.perf_vs})

    samples: list[tuple[float, float]] = []  # (v, gap = s_inner - s_outer)
    for v in grid:
        s_o = _interp_s(outer, min(v, outer.perf_vs[-1][0]))
        s_i = _interp_s(inner, min(v, inner.perf_vs[-1][0])
                        if inner is not None else v, anchor_m)
        samples.append((v, s_i - s_o))

    # 最小距离裕量与最小时间裕量（仅在未倒置的样本中统计）
    pos = [(v, g) for v, g in samples if g >= -EPS_M]
    min_v, min_gap = min(pos, key=lambda x: x[1]) if pos else (None, None)
    time_candidates = [(v, g, g / v) for v, g in pos if v > 1e-9]
    tmin_v, _, min_time = (min(time_candidates, key=lambda x: x[2])
                           if time_candidates else (None, None, None))

    # 交叉点：相邻样本间符号由正转负，线性插值求穿越速度
    crossing = None
    for (v0, g0), (v1, g1) in zip(samples, samples[1:]):
        if g0 >= -EPS_M and g1 < -EPS_M:
            frac = g0 / (g0 - g1) if g1 != g0 else 0.0
            vc = v0 + frac * (v1 - v0)
            sc = _interp_s(outer, min(vc, outer.perf_vs[-1][0]))
            crossing = {"v_mps": vc, "outer_s_m": sc,
                        "gap_m": g0 + frac * (g1 - g0)}
            break
        if g0 < -EPS_M and g1 >= -EPS_M and crossing is None:
            # 域起点即倒置：报告域内最严重处
            pass
    if crossing is None:
        worst = min(samples, key=lambda x: x[1])
        if worst[1] < -EPS_M:
            crossing = {"v_mps": worst[0],
                        "outer_s_m": _interp_s(
                            outer, min(worst[0], outer.perf_vs[-1][0])),
                        "gap_m": worst[1]}

    req_d = thr.min_takeover_distance_m
    req_t = thr.min_takeover_time_s
    return {
        "min_distance_m": round(min_gap, 2) if min_gap is not None else None,
        "min_distance_at_kmh": round(min_v * 3.6, 2) if min_v is not None else None,
        "min_time_s": round(min_time, 3) if min_time is not None else None,
        "min_time_at_kmh": round(tmin_v * 3.6, 2) if tmin_v is not None else None,
        "required_distance_m": req_d,
        "required_time_s": req_t,
        "crossing": crossing,
        "distance_shortfall_m": (round(req_d - min_gap, 2)
                                 if min_gap is not None and min_gap < req_d
                                 else 0.0),
        "time_shortfall_s": (round(req_t - min_time, 3)
                             if min_time is not None and min_time < req_t
                             else 0.0),
    }


# ------------------------------------------------------------------- 包络 ---

def _thresholds_by_level(cfg: SupervisionConfig) -> dict:
    return {"warning": cfg.warning, "service": cfg.service,
            "emergency": cfg.emergency}


def _config_echo(cfg: SupervisionConfig, vscale_in: float) -> dict:
    levels = {}
    for lv in LEVELS:
        t = _thresholds_by_level(cfg)[lv]
        levels[lv] = {
            "speed_error": t.speed_error,
            "speed_error_kmh": round(t.speed_error * vscale_in * 3.6, 3),
            "position_error_m": t.position_error_m,
            "trigger_delay_s": t.trigger_delay_s,
            "min_takeover_distance_m": t.min_takeover_distance_m,
            "min_takeover_time_s": t.min_takeover_time_s,
        }
    return {"name": cfg.name, "levels": levels,
            "max_curve_points": cfg.max_curve_points}


def build_envelope(req: SimulationRequest, track, cfg: SupervisionConfig,
                   service_model: ForceModel,
                   emergency_model: ForceModel) -> dict:
    """对单个工况生成三级监督包络、校核问题与最小接管裕量汇总。"""
    vscale_in = 1.0 / 3.6 if req.speed_unit == "km/h" else 1.0
    s0 = req.initial_position_m
    bw_min = track.start_m - 500.0
    thr = _thresholds_by_level(cfg)
    models = {"warning": service_model, "service": service_model,
              "emergency": emergency_model}

    lp_out: list[dict] = []
    all_problems: list[dict] = []
    min_takeover: dict[str, dict] = {}

    for i, seg in enumerate(req.limits):
        if seg.start_m <= s0 + 1e-9:
            continue
        lim_mps = seg.limit * vscale_in
        prev_lim = req.limits[i - 1].limit * vscale_in if i > 0 \
            else req.initial_speed * vscale_in
        v_app = min(prev_lim, req.initial_speed * vscale_in)
        if lim_mps >= v_app:
            lp_out.append({
                "at_m": seg.start_m,
                "limit_kmh": round(lim_mps * 3.6, 2),
                "approach_speed_kmh": round(v_app * 3.6, 2),
                "skipped": True,
                "note": "no_braking_required: 该限速点非收紧点，监督包络不适用",
            })
            continue

        # 告警延迟 = 常用车辆等效延迟之外，额外前置 warning.trigger_delay_s
        curves: dict[str, _Curve] = {}
        for lv in LEVELS:
            curves[lv] = _build_curve(
                lv, models[lv], seg.start_m, lim_mps, v_app, bw_min,
                thr[lv], thr[lv].trigger_delay_s, vscale_in)

        checks: list[dict] = []
        cur_problems: list[dict] = []

        # 1) 单曲线可行性（反推不可达 / 触发点越出线路起点）
        for lv in LEVELS:
            c = curves[lv]
            trig = c.trigger_at(v_app)
            feasible = c.note == "ok" and trig is not None
            if not feasible:
                reason = c.note
                if c.note == "ok":
                    reason = (f"track_start_reached: 计入误差与触发延迟后，"
                              f"{lv} 触发位置在接近速度处越出线路起点 "
                              f"{track.start_m:.0f} m")
                prob = {
                    "code": "curve_infeasible",
                    "level": lv, "pair": None,
                    "at_m": seg.start_m,
                    "limit_kmh": round(lim_mps * 3.6, 2),
                    "problem_mileage_m": round(c.min_reached_m, 1),
                    "related_speed_kmh": round(v_app * 3.6, 2),
                    "min_distance_m": None, "min_time_s": None,
                    "reason": reason,
                }
                checks.append(prob)
                cur_problems.append(prob)

        # 2) 相邻曲线对：次序/交叉/裕量
        for outer_lv, inner_lv in PAIRS:
            outer = curves[outer_lv]
            anchor = seg.start_m if inner_lv == ANCHOR else None
            inner = None if anchor is not None else curves[inner_lv]
            # 要求裕量按外级阈值配置（外级触发后应留给驾驶员/下一级的空间）
            gap = _pair_gap(outer, inner, anchor, thr[outer_lv])
            pair_name = f"{outer_lv}_to_{inner_lv}"
            entry = {"pair": pair_name, **gap}
            checks.append(entry)

            cr = gap["crossing"]
            if cr is not None:
                prob = {
                    "code": "curve_crossing",
                    "level": outer_lv, "pair": pair_name,
                    "at_m": seg.start_m,
                    "limit_kmh": round(lim_mps * 3.6, 2),
                    "problem_mileage_m": round(cr["outer_s_m"], 1),
                    "related_speed_kmh": round(cr["v_mps"] * 3.6, 2),
                    "min_distance_m": round(cr["gap_m"], 2),
                    "min_time_s": None,
                    "required_distance_m": gap["required_distance_m"],
                    "required_time_s": gap["required_time_s"],
                    "reason": (
                        f"{outer_lv} 曲线在 {cr['v_mps'] * 3.6:.1f} km/h 处"
                        f"越过（交叉）内级 {inner_lv} 曲线，监督次序不成立："
                        "外级未先触发即可能直接进入更强制动级"),
                }
                cur_problems.append(prob)
            else:
                # 最小裕量汇总（跨限速点再取最小）
                prev = min_takeover.get(pair_name)
                if (gap["min_distance_m"] is not None and (
                        prev is None or gap["min_distance_m"] < prev["distance_m"])):
                    min_takeover[pair_name] = {
                        "distance_m": gap["min_distance_m"],
                        "distance_at_kmh": gap["min_distance_at_kmh"],
                        "time_s": gap["min_time_s"],
                        "time_at_kmh": gap["min_time_at_kmh"],
                        "at_m": seg.start_m,
                    }
                elif prev is not None and gap["min_time_s"] is not None \
                        and gap["min_time_s"] < prev["time_s"]:
                    prev["time_s"] = gap["min_time_s"]
                    prev["time_at_kmh"] = gap["min_time_at_kmh"]
                if gap["distance_shortfall_m"] > 0 or \
                        gap["time_shortfall_s"] > 0:
                    reasons = []
                    if gap["distance_shortfall_m"] > 0:
                        reasons.append(
                            f"距离裕量 {gap['min_distance_m']:.1f} m < 要求 "
                            f"{gap['required_distance_m']:.0f} m（差 "
                            f"{gap['distance_shortfall_m']:.1f} m）")
                    if gap["time_shortfall_s"] > 0:
                        reasons.append(
                            f"时间裕量 {gap['min_time_s']:.2f} s < 要求 "
                            f"{gap['required_time_s']:.2f} s（差 "
                            f"{gap['time_shortfall_s']:.2f} s）")
                    prob = {
                        "code": "takeover_margin_insufficient",
                        "level": outer_lv, "pair": pair_name,
                        "at_m": seg.start_m,
                        "limit_kmh": round(lim_mps * 3.6, 2),
                        "problem_mileage_m": round(outer.trigger_s(
                            gap["min_distance_at_kmh"] / 3.6), 1),
                        "related_speed_kmh": gap["min_distance_at_kmh"],
                        "min_distance_m": gap["min_distance_m"],
                        "min_time_s": gap["min_time_s"],
                        "required_distance_m": gap["required_distance_m"],
                        "required_time_s": gap["required_time_s"],
                        "reason": f"{outer_lv}→{inner_lv} 接管裕量不足："
                                  + "；".join(reasons),
                    }
                    cur_problems.append(prob)

        # 接近速度处的触发位置与触发区间
        trig_pos: dict[str, Optional[float]] = {}
        for lv in LEVELS:
            trig_pos[lv] = curves[lv].trigger_at(v_app)
        intervals = {
            "warning_window_m": _wd(trig_pos["warning"], trig_pos["service"]),
            "service_window_m": _wd(trig_pos["service"], trig_pos["emergency"]),
            "emergency_window_m": _wd(trig_pos["emergency"], seg.start_m),
        }

        curves_out = {}
        for lv in LEVELS:
            c = curves[lv]
            t = thr[lv]
            curves_out[lv] = {
                "feasible": c.note == "ok" and trig_pos[lv] is not None,
                "note": c.note,
                "nominal_trigger_at_approach_m": (
                    round(c.nominal_trigger_s(v_app), 1)
                    if c.note == "ok" else None),
                "trigger_at_approach_m": (round(trig_pos[lv], 1)
                                          if trig_pos[lv] is not None else None),
                "trigger_delay_distance_m": round(
                    v_app * (c.teq_at(v_app) + t.trigger_delay_s), 1),
                "supervision_delay_distance_m": round(
                    v_app * t.trigger_delay_s, 1),
                "speed_error_shift_m": round(
                    c.nominal_trigger_s(v_app - c.speed_error_mps)
                    - c.nominal_trigger_s(v_app), 1),
                "position_error_shift_m": round(c.position_error_m, 1),
                "curve": _sample_curve(c, cfg.max_curve_points),
            }

        lp_out.append({
            "at_m": seg.start_m,
            "limit_kmh": round(lim_mps * 3.6, 2),
            "approach_speed_kmh": round(v_app * 3.6, 2),
            "skipped": False,
            "curves": curves_out,
            "trigger_positions_m": {lv: (round(p, 1) if p is not None else None)
                                    for lv, p in trig_pos.items()},
            "trigger_intervals_m": intervals,
            "checks": checks,
            "problems": cur_problems,
        })
        all_problems.extend(cur_problems)

    return {
        "config": _config_echo(cfg, vscale_in),
        "limit_points": lp_out,
        "min_takeover": {
            pair: {**vals, "distance_m": round(vals["distance_m"], 2),
                   "time_s": round(vals["time_s"], 3)}
            for pair, vals in min_takeover.items()
        },
        "problems": all_problems,
        "problem_count": len(all_problems),
    }


def _wd(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """两级触发位置之差（b - a，内级减外级）；任一缺失为 None。"""
    return round(b - a, 1) if a is not None and b is not None else None


# ------------------------------------------------------------------- 对比 ---

def _problem_key(p: dict) -> tuple:
    return (p["code"], p["pair"], p["at_m"])


def compare_envelopes(reference: dict, alternative: dict) -> dict:
    """对比两套阈值配置：触发区间/位置变化、接管空间变化、问题新增与消除。"""
    ref_pts = {p["at_m"]: p for p in reference["limit_points"]
               if not p.get("skipped")}
    alt_pts = {p["at_m"]: p for p in alternative["limit_points"]
               if not p.get("skipped")}
    ref_probs = {_problem_key(p): p for p in reference["problems"]}
    alt_probs = {_problem_key(p): p for p in alternative["problems"]}

    pts_out: list[dict] = []
    trigger_deltas: list[float] = []
    takeover_dist_deltas: list[float] = []
    takeover_time_deltas: list[float] = []

    for at_m in sorted(set(ref_pts) | set(alt_pts)):
        rp, ap = ref_pts.get(at_m), alt_pts.get(at_m)
        triggers = {}
        for lv in LEVELS:
            r = (rp or {}).get("trigger_positions_m", {}).get(lv)
            a = (ap or {}).get("trigger_positions_m", {}).get(lv)
            delta = round(a - r, 1) if r is not None and a is not None else None
            if delta is not None:
                trigger_deltas.append(delta)
            triggers[lv] = {"reference_m": r, "alternative_m": a,
                            "delta_m": delta}

        intervals = {}
        for key in ("warning_window_m", "service_window_m",
                    "emergency_window_m"):
            r = (rp or {}).get("trigger_intervals_m", {}).get(key)
            a = (ap or {}).get("trigger_intervals_m", {}).get(key)
            intervals[key] = {
                "reference_m": r, "alternative_m": a,
                "delta_m": round(a - r, 1)
                if r is not None and a is not None else None,
            }

        # 最小接管裕量变化（从 checks 中按 pair 提取）
        takeover = {}
        r_checks = {c["pair"]: c for c in (rp or {}).get("checks", [])
                    if "pair" in c and "min_distance_m" in c}
        a_checks = {c["pair"]: c for c in (ap or {}).get("checks", [])
                    if "pair" in c and "min_distance_m" in c}
        for pair in sorted(set(r_checks) | set(a_checks)):
            rc, ac = r_checks.get(pair, {}), a_checks.get(pair, {})
            rd, ad = rc.get("min_distance_m"), ac.get("min_distance_m")
            rt, at_ = rc.get("min_time_s"), ac.get("min_time_s")
            dd = round(ad - rd, 1) if rd is not None and ad is not None else None
            dt = round(at_ - rt, 3) if rt is not None and at_ is not None else None
            if dd is not None:
                takeover_dist_deltas.append(dd)
            if dt is not None:
                takeover_time_deltas.append(dt)
            takeover[pair] = {
                "distance_reference_m": rd, "distance_alternative_m": ad,
                "distance_delta_m": dd,
                "time_reference_s": rt, "time_alternative_s": at_,
                "time_delta_s": dt,
            }

        resolved = [ref_probs[k] for k in
                    set(ref_probs) - set(alt_probs)
                    if ref_probs[k]["at_m"] == at_m]
        new = [alt_probs[k] for k in
               set(alt_probs) - set(ref_probs)
               if alt_probs[k]["at_m"] == at_m]
        pts_out.append({
            "at_m": at_m,
            "limit_kmh": (ap or rp)["limit_kmh"],
            "trigger_positions": triggers,
            "trigger_intervals": intervals,
            "takeover": takeover,
            "problems_resolved": [_brief_problem(p) for p in resolved],
            "problems_new": [_brief_problem(p) for p in new],
        })

    return {
        "reference_config": reference["config"]["name"],
        "alternative_config": alternative["config"]["name"],
        "limit_points": pts_out,
        "summary": {
            "trigger_position_delta_m": {
                "max_advance_m": round(min(trigger_deltas), 1)
                if trigger_deltas else None,
                "max_postpone_m": round(max(trigger_deltas), 1)
                if trigger_deltas else None,
            },
            "min_takeover_distance_delta_m": (
                round(min(takeover_dist_deltas), 1)
                if takeover_dist_deltas else None),
            "min_takeover_time_delta_s": (
                round(min(takeover_time_deltas), 3)
                if takeover_time_deltas else None),
            "new_problem_count": len(set(alt_probs) - set(ref_probs)),
            "resolved_problem_count": len(set(ref_probs) - set(alt_probs)),
        },
    }


def _brief_problem(p: dict) -> dict:
    return {"code": p["code"], "pair": p["pair"], "at_m": p["at_m"],
            "reason": p["reason"]}


# ------------------------------------------------------- CSV 轨迹阈值查询 ---

def threshold_speeds_kmh(envelope: dict, s_m: float) -> Optional[dict]:
    """按里程从监督包络查三级触发速度 (km/h)，供轨迹 CSV 标注。

    取该里程前方最近的限速收紧点：s 位于其某级曲线里程域内时，线性插值
    该级触发速度；已越过收紧点或曲线不覆盖该里程时对应级为 None。
    """
    point = None
    for lp in envelope["limit_points"]:
        if lp.get("skipped"):
            continue
        if lp["at_m"] + 1e-9 >= s_m:
            point = lp
            break
    if point is None:
        return None
    out: dict[str, Optional[float]] = {}
    for lv in LEVELS:
        pts = point["curves"][lv]["curve"]  # s 升序、v 降序
        ss = [p["s_m"] for p in pts]
        if not ss or s_m < ss[0] - 1e-9 or s_m > ss[-1] + 1e-9:
            out[lv] = None
            continue
        found = None
        for i in range(len(ss) - 1):
            if ss[i] - 1e-9 <= s_m <= ss[i + 1] + 1e-9:
                s0, s1 = ss[i], ss[i + 1]
                v0 = pts[i]["v_kmh"]
                v1 = pts[i + 1]["v_kmh"]
                if s1 <= s0:
                    found = v0
                else:
                    found = v0 + (v1 - v0) * (s_m - s0) / (s1 - s0)
                break
        if found is None:  # 数值容差外的端点
            found = pts[0]["v_kmh"] if abs(s_m - ss[0]) <= 1e-6 \
                else pts[-1]["v_kmh"]
        out[lv] = round(found, 2)
    return out
