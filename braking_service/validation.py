"""请求级校验：区段断裂/重叠、单位冲突、非物理参数。

返回 (errors, warnings)；errors 非空时接口以 422 拒绝。
每条问题为 {"code", "message", "location"}。
"""

from __future__ import annotations

from .models import SimulationRequest

EPS_M = 1e-6

# 铁路物理常识界限
MAX_GRADE_RATIO = 0.25       # 25% 已远超任何轮轨线路
MAX_RAIL_SPEED_KMH = 500.0   # 轮轨速度上限（含试验车）
TYPICAL_MPS_CEILING = 70.0   # 70 m/s ≈ 252 km/h，超出则疑似单位混淆
MAX_ADHESION_STEEL = 0.45    # 钢轮钢轨黏着系数经验上限


def _issue(code: str, message: str, location: str) -> dict:
    return {"code": code, "message": message, "location": location}


def _check_segments(segs, kind: str, errors: list[dict]) -> None:
    """检查区段断裂（gap）、里程重叠（overlap）与排序。"""
    for i in range(1, len(segs)):
        prev, cur = segs[i - 1], segs[i]
        loc = f"{kind}[{i}]"
        if cur.start_m < prev.start_m - EPS_M:
            errors.append(_issue(
                "segments_not_sorted",
                f"{kind} 区段未按里程升序排列：第 {i} 段起点 {cur.start_m} m "
                f"小于第 {i - 1} 段起点 {prev.start_m} m", loc))
        elif cur.start_m < prev.end_m - EPS_M:
            errors.append(_issue(
                "segment_overlap",
                f"{kind} 区段里程重叠：[{prev.start_m}, {prev.end_m}) 与 "
                f"[{cur.start_m}, {cur.end_m}) 重叠 "
                f"{prev.end_m - cur.start_m:.3f} m", loc))
        elif cur.start_m > prev.end_m + EPS_M:
            errors.append(_issue(
                "segment_gap",
                f"{kind} 区段断裂：{prev.end_m} m 与 {cur.start_m} m 之间存在 "
                f"{cur.start_m - prev.end_m:.3f} m 空缺", loc))


def _check_adhesion_segments(req: SimulationRequest, si: int, segs,
                             errors: list[dict], warnings: list[dict],
                             g_start: float, g_end: float) -> None:
    """校验单个工况的黏着区段：排序/空缺/重叠、覆盖范围与非物理取值。"""
    kind = f"scenarios[{si}].adhesion_segments"
    # 排序、重叠、断裂（与坡度/限速同一套规则与错误码）
    _check_segments(segs, kind, errors)

    for j, seg in enumerate(segs):
        loc = f"{kind}[{j}]"
        # 非物理取值（Pydantic 已限制 (0,1]，这里拦截极端低值等异常输入）
        if seg.adhesion <= 0 or seg.adhesion > 1.0:
            errors.append(_issue(
                "non_physical",
                f"黏着系数 {seg.adhesion} 越出物理区间 (0, 1]",
                f"{loc}.adhesion"))
        elif seg.adhesion > MAX_ADHESION_STEEL:
            warnings.append(_issue(
                "non_physical_suspect",
                f"工况黏着区段 [{seg.start_m}, {seg.end_m}) m 黏着系数 "
                f"{seg.adhesion} 高于钢轮钢轨经验上限 {MAX_ADHESION_STEEL}",
                f"{loc}.adhesion"))

        # 覆盖范围：整段在线路覆盖外为错误，跨界为警告（界外按标量黏着外延）
        if seg.end_m <= g_start - EPS_M or seg.start_m >= g_end + EPS_M:
            errors.append(_issue(
                "adhesion_segment_out_of_coverage",
                f"黏着区段 [{seg.start_m}, {seg.end_m}) m 完全位于线路覆盖 "
                f"[{g_start}, {g_end}] m 之外，对仿真无作用", loc))
        elif seg.start_m < g_start - EPS_M or seg.end_m > g_end + EPS_M:
            warnings.append(_issue(
                "adhesion_segment_partial_coverage",
                f"黏着区段 [{seg.start_m}, {seg.end_m}) m 超出线路覆盖 "
                f"[{g_start}, {g_end}] m，界外部分按标量黏着处理", loc))


def validate_request(req: SimulationRequest) -> tuple[list[dict], list[dict]]:
    errors: list[dict] = []
    warnings: list[dict] = []

    _check_segments(req.grades, "grades", errors)
    _check_segments(req.limits, "limits", errors)

    g_start, g_end = req.grades[0].start_m, req.grades[-1].end_m
    l_start, l_end = req.limits[0].start_m, req.limits[-1].end_m

    # --- 覆盖范围 ---
    if not (g_start - EPS_M <= req.initial_position_m <= g_end + EPS_M):
        errors.append(_issue(
            "position_out_of_coverage",
            f"初始里程 {req.initial_position_m} m 不在坡度覆盖 "
            f"[{g_start}, {g_end}] m 内", "initial_position_m"))
    if abs(l_start - g_start) > EPS_M or abs(l_end - g_end) > EPS_M:
        warnings.append(_issue(
            "coverage_mismatch",
            f"限速覆盖 [{l_start}, {l_end}] m 与坡度覆盖 [{g_start}, {g_end}] m "
            "不一致，未覆盖区段按无限速处理", "limits"))
    if req.target_stop_m is not None:
        if req.target_stop_m <= req.initial_position_m:
            errors.append(_issue(
                "non_physical",
                f"目标停车里程 {req.target_stop_m} m 必须大于初始里程 "
                f"{req.initial_position_m} m", "target_stop_m"))
        elif req.target_stop_m > g_end:
            warnings.append(_issue(
                "target_beyond_coverage",
                f"目标停车里程 {req.target_stop_m} m 超出坡度覆盖终点 {g_end} m，"
                "超出部分按末端坡度外延", "target_stop_m"))

    # --- 单位冲突检查 ---
    to_kmh = 3.6 if req.speed_unit == "m/s" else 1.0
    if req.speed_unit == "m/s":
        for i, seg in enumerate(req.limits):
            if seg.limit > TYPICAL_MPS_CEILING:
                errors.append(_issue(
                    "unit_conflict",
                    f"限速 {seg.limit} m/s（≈{seg.limit * 3.6:.0f} km/h）超出 m/s "
                    "合理范围，疑似把 km/h 当作 m/s 输入", f"limits[{i}].limit"))
        if req.initial_speed > TYPICAL_MPS_CEILING:
            errors.append(_issue(
                "unit_conflict",
                f"初速度 {req.initial_speed} m/s 超出合理范围，疑似单位混淆",
                "initial_speed"))
    else:  # km/h
        for i, seg in enumerate(req.limits):
            if seg.limit > MAX_RAIL_SPEED_KMH:
                errors.append(_issue(
                    "unit_conflict",
                    f"限速 {seg.limit} km/h 超出轮轨速度上限 "
                    f"{MAX_RAIL_SPEED_KMH:.0f} km/h，疑似单位混淆",
                    f"limits[{i}].limit"))
            elif 0 < seg.limit < 5:
                warnings.append(_issue(
                    "unit_suspect",
                    f"限速 {seg.limit} km/h 过低，请确认不是以 m/s 输入",
                    f"limits[{i}].limit"))

    # --- 坡度物理性 ---
    scale = {"permille": 1e-3, "percent": 1e-2, "ratio": 1.0}[req.grade_unit]
    for i, seg in enumerate(req.grades):
        ratio = seg.value * scale
        if abs(ratio) > MAX_GRADE_RATIO:
            errors.append(_issue(
                "non_physical",
                f"坡度 {seg.value} ({req.grade_unit}) = {ratio * 1000:.0f}‰，"
                "超出轮轨线路物理界限，请检查坡度单位", f"grades[{i}].value"))
        elif req.grade_unit == "percent" and abs(seg.value) > 8:
            warnings.append(_issue(
                "unit_suspect",
                f"坡度 {seg.value}% 对铁路异常陡峭，请确认不是把 ‰ 当作 % 输入",
                f"grades[{i}].value"))

    # --- 黏着与制动 ---
    if req.adhesion > MAX_ADHESION_STEEL:
        warnings.append(_issue(
            "non_physical_suspect",
            f"黏着系数 {req.adhesion} 高于钢轮钢轨经验上限 "
            f"{MAX_ADHESION_STEEL}", "adhesion"))
    for i, sc in enumerate(req.scenarios):
        mu = sc.adhesion if sc.adhesion is not None else req.adhesion
        if mu > MAX_ADHESION_STEEL:
            warnings.append(_issue(
                "non_physical_suspect",
                f"工况 '{sc.name}' 黏着系数 {mu} 高于经验上限 "
                f"{MAX_ADHESION_STEEL}", f"scenarios[{i}].adhesion"))
        if sc.adhesion_segments:
            _check_adhesion_segments(
                req, i, sc.adhesion_segments, errors, warnings,
                g_start, g_end)

    # --- 制动力曲线 ---
    for name, curve in (("service_brake", req.service_brake),
                        ("emergency_brake", req.emergency_brake)):
        if max(p.force_kn for p in curve.points) <= 0:
            errors.append(_issue(
                "non_physical", f"{name} 制动力曲线全为零，无法制动", name))
        vmax_curve = max(p.speed for p in curve.points) * to_kmh
        if vmax_curve > MAX_RAIL_SPEED_KMH:
            warnings.append(_issue(
                "unit_suspect",
                f"{name} 曲线最大速度 {vmax_curve:.0f} km/h 异常，请核对速度单位",
                name))
        # 减速度合理性：F/m
        fmax_n = max(p.force_kn for p in curve.points) * 1000.0
        decel = fmax_n / (req.vehicle.mass_t * 1000.0)
        if decel > 3.0:
            warnings.append(_issue(
                "non_physical_suspect",
                f"{name} 最大减速度约 {decel:.2f} m/s²，超出常规轨道车辆范围",
                name))

    # --- 初速度 vs 限速 ---
    v0_kmh = req.initial_speed * to_kmh
    if v0_kmh == 0:
        warnings.append(_issue(
            "non_physical_suspect", "初速度为 0，列车已处于静止状态",
            "initial_speed"))
    lim0 = next((seg.limit for seg in req.limits
                 if seg.start_m <= req.initial_position_m < seg.end_m), None)
    if lim0 is not None and v0_kmh > lim0 * to_kmh + 0.5:
        warnings.append(_issue(
            "initial_overspeed",
            f"初速度 {v0_kmh:.1f} km/h 已超过初始位置限速 "
            f"{lim0 * to_kmh:.1f} km/h，轨迹将包含超速区段", "initial_speed"))

    return errors, warnings
