"""FastAPI 入口：POST /api/simulate（JSON）与 /api/simulate/csv（CSV 下载）。"""

from __future__ import annotations

import copy
import csv
import io
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from . import __version__
from .models import SimulationRequest
from .simulator import MODES, run_simulation
from .supervision import LEVELS as SUPERVISION_LEVELS, threshold_speeds_kmh
from .validation import validate_request

# ---------------------------------------------------------------- 示例请求 ---

EXAMPLE_REQUEST: dict[str, Any] = {
    "speed_unit": "km/h",
    "grade_unit": "permille",
    "grades": [
        {"start_m": 0, "end_m": 1500, "value": 0},
        {"start_m": 1500, "end_m": 3000, "value": -12},
        {"start_m": 3000, "end_m": 5000, "value": 5},
    ],
    "limits": [
        {"start_m": 0, "end_m": 1200, "limit": 120},
        {"start_m": 1200, "end_m": 2600, "limit": 80},
        {"start_m": 2600, "end_m": 5000, "limit": 100},
    ],
    "vehicle": {"mass_t": 420, "rotary_inertia_factor": 1.06},
    "initial_position_m": 0,
    "initial_speed": 120,
    "service_brake": {
        "points": [
            {"speed": 0, "force_kn": 360},
            {"speed": 50, "force_kn": 380},
            {"speed": 80, "force_kn": 390},
            {"speed": 120, "force_kn": 400},
            {"speed": 160, "force_kn": 410},
        ]
    },
    "emergency_brake": {
        "points": [
            {"speed": 0, "force_kn": 500},
            {"speed": 50, "force_kn": 520},
            {"speed": 80, "force_kn": 535},
            {"speed": 120, "force_kn": 550},
            {"speed": 160, "force_kn": 560},
        ]
    },
    "brake_delay_s": 1.5,
    "brake_buildup_s": 1.0,
    "adhesion": 0.15,
    "rolling_resistance": {"a_n": 4000, "b_n_per_mps": 80, "c_n_per_mps2": 6},
    "scenarios": [
        {"name": "dry_rail", "is_baseline": True},
        {"name": "wet_rail", "adhesion": 0.08, "brake_force_factor": 0.9},
        {"name": "partial_brake_failure", "brake_force_factor": 0.6,
         "delay_s": 2.5},
        {
            "name": "leaf_film_tunnel",
            "adhesion_segments": [
                # 制动中驶入 250 m 处的落叶低黏着区，450 m 处驶出后黏着恢复；
                # 900~1100 m 的隧道渗水点不在本次走行轨迹内，但位于
                # 1200 m 限速点的反推制动路径上，最晚制动位置因此前移。
                {"start_m": 250, "end_m": 450, "adhesion": 0.06},
                {"start_m": 450, "end_m": 900, "adhesion": 0.15},
                {"start_m": 900, "end_m": 1100, "adhesion": 0.06},
            ],
        },
    ],
    "target_stop_m": 1000,
    "max_trajectory_points": 300,
}


def _grouped_example() -> dict[str, Any]:
    """长编组分组制动传播示例：420 t 列车分为头/中/尾三组。

    各组曲线按质量份额拆分（合计与顶层曲线一致），延迟 0.5/1.5/2.5 s
    沿列车由前向后递增；统一模型（delay 1.5 s）将低估尾部晚建立
    制动力带来的空走距离。
    """
    req = copy.deepcopy(EXAMPLE_REQUEST)
    req["scenarios"] = [
        {"name": "dry_rail", "is_baseline": True},
        {"name": "wet_rail", "adhesion": 0.08},
        {"name": "slow_command", "delay_s": 1.0},
    ]
    svc = [(0, 360), (50, 380), (80, 390), (120, 400), (160, 410)]
    emg = [(0, 500), (50, 520), (80, 535), (120, 550), (160, 560)]

    def curve(points: list[tuple[float, float]], share: float) -> dict:
        return {"points": [{"speed": v, "force_kn": round(f * share, 1)}
                           for v, f in points]}

    req["brake_groups"] = [
        {"name": "head", "mass_t": 60, "delay_s": 0.5, "buildup_s": 1.0,
         "service_brake": curve(svc, 60 / 420),
         "emergency_brake": curve(emg, 60 / 420)},
        {"name": "middle", "mass_t": 240, "delay_s": 1.5, "buildup_s": 1.0,
         "service_brake": curve(svc, 240 / 420),
         "emergency_brake": curve(emg, 240 / 420)},
        {"name": "tail", "mass_t": 120, "delay_s": 2.5, "buildup_s": 1.0,
         "service_brake": curve(svc, 120 / 420),
         "emergency_brake": curve(emg, 120 / 420)},
    ]
    return req


EXAMPLE_GROUPED_REQUEST: dict[str, Any] = _grouped_example()


def _supervision_example() -> dict[str, Any]:
    """限速监督包络示例：三级阈值 + 新旧两套配置对比。

    名义配置（nominal）误差与延迟较小；老化设备配置（aged_equipment）
    测速/定位误差更大、系统反应更慢，对比可见三级触发位置向限速点方向
    后移、接管窗口收窄；大下坡段前的 1200 m 收紧点还可能出现
    告警曲线不可达或接管裕量不足。
    """
    req = copy.deepcopy(EXAMPLE_REQUEST)
    req["scenarios"] = [
        {"name": "dry_rail", "is_baseline": True},
        {"name": "wet_rail", "adhesion": 0.08, "brake_force_factor": 0.9},
        {
            "name": "leaf_film_tunnel",
            "adhesion_segments": [
                {"start_m": 250, "end_m": 450, "adhesion": 0.06},
                {"start_m": 450, "end_m": 900, "adhesion": 0.15},
                {"start_m": 900, "end_m": 1100, "adhesion": 0.06},
            ],
        },
    ]
    req["target_stop_m"] = 1000
    req["supervision"] = {
        "name": "nominal",
        "warning": {"speed_error": 2.0, "position_error_m": 5.0,
                    "trigger_delay_s": 4.0,
                    "min_takeover_distance_m": 30.0,
                    "min_takeover_time_s": 1.0},
        "service": {"speed_error": 2.0, "position_error_m": 5.0,
                    "trigger_delay_s": 1.2,
                    "min_takeover_distance_m": 10.0,
                    "min_takeover_time_s": 0.5},
        "emergency": {"speed_error": 2.0, "position_error_m": 5.0,
                      "trigger_delay_s": 0.6,
                      "min_takeover_distance_m": 0.0,
                      "min_takeover_time_s": 0.0},
    }
    req["alternative_supervision"] = {
        "name": "aged_equipment",
        "warning": {"speed_error": 5.0, "position_error_m": 15.0,
                    "trigger_delay_s": 6.0,
                    "min_takeover_distance_m": 30.0,
                    "min_takeover_time_s": 1.0},
        "service": {"speed_error": 5.0, "position_error_m": 15.0,
                    "trigger_delay_s": 2.5,
                    "min_takeover_distance_m": 10.0,
                    "min_takeover_time_s": 0.5},
        "emergency": {"speed_error": 5.0, "position_error_m": 15.0,
                      "trigger_delay_s": 1.2,
                      "min_takeover_distance_m": 0.0,
                      "min_takeover_time_s": 0.0},
    }
    return req


EXAMPLE_SUPERVISION_REQUEST: dict[str, Any] = _supervision_example()

# ------------------------------------------------------------------- 应用 ---

app = FastAPI(
    title="列车制动仿真 API",
    version=__version__,
    description=(
        "接收按里程排列的坡度/限速区段、车辆与制动参数，"
        "用分段 RK4 积分生成常用/紧急制动的速度—里程轨迹；"
        "校验区段断裂、里程重叠、单位冲突与非物理参数；"
        "输出每个限速点的最晚制动位置、停车余量、最大减速度与超速区间；"
        "支持每工况提交按里程连续的 adhesion_segments 表达落叶/渗水等"
        "局部低黏着区（缺省沿用标量 adhesion），汇总列车进出低黏着区的速度、"
        "区内最低减速度、受影响限速点与最晚制动位置前移量；"
        "支持 brake_groups 分组制动传播：按列车前后顺序提交车辆组的质量、"
        "制动力曲线、指令传播延迟与建立时间，积分按各组实际生效时刻汇总"
        "制动力（仍受沿线黏着上限约束），输出各组开始响应/达到全力的时刻"
        "与里程、全列制动力—时间序列、最后建立的车辆组，并以现有统一延迟"
        "模型为基线对比停车距离、限速点最晚制动位置与停车余量；"
        "支持多工况对比（干轨/湿轨/部分失效），按停车余量排序。"
        "提交 supervision 后，沿每个限速收紧点以 RK4 反向求解生成告警/常用/"
        "紧急三条速度—里程触发曲线，按不利方向计入速度/里程测量误差与触发"
        "延迟，校核曲线次序、交叉与最小接管裕量，返回问题里程、关联限速点、"
        "最小距离与时间裕量及原因；可同时提交 alternative_supervision 对比"
        "两套阈值配置的触发区间与接管空间变化。未提交监督配置时输出结构不变。"
    ),
)

OPENAPI_EXAMPLES = {
    "four_scenarios": {
        "summary": "四工况（干轨/湿轨/部分失效/局部低黏着）完整示例",
        "description": "5 km 线路，含下坡段与两级限速收紧；目标停车点 1000 m，"
                       "部分制动失效工况将错过停车目标。leaf_film_tunnel 工况以 "
                       "adhesion_segments 表达 250~450 m 落叶低黏着区（制动中驶入、"
                       "驶出后恢复）与 900~1100 m 隧道渗水点（影响 1200 m 限速点的"
                       "最晚制动位置）。",
        "value": EXAMPLE_REQUEST,
    },
    "grouped_propagation": {
        "summary": "长编组分组制动传播（头/中/尾三组，延迟递增）",
        "description": "420 t 列车分为 head/middle/tail 三组，指令传播延迟 "
                       "0.5/1.5/2.5 s；各组曲线按质量份额拆分（合计与顶层曲线"
                       "一致），对比统一延迟模型可见尾部晚建立制动力带来的额外"
                       "空走距离：停车距离变长、限速点最晚制动位置前移、停车"
                       "余量减小。slow_command 工况演示工况级 delay_s 作为统一"
                       "附加延迟叠加到各组传播延迟上。",
        "value": EXAMPLE_GROUPED_REQUEST,
    },
    "speed_supervision": {
        "summary": "限速监督包络（三级触发曲线 + 两套阈值配置对比）",
        "description": "告警/常用/紧急三级阈值分别配置速度误差、里程误差、"
                       "触发延迟与最小接管裕量；主配置 nominal 与老化设备 "
                       "aged_equipment 对比，展示触发位置后移、接管窗口收窄 "
                       "与新增/消除的监督问题。",
        "value": EXAMPLE_SUPERVISION_REQUEST,
    },
}


def _execute(req: SimulationRequest) -> dict:
    errors, warnings = validate_request(req)
    if errors:
        raise HTTPException(
            status_code=422,
            detail={"message": "请求校验失败", "errors": errors,
                    "warnings": warnings},
        )
    return run_simulation(req, warnings)


# ------------------------------------------------------- CSV 监督列辅助 ---

SUPERVISION_SUMMARY_COLUMNS = [
    "supervision_config", "supervision_problem_count",
    "sup_min_takeover_warning_service_m",
    "sup_min_takeover_service_emergency_m",
    "sup_min_takeover_emergency_limit_m",
    "sup_min_takeover_warning_service_s",
    "sup_min_takeover_service_emergency_s",
    "sup_trigger_warning_m", "sup_trigger_service_m",
    "sup_trigger_emergency_m",
    "sup_window_warning_m", "sup_window_service_m",
    "sup_window_emergency_m",
    "sup_alt_config", "sup_alt_trigger_delta_min_m",
    "sup_alt_takeover_distance_delta_min_m",
    "sup_alt_takeover_time_delta_min_s",
    "sup_alt_new_problem_count", "sup_alt_resolved_problem_count",
]


def _agg(values):
    """数值列表的最小值（None 视为缺失）；全缺失返回空串。"""
    vals = [v for v in values if v is not None]
    return round(min(vals), 2) if vals else ""


def _supervision_summary_cells(sc: dict) -> list:
    """工况级监督包络汇总（同一工况的 service/emergency 两行取值相同）。"""
    sup = sc.get("supervision")
    if sup is None:
        return [""] * len(SUPERVISION_SUMMARY_COLUMNS)
    pts = [p for p in sup["limit_points"] if not p.get("skipped")]
    mt = sup["min_takeover"]

    def mtv(pair: str, key: str):
        return mt.get(pair, {}).get(key)

    triggers = {lv: [p["trigger_positions_m"][lv] for p in pts]
                for lv in SUPERVISION_LEVELS}
    windows = {k: [p["trigger_intervals_m"][k] for p in pts] for k in
               ("warning_window_m", "service_window_m",
                "emergency_window_m")}

    cmp_ = sc.get("supervision_comparison")
    if cmp_ is not None:
        trig_deltas = [t["delta_m"] for p in cmp_["limit_points"]
                       for t in p["trigger_positions"].values()]
        to_d = [c["distance_delta_m"] for p in cmp_["limit_points"]
                for c in p["takeover"].values()]
        tt_d = [c["time_delta_s"] for p in cmp_["limit_points"]
                for c in p["takeover"].values()]
        cells = [
            cmp_["alternative_config"],
            _agg(trig_deltas),
            _agg(to_d),
            _agg(tt_d),
            cmp_["summary"]["new_problem_count"],
            cmp_["summary"]["resolved_problem_count"],
        ]
    else:
        cells = ["", "", "", "", "", ""]

    return [
        sup["config"]["name"], sup["problem_count"],
        mtv("warning_to_service", "distance_m") or "",
        mtv("service_to_emergency", "distance_m") or "",
        mtv("emergency_to_limit_point", "distance_m") or "",
        mtv("warning_to_service", "time_s") or "",
        mtv("service_to_emergency", "time_s") or "",
        _agg(triggers["warning"]),
        _agg(triggers["service"]),
        _agg(triggers["emergency"]),
        _agg(windows["warning_window_m"]),
        _agg(windows["service_window_m"]),
        _agg(windows["emergency_window_m"]),
        *cells,
    ]


@app.post(
    "/api/simulate",
    summary="运行制动仿真，返回结构化 JSON",
    tags=["simulation"],
)
def simulate(req: SimulationRequest = Body(
        openapi_examples=OPENAPI_EXAMPLES)) -> dict:
    return _execute(req)


@app.post(
    "/api/simulate/csv",
    summary="运行仿真并下载 CSV",
    description="kind=trajectory 输出全部轨迹点；kind=summary 输出工况汇总。",
    tags=["simulation"],
)
def simulate_csv(
    kind: str = Query("trajectory", enum=["trajectory", "summary"]),
    req: SimulationRequest = Body(openapi_examples=OPENAPI_EXAMPLES),
) -> StreamingResponse:
    result = _execute(req)

    buf = io.StringIO()
    writer = csv.writer(buf)
    has_supervision = any("supervision" in sc for sc in result["scenarios"])
    if kind == "summary":
        header = [
            "rank", "scenario", "is_baseline", "adhesion",
            "adhesion_segments",
            "brake_force_factor", "delay_s", "mode", "status",
            "stop_position_m", "stop_distance_m", "stopping_margin_m",
            "max_deceleration_mps2",
            "low_adhesion_active", "low_adhesion_zones_m",
            "low_adhesion_entry_speeds_kmh", "low_adhesion_exit_speeds_kmh",
            "low_adhesion_min_deceleration_mps2",
            "low_adhesion_affected_limit_points_m",
            "max_latest_brake_advance_m",
            "stop_distance_delta_m_vs_baseline",
            "stop_distance_delta_pct_vs_baseline", "termination_reason",
            "brake_groups", "last_to_full_group", "max_full_time_s",
            "equivalent_delay_grouped_s", "equivalent_delay_unified_s",
            "stop_distance_delta_m_vs_unified",
            "stop_distance_delta_pct_vs_unified",
            "stopping_margin_delta_m_vs_unified",
            "max_latest_brake_delta_m_vs_unified",
        ]
        if has_supervision:
            header += SUPERVISION_SUMMARY_COLUMNS
        writer.writerow(header)
        for sc in result["scenarios"]:
            segs = sc["parameters"].get("adhesion_segments")
            seg_desc = (";".join(f"[{s['start_m']},{s['end_m']}):{s['adhesion']}"
                                 for s in segs) if segs else "")
            prop = sc.get("brake_propagation")
            groups_desc = ""
            if prop is not None:
                groups_desc = ";".join(
                    f"{g['name']}:{g['delay_s']}s"
                    for g in sc["parameters"]["brake_groups"])
            for mode in MODES:
                m = sc["modes"][mode]
                d = sc["vs_baseline"][mode]
                la = m["low_adhesion"]
                zones = la["zones"]
                advances = [a["latest_brake_advance_m"]
                            for z in zones for a in z["affected_limit_points_m"]
                            if a["latest_brake_advance_m"] is not None]
                if prop is not None:
                    cmp_m = prop["vs_unified"][mode]
                    lp_deltas = [lp["latest_brake_delta_m"]
                                 for lp in cmp_m["limit_points"]
                                 if lp["latest_brake_delta_m"] is not None]
                    prop_cells: list = [
                        groups_desc, prop["last_to_full_group"],
                        prop["max_full_time_s"],
                        cmp_m["equivalent_delay_grouped_s"],
                        cmp_m["equivalent_delay_unified_s"],
                        cmp_m["stop_distance_delta_m"],
                        cmp_m["stop_distance_delta_pct"],
                        cmp_m["stopping_margin_delta_m"],
                        max(lp_deltas, key=abs) if lp_deltas else "",
                    ]
                else:
                    prop_cells = ["", "", "", "", "", "", "", "", ""]
                row = [
                    sc["rank"], sc["name"], sc["is_baseline"],
                    sc["parameters"]["adhesion"], seg_desc,
                    sc["parameters"]["brake_force_factor"],
                    sc["parameters"]["delay_s"], mode, m["status"],
                    m["stop_position_m"], m["stop_distance_m"],
                    sc["stopping_margin_m"], m["max_deceleration_mps2"],
                    la["active"],
                    ";".join(f"[{z['start_m']},{z['end_m']}]:{z['min_adhesion']}"
                             for z in zones),
                    ";".join("" if z["entry_speed_kmh"] is None
                             else str(z["entry_speed_kmh"]) for z in zones),
                    ";".join("" if z["exit_speed_kmh"] is None
                             else str(z["exit_speed_kmh"]) for z in zones),
                    ";".join("" if z["min_deceleration_mps2"] is None
                             else str(z["min_deceleration_mps2"]) for z in zones),
                    ";".join(str(a["at_m"]) for z in zones
                             for a in z["affected_limit_points_m"]),
                    max(advances) if advances else "",
                    d["stop_distance_delta_m"], d["stop_distance_delta_pct"],
                    (m["termination"] or {}).get("reason", ""),
                    *prop_cells,
                ]
                if has_supervision:
                    row += _supervision_summary_cells(sc)
                writer.writerow(row)
        filename = "braking_summary.csv"
    else:
        traj_header = [
            "scenario", "mode", "t_s", "s_m", "v_mps", "v_kmh",
            "a_mps2", "adhesion", "grade_permille", "limit_kmh",
        ]
        if has_supervision:
            traj_header += ["warning_threshold_kmh",
                            "service_threshold_kmh",
                            "emergency_threshold_kmh"]
        writer.writerow(traj_header)
        for sc in result["scenarios"]:
            envelope = sc.get("supervision")
            for mode in MODES:
                for p in sc["modes"][mode]["trajectory"]:
                    row = [
                        sc["name"], mode, p["t_s"], p["s_m"], p["v_mps"],
                        p["v_kmh"], p["a_mps2"], p["adhesion"],
                        p["grade_permille"],
                        p["limit_kmh"] if p["limit_kmh"] is not None else "",
                    ]
                    if has_supervision:
                        # 监督触发阈值按常用包络标注（三级曲线来源一致，
                        # 与制动模式无关）
                        thr = (threshold_speeds_kmh(envelope, p["s_m"])
                               if envelope is not None and mode == "service"
                               else None)
                        row += [thr["warning"] if thr else "",
                                thr["service"] if thr else "",
                                thr["emergency"] if thr else ""]
                    writer.writerow(row)
        filename = "braking_trajectories.csv"

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/example", summary="获取示例请求体", tags=["meta"])
def get_example() -> dict:
    return EXAMPLE_REQUEST


@app.get("/api/health", summary="健康检查", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "version": __version__}
