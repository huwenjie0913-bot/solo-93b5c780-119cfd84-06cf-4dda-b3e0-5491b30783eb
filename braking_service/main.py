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
    if kind == "summary":
        writer.writerow([
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
        ])
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
                writer.writerow([
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
                ])
        filename = "braking_summary.csv"
    else:
        writer.writerow([
            "scenario", "mode", "t_s", "s_m", "v_mps", "v_kmh",
            "a_mps2", "adhesion", "grade_permille", "limit_kmh",
        ])
        for sc in result["scenarios"]:
            for mode in MODES:
                for p in sc["modes"][mode]["trajectory"]:
                    writer.writerow([
                        sc["name"], mode, p["t_s"], p["s_m"], p["v_mps"],
                        p["v_kmh"], p["a_mps2"], p["adhesion"],
                        p["grade_permille"],
                        p["limit_kmh"] if p["limit_kmh"] is not None else "",
                    ])
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
