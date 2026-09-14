"""FastAPI 入口：POST /api/simulate（JSON）与 /api/simulate/csv（CSV 下载）。"""

from __future__ import annotations

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
    ],
    "target_stop_m": 1000,
    "max_trajectory_points": 300,
}

# ------------------------------------------------------------------- 应用 ---

app = FastAPI(
    title="列车制动仿真 API",
    version=__version__,
    description=(
        "接收按里程排列的坡度/限速区段、车辆与制动参数，"
        "用分段 RK4 积分生成常用/紧急制动的速度—里程轨迹；"
        "校验区段断裂、里程重叠、单位冲突与非物理参数；"
        "输出每个限速点的最晚制动位置、停车余量、最大减速度与超速区间；"
        "支持多工况对比（干轨/湿轨/部分失效），按停车余量排序。"
    ),
)

OPENAPI_EXAMPLES = {
    "three_scenarios": {
        "summary": "三工况（干轨/湿轨/部分制动失效）完整示例",
        "description": "5 km 线路，含下坡段与两级限速收紧；目标停车点 1000 m，"
                       "部分制动失效工况将错过停车目标。",
        "value": EXAMPLE_REQUEST,
    }
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
            "brake_force_factor", "delay_s", "mode", "status",
            "stop_position_m", "stop_distance_m", "stopping_margin_m",
            "max_deceleration_mps2", "stop_distance_delta_m_vs_baseline",
            "stop_distance_delta_pct_vs_baseline", "termination_reason",
        ])
        for sc in result["scenarios"]:
            for mode in MODES:
                m = sc["modes"][mode]
                d = sc["vs_baseline"][mode]
                writer.writerow([
                    sc["rank"], sc["name"], sc["is_baseline"],
                    sc["parameters"]["adhesion"],
                    sc["parameters"]["brake_force_factor"],
                    sc["parameters"]["delay_s"], mode, m["status"],
                    m["stop_position_m"], m["stop_distance_m"],
                    sc["stopping_margin_m"], m["max_deceleration_mps2"],
                    d["stop_distance_delta_m"], d["stop_distance_delta_pct"],
                    (m["termination"] or {}).get("reason", ""),
                ])
        filename = "braking_summary.csv"
    else:
        writer.writerow([
            "scenario", "mode", "t_s", "s_m", "v_mps", "v_kmh",
            "a_mps2", "grade_permille", "limit_kmh",
        ])
        for sc in result["scenarios"]:
            for mode in MODES:
                for p in sc["modes"][mode]["trajectory"]:
                    writer.writerow([
                        sc["name"], mode, p["t_s"], p["s_m"], p["v_mps"],
                        p["v_kmh"], p["a_mps2"], p["grade_permille"],
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
