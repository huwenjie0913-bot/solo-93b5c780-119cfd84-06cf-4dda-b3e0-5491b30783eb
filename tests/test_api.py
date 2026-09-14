"""API 端到端测试：正常仿真、校验拒绝、CSV 下载、积分终止。"""

from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from braking_service.main import EXAMPLE_REQUEST, app

client = TestClient(app)


def _req() -> dict:
    return copy.deepcopy(EXAMPLE_REQUEST)


# ------------------------------------------------------------------ 正常 ---

def test_simulate_ok_structure():
    r = client.post("/api/simulate", json=_req())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["baseline_scenario"] == "dry_rail"
    assert len(body["scenarios"]) == 3
    for sc in body["scenarios"]:
        assert set(sc["modes"]) == {"service", "emergency"}
        svc = sc["modes"]["service"]
        assert svc["stopped"] is True
        assert svc["stop_distance_m"] > 0
        assert svc["max_deceleration_mps2"] > 0
        assert svc["trajectory"], "轨迹不能为空"
        assert svc["limit_points"], "应输出限速点分析"
        # 紧急制动距离应不大于常用制动
        assert (sc["modes"]["emergency"]["stop_distance_m"]
                <= svc["stop_distance_m"] + 1.0)


def test_scenarios_sorted_by_margin_and_baseline_delta():
    body = client.post("/api/simulate", json=_req()).json()
    margins = [sc["stopping_margin_m"] for sc in body["scenarios"]]
    assert margins == sorted(margins)
    # 部分制动失效余量应最小（排第一）
    assert body["scenarios"][0]["name"] == "partial_brake_failure"
    # 基准工况自身 delta 为 0
    base = next(sc for sc in body["scenarios"] if sc["is_baseline"])
    assert base["vs_baseline"]["service"]["stop_distance_delta_m"] == 0
    # 失效工况制动距离应变长
    weak = next(sc for sc in body["scenarios"]
                if sc["name"] == "partial_brake_failure")
    assert weak["vs_baseline"]["service"]["stop_distance_delta_m"] > 0
    assert weak["vs_baseline"]["service"]["stop_distance_delta_pct"] > 0


def test_limit_points_latest_brake():
    body = client.post("/api/simulate", json=_req()).json()
    svc = body["scenarios"][0]["modes"]["service"]
    pts = {p["at_m"]: p for p in svc["limit_points"]}
    assert 1200 in pts and 2600 in pts
    p1200 = pts[1200]
    # 限速 80 < 接近速度 120，需要制动，最晚制动点应在限速点之前
    assert p1200["latest_brake_m"] is not None
    assert p1200["latest_brake_m"] < 1200
    assert p1200["feasible"] is True
    # 限速 100 > 接近速度 80，无需制动
    assert pts[2600]["note"] == "no_braking_required"


def test_overspeed_violation_detected():
    req = _req()
    req["initial_speed"] = 160  # 超过初始区段限速 120
    body = client.post("/api/simulate", json=req).json()
    svc = body["scenarios"][0]["modes"]["service"]
    speed_viol = [v for v in svc["violations"] if v["kind"] == "speed_limit"]
    assert speed_viol, "应检出超速区间"
    assert speed_viol[0]["max_overspeed_kmh"] > 0


def test_stop_target_missed_flagged():
    req = _req()
    req["target_stop_m"] = 300  # 不可能达到的停车目标
    body = client.post("/api/simulate", json=req).json()
    svc = body["scenarios"][0]["modes"]["service"]
    stop_viol = [v for v in svc["violations"] if v["kind"] == "stop_target"]
    assert stop_viol, "应标出停车目标不可达区间"
    assert body["scenarios"][0]["stopping_margin_m"] < 0


def test_terminated_when_cannot_stop():
    req = _req()
    req["scenarios"] = [{"name": "tiny_brakes", "brake_force_factor": 0.02}]
    req["grades"] = [{"start_m": 0, "end_m": 5000, "value": -30}]  # 大下坡
    body = client.post("/api/simulate", json=req).json()
    svc = body["scenarios"][0]["modes"]["service"]
    assert svc["status"] == "terminated"
    assert svc["termination"]["reason"].startswith("exceeded_max_distance")
    assert svc["termination"]["position_m"] > 0
    assert svc["trajectory"], "终止时仍应保留已算轨迹"
    assert body["scenarios"][0]["stopping_margin_m"] is None


# ------------------------------------------------------------------ 校验 ---

def _assert_422(req: dict, code: str):
    r = client.post("/api/simulate", json=req)
    assert r.status_code == 422, r.text
    codes = [e["code"] for e in r.json()["detail"]["errors"]]
    assert code in codes, f"期望错误码 {code}，实际 {codes}"


def test_segment_gap_rejected():
    req = _req()
    req["grades"][1]["start_m"] = 1600  # 1500~1600 断裂
    _assert_422(req, "segment_gap")


def test_segment_overlap_rejected():
    req = _req()
    req["limits"][1]["start_m"] = 1100  # 与上一段 [0,1200) 重叠
    _assert_422(req, "segment_overlap")


def test_unit_conflict_rejected():
    req = _req()
    req["speed_unit"] = "m/s"  # 但限速仍按 km/h 数值填写
    _assert_422(req, "unit_conflict")


def test_non_physical_grade_rejected():
    req = _req()
    req["grades"][1]["value"] = -400  # -400‰ 非物理
    _assert_422(req, "non_physical")


def test_non_physical_params_rejected_by_schema():
    req = _req()
    req["vehicle"]["mass_t"] = -5
    assert client.post("/api/simulate", json=req).status_code == 422
    req = _req()
    req["scenarios"] = [{"name": "x", "brake_force_factor": 1.5}]
    assert client.post("/api/simulate", json=req).status_code == 422
    req = _req()
    req["brake_delay_s"] = -1
    assert client.post("/api/simulate", json=req).status_code == 422


def test_duplicate_scenario_names_rejected():
    req = _req()
    req["scenarios"].append({"name": "dry_rail"})
    assert client.post("/api/simulate", json=req).status_code == 422


# ------------------------------------------------------------------- CSV ---

def test_csv_trajectory_download():
    r = client.post("/api/simulate/csv", json=_req())
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert "attachment" in r.headers["content-disposition"]
    lines = r.text.strip().splitlines()
    assert lines[0].startswith("scenario,mode,t_s,s_m")
    assert len(lines) > 100
    assert any(line.startswith("wet_rail,emergency,") for line in lines[1:])


def test_csv_summary_download():
    r = client.post("/api/simulate/csv?kind=summary", json=_req())
    assert r.status_code == 200
    lines = r.text.strip().splitlines()
    assert lines[0].startswith("rank,scenario")
    assert len(lines) == 1 + 3 * 2  # 表头 + 3 工况 × 2 模式


# ------------------------------------------------------------------ 其他 ---

def test_health_and_example():
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/example").json()["scenarios"]


def test_openapi_has_example():
    spec = client.get("/openapi.json").json()
    post = spec["paths"]["/api/simulate"]["post"]
    examples = post["requestBody"]["content"]["application/json"]["examples"]
    assert "three_scenarios" in examples
