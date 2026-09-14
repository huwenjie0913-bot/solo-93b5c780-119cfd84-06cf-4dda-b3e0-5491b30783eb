"""限速监督包络：三级触发曲线、误差不利方向、校核与双配置对比。"""

from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from braking_service.main import (EXAMPLE_REQUEST, EXAMPLE_SUPERVISION_REQUEST,
                                  app)
from braking_service.models import SimulationRequest
from braking_service.supervision import LEVELS, threshold_speeds_kmh
from braking_service.simulator import Track
from braking_service.validation import validate_request

client = TestClient(app)


def _req() -> dict:
    return copy.deepcopy(EXAMPLE_REQUEST)


def _with_supervision(extra: dict | None = None,
                      alt: dict | None = None,
                      scenarios: list | None = None) -> dict:
    req = _req()
    req["scenarios"] = scenarios or [
        {"name": "dry_rail", "is_baseline": True}]
    req["supervision"] = {
        "name": "nominal",
        "warning": {"speed_error": 2.0, "position_error_m": 5.0,
                    "trigger_delay_s": 4.0,
                    "min_takeover_distance_m": 10.0,
                    "min_takeover_time_s": 0.3},
        "service": {"speed_error": 2.0, "position_error_m": 5.0,
                    "trigger_delay_s": 1.2,
                    "min_takeover_distance_m": 5.0,
                    "min_takeover_time_s": 0.2},
        "emergency": {"speed_error": 2.0, "position_error_m": 5.0,
                      "trigger_delay_s": 0.6},
    }
    if extra:
        req["supervision"].update(extra)
    if alt:
        req["alternative_supervision"] = alt
    return req


# ------------------------------------------------------- 兼容与结构 ---

def test_no_supervision_keeps_response_unchanged():
    body = client.post("/api/simulate", json=_req()).json()
    for sc in body["scenarios"]:
        assert "supervision" not in sc
        assert "supervision_comparison" not in sc


def test_supervision_structure():
    body = client.post("/api/simulate", json=_with_supervision()).json()
    dry = next(s for s in body["scenarios"] if s["name"] == "dry_rail")
    sup = dry["supervision"]
    assert sup["config"]["name"] == "nominal"
    pts = {p["at_m"]: p for p in sup["limit_points"]}
    p1200 = pts[1200.0]
    assert p1200["skipped"] is False
    for lv in LEVELS:
        c = p1200["curves"][lv]
        assert c["feasible"] is True
        assert len(c["curve"]) <= 80
        # 采样点按里程升序
        ss = [q["s_m"] for q in c["curve"]]
        assert ss == sorted(ss)
    # 2600 m 为放宽点，跳过
    assert pts[2600.0]["skipped"] is True
    assert "no_braking_required" in pts[2600.0]["note"]
    # 三对相邻级别的校核都存在
    pairs = {c["pair"] for c in p1200["checks"] if "pair" in c}
    assert pairs == {"warning_to_service", "service_to_emergency",
                     "emergency_to_limit_point"}


def test_curve_order_warning_outermost():
    """次序：告警触发位置 < 常用 < 紧急 < 限速点。"""
    body = client.post("/api/simulate", json=_with_supervision()).json()
    lp = body["scenarios"][0]["supervision"]["limit_points"][0]
    pos = lp["trigger_positions_m"]
    assert pos["warning"] < pos["service"] < pos["emergency"] < 1200
    # 触发区间为正
    for v in lp["trigger_intervals_m"].values():
        assert v > 0


# ------------------------------------------------------- 误差不利方向 ---

def test_errors_postpone_trigger_toward_limit():
    """速度/里程误差按不利方向计入：误差越大触发位置越靠近限速点。

    两级配置的触发延迟保持相同（隔离误差影响；告警级额外延迟另测）。
    """
    with_errors = _with_supervision({
        "name": "with_errors",
        "warning": {"speed_error": 2.0, "position_error_m": 5.0,
                    "trigger_delay_s": 1.2},
        "service": {"speed_error": 2.0, "position_error_m": 5.0,
                    "trigger_delay_s": 1.2},
        "emergency": {"speed_error": 2.0, "position_error_m": 5.0,
                      "trigger_delay_s": 0.6},
    })
    zero_errors = _with_supervision({
        "name": "zero_errors",
        "warning": {"speed_error": 0, "position_error_m": 0,
                    "trigger_delay_s": 1.2},
        "service": {"speed_error": 0, "position_error_m": 0,
                    "trigger_delay_s": 1.2},
        "emergency": {"speed_error": 0, "position_error_m": 0,
                      "trigger_delay_s": 0.6},
    })
    we = client.post("/api/simulate",
                     json=with_errors).json()["scenarios"][0]
    ze = client.post("/api/simulate",
                     json=zero_errors).json()["scenarios"][0]
    for lv in LEVELS:
        pe = we["supervision"]["limit_points"][0][
            "trigger_positions_m"][lv]
        pz = ze["supervision"]["limit_points"][0][
            "trigger_positions_m"][lv]
        # 含误差 => 触发位置更靠近限速点（里程更大）
        assert pe > pz, (lv, pe, pz)


def test_service_nominal_matches_latest_brake_position():
    """零误差、零监督延迟时，常用名义触发位置即现有最晚制动位置。"""
    req = _with_supervision({
        "name": "zero",
        "warning": {"trigger_delay_s": 0},
        "service": {"trigger_delay_s": 0},
        "emergency": {"trigger_delay_s": 0},
    })
    body = client.post("/api/simulate", json=req).json()
    dry = body["scenarios"][0]
    svc_lp = {p["at_m"]: p for p in dry["modes"]["service"]["limit_points"]}
    sup_lp = dry["supervision"]["limit_points"][0]
    nominal = sup_lp["curves"]["service"]["nominal_trigger_at_approach_m"]
    assert nominal is not None
    assert abs(nominal - svc_lp[1200.0]["latest_brake_m"]) < 1.0


# ------------------------------------------------------- 校核问题 ---

def test_takeover_margin_insufficient_reported():
    req = _with_supervision({
        "name": "tight",
        "warning": {"trigger_delay_s": 1.3,
                    "min_takeover_distance_m": 200,
                    "min_takeover_time_s": 10},
        "service": {"trigger_delay_s": 1.2,
                    "min_takeover_distance_m": 50,
                    "min_takeover_time_s": 5},
        "emergency": {"trigger_delay_s": 1.1},
    })
    sup = client.post("/api/simulate", json=req).json()["scenarios"][0][
        "supervision"]
    codes = [p["code"] for p in sup["problems"]]
    assert "takeover_margin_insufficient" in codes
    p = next(p for p in sup["problems"]
             if p["code"] == "takeover_margin_insufficient")
    assert p["at_m"] == 1200.0
    assert p["problem_mileage_m"] < 1200
    assert p["min_distance_m"] is not None
    assert p["min_time_s"] is not None
    assert "接管裕量不足" in p["reason"]


def test_curve_crossing_reported():
    req = _with_supervision({
        "name": "cross",
        "warning": {"trigger_delay_s": 0.0},
        "service": {"trigger_delay_s": 3.0},
        "emergency": {"trigger_delay_s": 0.0},
    })
    sup = client.post("/api/simulate", json=req).json()["scenarios"][0][
        "supervision"]
    cross = [p for p in sup["problems"] if p["code"] == "curve_crossing"]
    assert cross
    p = cross[0]
    assert p["pair"] == "warning_to_service"
    assert p["problem_mileage_m"] < 1200
    assert p["related_speed_kmh"] > 0
    assert "次序" in p["reason"]


def test_curve_infeasible_on_steep_downgrade():
    req = _req()
    req["grades"] = [{"start_m": 0, "end_m": 5000, "value": -25}]
    req["scenarios"] = [{"name": "weak", "brake_force_factor": 0.2}]
    req["supervision"] = {
        "name": "x",
        "warning": {"trigger_delay_s": 30},
        "service": {"trigger_delay_s": 20},
        "emergency": {"trigger_delay_s": 10},
    }
    sup = client.post("/api/simulate", json=req).json()["scenarios"][0][
        "supervision"]
    infeasible = [p for p in sup["problems"] if p["code"] == "curve_infeasible"]
    assert infeasible
    assert {p["level"] for p in infeasible} <= set(LEVELS)
    assert all(p["at_m"] == 1200.0 for p in infeasible)


# ------------------------------------------------------- 双配置对比 ---

def test_comparison_lists_trigger_and_takeover_changes():
    body = client.post(
        "/api/simulate", json=EXAMPLE_SUPERVISION_REQUEST).json()
    dry = next(s for s in body["scenarios"] if s["name"] == "dry_rail")
    cmp_ = dry["supervision_comparison"]
    assert cmp_["reference_config"] == "nominal"
    assert cmp_["alternative_config"] == "aged_equipment"
    lp = cmp_["limit_points"][0]
    assert lp["at_m"] == 1200.0
    # 常用/紧急：更大误差 => 触发位置后移（delta_m > 0）
    assert lp["trigger_positions"]["service"]["delta_m"] > 0
    assert lp["trigger_positions"]["emergency"]["delta_m"] > 0
    # 触发区间变化已列出
    assert "warning_window_m" in lp["trigger_intervals"]
    # 接管空间变化
    assert "service_to_emergency" in lp["takeover"]
    s = cmp_["summary"]
    assert set(s) >= {"trigger_position_delta_m",
                      "min_takeover_distance_delta_m",
                      "min_takeover_time_delta_s",
                      "new_problem_count", "resolved_problem_count"}


def test_comparison_new_and_resolved_problems():
    """宽松配置有问题、严格配置无问题 => 反向提交时出现 resolved/new。"""
    good = {
        "name": "good",
        "warning": {"trigger_delay_s": 4.0,
                    "min_takeover_distance_m": 10, "min_takeover_time_s": 0.3},
        "service": {"trigger_delay_s": 1.2,
                    "min_takeover_distance_m": 5, "min_takeover_time_s": 0.2},
        "emergency": {"trigger_delay_s": 0.6},
    }
    bad = {
        "name": "bad",
        "warning": {"trigger_delay_s": 1.3,
                    "min_takeover_distance_m": 200, "min_takeover_time_s": 10},
        "service": {"trigger_delay_s": 1.2,
                    "min_takeover_distance_m": 50, "min_takeover_time_s": 5},
        "emergency": {"trigger_delay_s": 1.1},
    }
    body = client.post(
        "/api/simulate", json=_with_supervision(good, alt=bad)).json()
    cmp_ = body["scenarios"][0]["supervision_comparison"]
    assert cmp_["summary"]["new_problem_count"] > 0
    assert any(p["takeover"] or p["problems_new"]
               for p in cmp_["limit_points"])
    new_codes = [p["code"] for lp in cmp_["limit_points"]
                 for p in lp["problems_new"]]
    assert "takeover_margin_insufficient" in new_codes

    # 反向：主配置差、对比配置好 => resolved
    body2 = client.post(
        "/api/simulate", json=_with_supervision(bad, alt=good)).json()
    cmp2 = body2["scenarios"][0]["supervision_comparison"]
    assert cmp2["summary"]["resolved_problem_count"] > 0


# ------------------------------------------------------- 校验 ---

def test_alternative_without_supervision_rejected():
    req = _req()
    req["alternative_supervision"] = {"name": "alt"}
    r = client.post("/api/simulate", json=req)
    assert r.status_code == 422
    assert any(e["code"] == "supervision_config_missing"
               for e in r.json()["detail"]["errors"])


def test_duplicate_supervision_names_rejected():
    req = _with_supervision(alt={"name": "nominal"})
    r = client.post("/api/simulate", json=req)
    assert r.status_code == 422
    assert any(e["code"] == "duplicate_supervision_names"
               for e in r.json()["detail"]["errors"])


def test_delay_order_warning():
    req_obj = SimulationRequest.model_validate(_with_supervision())
    # 故意制造延迟倒置
    req_obj.supervision.service.trigger_delay_s = 5.0
    errors, warnings = validate_request(req_obj)
    assert any(w["code"] == "supervision_delay_order" for w in warnings)
    assert errors == []


# ------------------------------------------------------- 单元与 CSV ---

def test_backward_curve_matches_position_solver():
    """曲线求解器与位置求解器在接近速度处给出相同的最晚全制动位置。"""
    from braking_service.physics import (backward_brake_curve,
                                         backward_brake_position)
    req_obj = SimulationRequest.model_validate(_req())
    track = Track(req_obj)
    from braking_service.simulator import _make_force_model
    model = _make_force_model(
        req_obj, track, req_obj.scenarios[0], req_obj.service_brake)
    v_app = 120 / 3.6
    lim = 80 / 3.6
    s_pos, note, _ = backward_brake_position(
        model, 1200.0, lim, v_app, track.start_m - 500)
    pts, note2, _ = backward_brake_curve(
        model, 1200.0, lim, v_app, track.start_m - 500)
    assert note == note2 == "ok"
    assert abs(pts[-1][0] - s_pos) < 0.25  # 同一反向步进，末点一致
    assert pts[0] == (1200.0, lim)


def test_threshold_speeds_lookup():
    body = client.post("/api/simulate", json=_with_supervision()).json()
    env = body["scenarios"][0]["supervision"]
    lp = env["limit_points"][0]
    far = lp["curves"]["service"]["curve"][5]["s_m"]
    near = lp["curves"]["service"]["curve"][-1]["s_m"]
    th_far = threshold_speeds_kmh(env, far)
    th_near = threshold_speeds_kmh(env, near)
    assert th_far is not None
    # 越靠近限速点，触发速度越低
    assert th_far["service"] > th_near["service"]
    # 越过硬（收紧点之后）无曲线
    assert threshold_speeds_kmh(env, 1300.0) is None


def test_csv_contains_supervision_columns_only_when_configured():
    r = client.post("/api/simulate/csv?kind=summary",
                    json=_with_supervision())
    header = r.text.splitlines()[0]
    for col in ("supervision_config", "supervision_problem_count",
                "sup_min_takeover_warning_service_m",
                "sup_trigger_emergency_m", "sup_window_service_m"):
        assert col in header
    rows = [l for l in r.text.splitlines()[1:] if ",nominal," in l]
    assert rows

    # 轨迹 CSV：三级触发阈值列
    r2 = client.post("/api/simulate/csv", json=_with_supervision())
    h2 = r2.text.splitlines()[0]
    assert "warning_threshold_kmh" in h2
    assert "emergency_threshold_kmh" in h2

    # 无监督配置：CSV 表头维持原样
    r3 = client.post("/api/simulate/csv?kind=summary", json=_req())
    assert "supervision_config" not in r3.text.splitlines()[0]
    r4 = client.post("/api/simulate/csv", json=_req())
    assert r4.text.splitlines()[0] == (
        "scenario,mode,t_s,s_m,v_mps,v_kmh,a_mps2,adhesion,"
        "grade_permille,limit_kmh")


def test_openapi_has_supervision_example():
    spec = client.get("/openapi.json").json()
    examples = spec["paths"]["/api/simulate"]["post"]["requestBody"][
        "content"]["application/json"]["examples"]
    assert "speed_supervision" in examples


def test_supervision_works_with_brake_groups():
    """分组制动模型同样可生成监督包络。"""
    from braking_service.main import EXAMPLE_GROUPED_REQUEST
    req = copy.deepcopy(EXAMPLE_GROUPED_REQUEST)
    req["supervision"] = {
        "name": "g",
        "warning": {"trigger_delay_s": 4},
        "service": {"trigger_delay_s": 1.2},
        "emergency": {"trigger_delay_s": 0.6},
    }
    body = client.post("/api/simulate", json=req).json()
    dry = next(s for s in body["scenarios"] if s["name"] == "dry_rail")
    lp = dry["supervision"]["limit_points"][0]
    assert lp["curves"]["emergency"]["feasible"] is True
    # 分组等效延迟已计入名义触发位置（与分组最晚制动口径一致）
    assert lp["curves"]["service"]["nominal_trigger_at_approach_m"] is not None
