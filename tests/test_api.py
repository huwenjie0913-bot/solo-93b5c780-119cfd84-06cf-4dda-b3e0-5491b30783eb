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
    assert len(body["scenarios"]) == 4
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
    svc = next(s for s in body["scenarios"]
               if s["name"] == "dry_rail")["modes"]["service"]
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
    svc = next(s for s in body["scenarios"]
               if s["name"] == "dry_rail")["modes"]["service"]
    speed_viol = [v for v in svc["violations"] if v["kind"] == "speed_limit"]
    assert speed_viol, "应检出超速区间"
    assert speed_viol[0]["max_overspeed_kmh"] > 0


def test_stop_target_missed_flagged():
    req = _req()
    req["target_stop_m"] = 300  # 不可能达到的停车目标
    body = client.post("/api/simulate", json=req).json()
    svc = next(s for s in body["scenarios"]
               if s["name"] == "partial_brake_failure")["modes"]["service"]
    stop_viol = [v for v in svc["violations"] if v["kind"] == "stop_target"]
    assert stop_viol, "应标出停车目标不可达区间"
    weak = next(s for s in body["scenarios"]
                if s["name"] == "partial_brake_failure")
    assert weak["stopping_margin_m"] < 0


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
    assert len(lines) == 1 + 4 * 2  # 表头 + 4 工况 × 2 模式


# ----------------------------------------------------- 分段黏着（低黏着） ---

def _scenario(body: dict, name: str) -> dict:
    return next(s for s in body["scenarios"] if s["name"] == name)


def test_trajectory_points_carry_actual_adhesion():
    body = client.post("/api/simulate", json=_req()).json()
    # 无分段工况：每个轨迹点黏着即标量值
    dry = _scenario(body, "dry_rail")
    assert dry["parameters"]["adhesion"] == 0.15
    assert dry["parameters"]["adhesion_segments"] is None
    for mode in ("service", "emergency"):
        traj = dry["modes"][mode]["trajectory"]
        assert all(p["adhesion"] == 0.15 for p in traj)
        assert dry["modes"][mode]["low_adhesion"]["active"] is False
    # 分段工况：轨迹点出现 0.06 与恢复后的 0.15
    leaf = _scenario(body, "leaf_film_tunnel")
    assert leaf["parameters"]["adhesion_segments"] is not None
    traj = leaf["modes"]["service"]["trajectory"]
    assert any(p["adhesion"] == 0.06 for p in traj)
    assert any(p["adhesion"] == 0.15 and p["s_m"] > 450 for p in traj)


def test_low_adhesion_boundary_entry_while_braking_and_recovery():
    """制动中驶入湿滑区（有入口速度、减速度下降），驶出后黏着恢复（减速度回升）。"""
    body = client.post("/api/simulate", json=_req()).json()
    svc = _scenario(body, "leaf_film_tunnel")["modes"]["service"]
    la = svc["low_adhesion"]
    assert la["active"] is True
    assert la["baseline_adhesion"] == 0.15

    zone1 = next(z for z in la["zones"] if z["start_m"] == 250.0)
    assert zone1["traversed"] is True
    assert zone1["entry_speed_kmh"] is not None
    assert zone1["exit_speed_kmh"] is not None
    assert zone1["exit_speed_kmh"] < zone1["entry_speed_kmh"]
    assert zone1["stopped_inside"] is False
    # 区内最低减速度：0.06 黏着下约 0.58 m/s²，显著低于全程最大减速度
    assert zone1["min_deceleration_mps2"] < svc["max_deceleration_mps2"]
    assert 250.0 <= zone1["min_deceleration_position_m"] <= 450.0

    # 驶出后恢复：450 m 之后的采样点减速度应大于区内最低减速度
    after = [p for p in svc["trajectory"]
             if 450 < p["s_m"] <= 600 and p["v_mps"] > 0]
    assert after
    assert min(p["a_mps2"] for p in after) < 0  # 仍在制动
    assert max(-p["a_mps2"] for p in after) > zone1["min_deceleration_mps2"]


def test_second_zone_affects_limit_point_with_advance():
    """未走行到的渗水点位于 1200 m 限速点反推路径上：标记受影响并给前移量。"""
    body = client.post("/api/simulate", json=_req()).json()
    svc = _scenario(body, "leaf_film_tunnel")["modes"]["service"]

    pts = {p["at_m"]: p for p in svc["limit_points"]}
    p1200 = pts[1200]
    assert p1200["low_adhesion_on_braking_path"] is True
    assert p1200["latest_brake_m"] < p1200["latest_brake_m_scalar_baseline"]
    assert p1200["latest_brake_advance_m"] > 0
    assert any(z["start_m"] == 900.0 for z in p1200["low_adhesion_zones_m"])

    # 回填到汇总区：affected_limit_points_m 含 1200 m 限速点
    zone2 = next(z for z in svc["low_adhesion"]["zones"]
                 if z["start_m"] == 900.0)
    assert zone2["traversed"] is False
    assert zone2["entry_speed_kmh"] is None
    aff = zone2["affected_limit_points_m"]
    assert [a["at_m"] for a in aff] == [1200]
    assert aff[0]["latest_brake_advance_m"] > 0


def test_segments_inactive_when_no_below_baseline():
    """分段黏着全部不低于标量基线时，汇总标记为 inactive。"""
    req = _req()
    req["scenarios"] = [
        {"name": "dry_rail", "is_baseline": True},
        {"name": "uniform", "adhesion_segments": [
            {"start_m": 0, "end_m": 1500, "adhesion": 0.15},
            {"start_m": 1500, "end_m": 5000, "adhesion": 0.2},
        ]},
    ]
    body = client.post("/api/simulate", json=req).json()
    la = _scenario(body, "uniform")["modes"]["service"]["low_adhesion"]
    assert la["active"] is False
    assert la["zones"] == []


def test_adhesion_segment_gap_rejected():
    req = _req()
    req["scenarios"] = [
        {"name": "dry_rail", "is_baseline": True},
        {"name": "gapped", "adhesion_segments": [
            {"start_m": 200, "end_m": 400, "adhesion": 0.06},
            {"start_m": 500, "end_m": 700, "adhesion": 0.06},  # 400~500 空缺
        ]},
    ]
    _assert_422(req, "segment_gap")


def test_adhesion_segment_overlap_and_order_rejected():
    req = _req()
    req["scenarios"] = [
        {"name": "dry_rail", "is_baseline": True},
        {"name": "overlap", "adhesion_segments": [
            {"start_m": 400, "end_m": 700, "adhesion": 0.06},
            {"start_m": 500, "end_m": 800, "adhesion": 0.06},
        ]},
    ]
    _assert_422(req, "segment_overlap")

    req = _req()
    req["scenarios"] = [
        {"name": "dry_rail", "is_baseline": True},
        {"name": "unsorted", "adhesion_segments": [
            {"start_m": 500, "end_m": 700, "adhesion": 0.06},
            {"start_m": 200, "end_m": 400, "adhesion": 0.06},
        ]},
    ]
    _assert_422(req, "segments_not_sorted")


def test_adhesion_segment_out_of_coverage_rejected():
    req = _req()
    req["scenarios"] = [
        {"name": "dry_rail", "is_baseline": True},
        {"name": "outside", "adhesion_segments": [
            {"start_m": 6000, "end_m": 7000, "adhesion": 0.06},
        ]},
    ]
    _assert_422(req, "adhesion_segment_out_of_coverage")


def test_adhesion_segment_nonphysical_by_schema():
    req = _req()
    req["scenarios"] = [
        {"name": "dry_rail", "is_baseline": True},
        {"name": "bad", "adhesion_segments": [
            {"start_m": 200, "end_m": 400, "adhesion": 0},
        ]},
    ]
    assert client.post("/api/simulate", json=req).status_code == 422


def test_csv_trajectory_has_adhesion_column():
    r = client.post("/api/simulate/csv", json=_req())
    header = r.text.splitlines()[0].split(",")
    assert "adhesion" in header
    rows = [ln for ln in r.text.splitlines()
            if ln.startswith("leaf_film_tunnel,")]
    assert any(",0.06," in ln for ln in rows)


def test_csv_summary_has_low_adhesion_columns():
    r = client.post("/api/simulate/csv?kind=summary", json=_req())
    header = r.text.splitlines()[0]
    for col in ("adhesion_segments", "low_adhesion_active",
                "low_adhesion_zones_m", "low_adhesion_entry_speeds_kmh",
                "low_adhesion_exit_speeds_kmh",
                "low_adhesion_min_deceleration_mps2",
                "low_adhesion_affected_limit_points_m",
                "max_latest_brake_advance_m"):
        assert col in header, col
    leaf_rows = [ln for ln in r.text.splitlines()
                 if ",leaf_film_tunnel," in ln]
    assert len(leaf_rows) == 2  # service + emergency
    assert any("True" in ln and "1200" in ln for ln in leaf_rows)


# ------------------------------------------------------------------ 其他 ---

def test_health_and_example():
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/example").json()["scenarios"]


def test_openapi_has_example():
    spec = client.get("/openapi.json").json()
    post = spec["paths"]["/api/simulate"]["post"]
    examples = post["requestBody"]["content"]["application/json"]["examples"]
    assert "four_scenarios" in examples
    assert "grouped_propagation" in examples


# ------------------------------------------------------- 分组制动传播 ---

def _grouped_req() -> dict:
    """头/中/尾三组，延迟 0.5/1.5/2.5 s 递增；曲线按质量份额拆分。"""
    req = _req()
    req["scenarios"] = [
        {"name": "dry_rail", "is_baseline": True},
        {"name": "wet_rail", "adhesion": 0.08},
        {"name": "slow_command", "delay_s": 1.0},
    ]
    svc = [(0, 360), (50, 380), (80, 390), (120, 400), (160, 410)]
    emg = [(0, 500), (50, 520), (80, 535), (120, 550), (160, 560)]

    def curve(points, share):
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


def test_grouped_propagation_milestones():
    body = client.post("/api/simulate", json=_grouped_req()).json()
    sc = _scenario(body, "dry_rail")
    prop = sc["brake_propagation"]
    groups = prop["groups"]
    assert [g["name"] for g in groups] == ["head", "middle", "tail"]
    # 各组开始响应时刻 = 传播延迟；达到全力 = 延迟 + 建立时间
    assert [g["response_time_s"] for g in groups] == [0.5, 1.5, 2.5]
    assert [g["full_time_s"] for g in groups] == [1.5, 2.5, 3.5]
    # 响应/全力里程随组序递增（列车在前行中依次建立）
    assert [g["response_position_m"] for g in groups] == sorted(
        g["response_position_m"] for g in groups)
    assert groups[0]["response_position_m"] > 0
    # 最后建立的车辆组为尾组
    assert prop["last_to_full_group"] == "tail"
    assert prop["max_full_time_s"] == 3.5
    assert [g["is_last_to_full"] for g in groups] == [False, False, True]
    assert prop["unified_model"] == {"delay_s": 1.5, "buildup_s": 1.0}


def test_grouped_force_time_series():
    body = client.post("/api/simulate", json=_grouped_req()).json()
    svc = _scenario(body, "dry_rail")["modes"]["service"]
    series = svc["brake_force_series"]
    assert len(series) == len(svc["trajectory"])
    # 传播早期：头组已出力，尾组尚未响应
    early = next(p for p in series if 0.6 <= p["t_s"] < 2.5)
    assert early["group_forces_kn"]["head"] > 0
    assert early["group_forces_kn"]["tail"] == 0
    # 首组响应前全列无力
    assert all(p["total_force_kn"] == 0 for p in series if p["t_s"] < 0.5)
    # 各组之和等于全列合力；全力期合力接近曲线值（400 kN @120 km/h）
    for p in series:
        assert abs(sum(p["group_forces_kn"].values())
                   - p["total_force_kn"]) < 0.05
    assert 390 <= max(p["total_force_kn"] for p in series) <= 405


def test_grouped_vs_unified_comparison():
    body = client.post("/api/simulate", json=_grouped_req()).json()
    prop = _scenario(body, "dry_rail")["brake_propagation"]
    su = prop["vs_unified"]["service"]
    # 统一模型低估空走距离：分组停车距离更长、余量更小
    assert su["stop_distance_delta_m"] > 0
    assert su["stop_distance_delta_pct"] > 0
    assert su["stopping_margin_delta_m"] < 0
    assert su["stop_distance_grouped_m"] > su["stop_distance_unified_m"]
    # 分组等效延迟（力加权）大于统一延迟
    assert su["equivalent_delay_grouped_s"] > su["equivalent_delay_unified_s"]
    # 限速点最晚制动位置前移（分组值更小）
    lp = {p["at_m"]: p for p in su["limit_points"]}
    assert lp[1200]["latest_brake_delta_m"] < 0
    assert lp[1200]["latest_brake_grouped_m"] < lp[1200]["latest_brake_unified_m"]
    # 紧急制动模式同样有对比
    assert prop["vs_unified"]["emergency"]["stop_distance_delta_m"] > 0


def test_grouped_matches_unified_when_delays_equal():
    """各组延迟/建立相同且曲线按质量拆分时，分组模型退化为统一模型。"""
    req = _grouped_req()
    req["scenarios"] = [{"name": "dry_rail", "is_baseline": True}]
    for g in req["brake_groups"]:
        g["delay_s"] = 1.5
        g["buildup_s"] = 1.0
    body = client.post("/api/simulate", json=req).json()
    su = _scenario(body, "dry_rail")["brake_propagation"]["vs_unified"]["service"]
    assert abs(su["stop_distance_delta_m"]) < 0.5
    assert su["equivalent_delay_grouped_s"] == su["equivalent_delay_unified_s"]


def test_grouped_scenario_delay_offset():
    """工况级 delay_s 作为统一附加延迟叠加到各组传播延迟上。"""
    body = client.post("/api/simulate", json=_grouped_req()).json()
    sc = _scenario(body, "slow_command")
    assert sc["parameters"]["group_delay_offset_s"] == 1.0
    times = [g["response_time_s"] for g in sc["brake_propagation"]["groups"]]
    assert times == [1.5, 2.5, 3.5]


def test_no_groups_keeps_original_behavior():
    body = client.post("/api/simulate", json=_req()).json()
    for sc in body["scenarios"]:
        assert "brake_propagation" not in sc
        assert "brake_groups" not in sc["parameters"]
        for mode in ("service", "emergency"):
            assert "brake_force_series" not in sc["modes"][mode]


def test_group_mass_mismatch_rejected():
    req = _grouped_req()
    req["brake_groups"][2]["mass_t"] = 130  # 合计 430 ≠ 420
    r = client.post("/api/simulate", json=req)
    assert r.status_code == 422
    errs = r.json()["detail"]["errors"]
    assert any(e["code"] == "group_mass_mismatch"
               and e["location"] == "brake_groups" for e in errs)


def test_group_delay_reversed_rejected():
    req = _grouped_req()
    req["brake_groups"][2]["delay_s"] = 0.1  # 尾组先于前组响应：组序颠倒
    r = client.post("/api/simulate", json=req)
    assert r.status_code == 422
    errs = r.json()["detail"]["errors"]
    assert any(e["code"] == "group_delay_reversed"
               and e["location"] == "brake_groups[2].delay_s" for e in errs)


def test_duplicate_group_names_rejected():
    req = _grouped_req()
    req["brake_groups"][1]["name"] = "head"
    r = client.post("/api/simulate", json=req)
    assert r.status_code == 422
    errs = r.json()["detail"]["errors"]
    assert any(e["code"] == "duplicate_group_names"
               and e["location"] == "brake_groups[1].name" for e in errs)


def test_group_all_zero_curves_rejected():
    req = _grouped_req()
    for g in req["brake_groups"]:
        g["service_brake"] = {"points": [{"speed": 0, "force_kn": 0},
                                         {"speed": 100, "force_kn": 0}]}
    _assert_422(req, "non_physical")


def test_group_warnings():
    # 单组退化
    req = _grouped_req()
    req["brake_groups"] = [req["brake_groups"][0]]
    req["vehicle"]["mass_t"] = 60
    body = client.post("/api/simulate", json=req).json()
    codes = [w["code"] for w in body["validation"]["warnings"]]
    assert "single_brake_group" in codes
    # 后组先于前组达到全力（建立时间差异）
    req = _grouped_req()
    req["brake_groups"][0]["buildup_s"] = 5.0  # 头组 0.5+5.0=5.5 > 尾组 2.5+1.0
    body = client.post("/api/simulate", json=req).json()
    codes = [w["code"] for w in body["validation"]["warnings"]]
    assert "group_full_order_reversed" in codes


def test_grouped_csv_summary_has_propagation_columns():
    r = client.post("/api/simulate/csv?kind=summary", json=_grouped_req())
    assert r.status_code == 200
    header = r.text.splitlines()[0]
    for col in ("brake_groups", "last_to_full_group", "max_full_time_s",
                "equivalent_delay_grouped_s", "equivalent_delay_unified_s",
                "stop_distance_delta_m_vs_unified",
                "stop_distance_delta_pct_vs_unified",
                "stopping_margin_delta_m_vs_unified",
                "max_latest_brake_delta_m_vs_unified"):
        assert col in header, col
    rows = [ln for ln in r.text.splitlines() if ",dry_rail," in ln]
    assert len(rows) == 2
    assert all(",tail," in ln for ln in rows)
    # 无分组请求：传播列存在但为空
    r2 = client.post("/api/simulate/csv?kind=summary", json=_req())
    row = r2.text.splitlines()[1]
    assert row.endswith("," * 9)
