"""核心动力学：分段数值积分（RK4）、反向制动曲线、超速区间扫描。

运动方程（沿里程积分，时间域 RK4）：
    m_eff * dv/dt = -(F_brake(v,t) + F_rr(v) + F_grade(s))
    ds/dt = v
其中：
- F_brake 受黏着限制：F <= mu * m * g；
- F_rr 为 Davis 滚动阻力；
- F_grade = m * g * slope(s)，坡度分段恒定，上坡为正。
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable, Optional

GRAVITY = 9.81
V_STOP = 0.01          # 判定停车的速度阈值 (m/s)
DT = 0.05              # 积分步长 (s)
MAX_STEPS = 120_000    # 步数上限，防止失控循环
BACKWARD_DS = 0.25     # 反向积分步长 (m)
OVERSPEED_TOL_KMH = 0.5  # 超速判定容差 (km/h)


def interp(x: float, xs: list[float], ys: list[float]) -> float:
    """分段线性插值，界外取端点值。"""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect_right(xs, x) - 1
    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[i], ys[i + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass
class ForceModel:
    """单个工况 + 制动模式下的合力模型。"""

    mass_kg: float           # 静态质量
    eff_mass_kg: float       # 计入回转惯量的等效质量
    curve_speeds: list[float]  # m/s，单调递增
    curve_forces: list[float]  # N
    adhesion: float
    force_factor: float
    delay_s: float
    buildup_s: float
    rr_a: float
    rr_b: float
    rr_c: float
    grade_at: Callable[[float], float]  # s -> slope (ratio)

    def brake_force(self, v: float, t: float) -> float:
        if t < self.delay_s:
            return 0.0
        if self.buildup_s > 0:
            scale = min(1.0, (t - self.delay_s) / self.buildup_s)
        else:
            scale = 1.0
        f = interp(max(v, 0.0), self.curve_speeds, self.curve_forces)
        f *= self.force_factor * scale
        cap = self.adhesion * self.mass_kg * GRAVITY  # 黏着上限
        return min(f, cap)

    def accel(self, t: float, s: float, v: float) -> float:
        vv = max(v, 0.0)
        f_b = self.brake_force(vv, t)
        f_r = self.rr_a + self.rr_b * vv + self.rr_c * vv * vv
        f_g = self.mass_kg * GRAVITY * self.grade_at(s)
        return -(f_b + f_r + f_g) / self.eff_mass_kg


@dataclass
class Trajectory:
    """积分结果：等时间间隔采样点 + 终止信息。"""

    points: list[dict]          # {t_s, s_m, v_mps, a_mps2}
    stopped: bool
    status: str                 # "ok" | "terminated"
    reason: str                 # 终止原因（stopped 时为 "stopped"）

    @property
    def last(self) -> dict:
        return self.points[-1]


def integrate(model: ForceModel, s0: float, v0: float, s_max: float,
              t_max: float = 6000.0) -> Trajectory:
    """前向积分至停车或越界。失败时保留已算轨迹并说明终止位置与原因。"""
    points: list[dict] = []
    t, s, v = 0.0, s0, v0
    a0 = model.accel(t, s, v)
    points.append({"t_s": t, "s_m": s, "v_mps": v, "a_mps2": a0})

    def finish(stopped: bool, status: str, reason: str) -> Trajectory:
        return Trajectory(points=points, stopped=stopped, status=status, reason=reason)

    if v0 <= V_STOP:
        return finish(True, "ok", "stopped")

    try:
        for _ in range(MAX_STEPS):
            # RK4（力依赖 t/s/v）
            k1s = v
            k1v = model.accel(t, s, v)
            k2s = v + 0.5 * DT * k1v
            k2v = model.accel(t + DT / 2, s + 0.5 * DT * k1s, v + 0.5 * DT * k1v)
            k3s = v + 0.5 * DT * k2v
            k3v = model.accel(t + DT / 2, s + 0.5 * DT * k2s, v + 0.5 * DT * k2v)
            k4s = v + DT * k3v
            k4v = model.accel(t + DT, s + DT * k3s, v + DT * k3v)

            s += DT / 6 * (k1s + 2 * k2s + 2 * k3s + k4s)
            v += DT / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            t += DT

            if not (math.isfinite(s) and math.isfinite(v)):
                return finish(False, "terminated",
                              f"non_finite_state: 数值发散于 t={t:.2f}s")

            if v <= V_STOP:
                v = 0.0
                points.append({"t_s": t, "s_m": s, "v_mps": v,
                               "a_mps2": model.accel(t, s, 0.0)})
                return finish(True, "ok", "stopped")

            a = model.accel(t, s, v)
            points.append({"t_s": t, "s_m": s, "v_mps": v, "a_mps2": a})

            if s >= s_max:
                return finish(False, "terminated",
                              f"exceeded_max_distance: 超过最大计算里程 {s_max:.0f} m 仍未停车")
            if t >= t_max:
                return finish(False, "terminated",
                              f"exceeded_max_time: 超过 {t_max:.0f} s 仍未停车")
    except Exception as exc:  # 保留已算轨迹
        return finish(False, "terminated", f"integration_error: {exc!r}")

    return finish(False, "terminated", "max_steps_exceeded: 积分步数超限")


def backward_brake_position(model: ForceModel, s_from: float, v_from: float,
                            v_target: float, s_min: float
                            ) -> tuple[Optional[float], str]:
    """从限速点反向积分全制动曲线，求达到 v_target 的最晚全制动位置。

    返回 (位置或 None, 说明)。None 表示在可用里程内无法把速度降到目标，
    即制动力不足以抵消下坡等因素。
    """
    if v_from >= v_target:
        return s_from, "no_braking_required"
    s, v = s_from, v_from
    prev_s, prev_v = s, v
    steps = 0
    while s > s_min:
        decel = -model.accel(1e9, s, v)  # t 取大 => 延迟与建立期已过，全制动
        if decel <= 1e-6:
            return None, (
                f"brake_insufficient: 在 s={s:.1f} m 处制动力无法克服下坡与阻力，"
                "无法继续反向减速"
            )
        prev_s, prev_v = s, v
        v += decel / v * BACKWARD_DS
        s -= BACKWARD_DS
        steps += 1
        if v >= v_target:
            # 线性插值求穿越点
            frac = (v_target - prev_v) / (v - prev_v)
            return prev_s - frac * BACKWARD_DS, "ok"
        if steps > 2_000_000:
            return None, "backward_steps_exceeded"
    return None, f"track_start_reached: 反推至里程 {s_min:.0f} m 仍未达到接近速度"


def scan_violations(points: list[dict],
                    limit_at_kmh: Callable[[float], Optional[float]]
                    ) -> list[dict]:
    """扫描轨迹，标出超速的连续里程区间。"""
    intervals: list[dict] = []
    cur: Optional[dict] = None
    for p in points:
        lim = limit_at_kmh(p["s_m"])
        v_kmh = p["v_mps"] * 3.6
        over = lim is not None and v_kmh > lim + OVERSPEED_TOL_KMH
        if over:
            excess = v_kmh - lim  # type: ignore[operator]
            if cur is None:
                cur = {"start_m": p["s_m"], "end_m": p["s_m"],
                       "max_overspeed_kmh": excess, "limit_kmh": lim}
            else:
                cur["end_m"] = p["s_m"]
                if excess > cur["max_overspeed_kmh"]:
                    cur["max_overspeed_kmh"] = excess
                    cur["limit_kmh"] = lim
        else:
            if cur is not None:
                cur["kind"] = "speed_limit"
                intervals.append(cur)
                cur = None
    if cur is not None:
        cur["kind"] = "speed_limit"
        intervals.append(cur)
    for iv in intervals:
        iv["start_m"] = round(iv["start_m"], 1)
        iv["end_m"] = round(iv["end_m"], 1)
        iv["max_overspeed_kmh"] = round(iv["max_overspeed_kmh"], 2)
    return intervals


def downsample(points: list[dict], max_points: int) -> list[dict]:
    """等距抽稀轨迹，始终保留首尾点。"""
    n = len(points)
    if n <= max_points:
        return points
    stride = math.ceil(n / max_points)
    out = points[::stride]
    if out[-1] is not points[-1]:
        out.append(points[-1])
    return out
