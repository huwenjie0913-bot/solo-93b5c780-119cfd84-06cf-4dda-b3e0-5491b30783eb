"""请求/响应数据模型（Pydantic v2）。

约定：
- 里程单位统一为米（m），时间为秒（s）；
- 速度输入单位由 speed_unit 声明（"km/h" 或 "m/s"），内部一律换算为 m/s；
- 坡度输入单位由 grade_unit 声明（"permille" ‰ / "percent" % / "ratio"），
  正值表示上坡，负值表示下坡；
- 制动力曲线为 速度 -> 力（kN） 的分段线性插值点列。
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

SpeedUnit = Literal["km/h", "m/s"]
GradeUnit = Literal["permille", "percent", "ratio"]


class GradeSegment(BaseModel):
    """坡度区段：[start_m, end_m) 内坡度恒定，value 按 grade_unit 解释。"""

    start_m: float = Field(..., description="区段起点里程 (m)")
    end_m: float = Field(..., description="区段终点里程 (m)")
    value: float = Field(..., description="坡度值，正=上坡，负=下坡")

    @model_validator(mode="after")
    def _check_order(self) -> "GradeSegment":
        if self.end_m <= self.start_m:
            raise ValueError("end_m 必须大于 start_m")
        return self


class LimitSegment(BaseModel):
    """限速区段：[start_m, end_m) 内限速恒定，limit 按 speed_unit 解释。"""

    start_m: float
    end_m: float
    limit: float = Field(..., gt=0, description="限速值（必须为正）")

    @model_validator(mode="after")
    def _check_order(self) -> "LimitSegment":
        if self.end_m <= self.start_m:
            raise ValueError("end_m 必须大于 start_m")
        return self


class CurvePoint(BaseModel):
    """制动力曲线采样点：某速度下的可用制动力。"""

    speed: float = Field(..., ge=0, description="速度（按 speed_unit）")
    force_kn: float = Field(..., ge=0, description="制动力 (kN)，不得为负")


class BrakeCurve(BaseModel):
    """制动力曲线：速度单调递增的分段线性插值点列。"""

    points: list[CurvePoint] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check_monotonic(self) -> "BrakeCurve":
        speeds = [p.speed for p in self.points]
        if any(b <= a for a, b in zip(speeds, speeds[1:])):
            raise ValueError("制动力曲线的速度点必须严格递增")
        return self


class RollingResistance(BaseModel):
    """滚动阻力 Davis 模型：F = a + b*v + c*v^2 （v 单位 m/s，F 单位 N）。"""

    a_n: float = Field(0.0, ge=0, description="常数项 (N)")
    b_n_per_mps: float = Field(0.0, ge=0, description="一次项 (N/(m/s))")
    c_n_per_mps2: float = Field(0.0, ge=0, description="二次项 (N/(m/s)^2)")


class Vehicle(BaseModel):
    mass_t: float = Field(..., gt=0, le=30000, description="车辆质量 (t)")
    rotary_inertia_factor: float = Field(
        1.06, ge=1.0, le=1.3,
        description="回转质量系数（计入转动惯量的等效质量放大系数）",
    )


class AdhesionSegment(BaseModel):
    """黏着区段：[start_m, end_m) 内黏着系数恒定为 adhesion。

    用于表达落叶、隧道渗水等仅覆盖部分里程的低黏着区；区段外沿用标量
    adhesion（请求级或工况级覆盖值）。
    """

    start_m: float = Field(..., description="区段起点里程 (m)")
    end_m: float = Field(..., description="区段终点里程 (m)")
    adhesion: float = Field(
        ..., gt=0, le=1.0, description="该区段黏着系数（0, 1]，非物理取值将被拒绝")

    @model_validator(mode="after")
    def _check_order(self) -> "AdhesionSegment":
        if self.end_m <= self.start_m:
            raise ValueError("end_m 必须大于 start_m")
        return self


class Scenario(BaseModel):
    """工况：覆盖基准参数以模拟干轨/湿轨/部分制动失效等。"""

    name: str = Field(..., min_length=1, max_length=64)
    adhesion: Optional[float] = Field(
        None, gt=0, le=1.0, description="黏着系数覆盖值；缺省用请求级 adhesion"
    )
    adhesion_segments: Optional[list[AdhesionSegment]] = Field(
        None,
        description=(
            "按里程连续排列的黏着区段（落叶/隧道渗水等局部低黏着）；"
            "区段内以此处黏着系数为黏着上限，区段外用 adhesion 标量；"
            "缺省或为空表示全程使用标量 adhesion，与旧请求兼容"
        ),
    )
    brake_force_factor: float = Field(
        1.0, gt=0, le=1.0, description="制动力比例系数，1=全力，<1 表示部分失效"
    )
    delay_s: Optional[float] = Field(
        None, ge=0, le=60, description="制动延迟覆盖 (s)；缺省用请求级 brake_delay_s"
    )
    is_baseline: bool = Field(False, description="是否作为对比基准工况")


class SimulationRequest(BaseModel):
    speed_unit: SpeedUnit = Field("km/h", description="速度输入单位")
    grade_unit: GradeUnit = Field("permille", description="坡度输入单位")

    grades: list[GradeSegment] = Field(..., min_length=1, description="坡度区段（按里程排列）")
    limits: list[LimitSegment] = Field(..., min_length=1, description="限速区段（按里程排列）")

    vehicle: Vehicle
    initial_position_m: float = Field(0.0, description="初始里程 (m)")
    initial_speed: float = Field(..., ge=0, description="初速度（按 speed_unit）")

    service_brake: BrakeCurve = Field(..., description="常用制动力曲线")
    emergency_brake: BrakeCurve = Field(..., description="紧急制动力曲线")

    brake_delay_s: float = Field(..., ge=0, le=60, description="制动指令到开始建立的延迟 (s)")
    brake_buildup_s: float = Field(0.0, ge=0, le=60, description="制动力线性建立时间 (s)")

    adhesion: float = Field(..., gt=0, le=1.0, description="基准黏着系数")
    rolling_resistance: RollingResistance = Field(default_factory=RollingResistance)

    scenarios: list[Scenario] = Field(..., min_length=1, description="工况列表，一次提交多个")

    target_stop_m: Optional[float] = Field(
        None, description="目标停车里程 (m)；缺省取线路覆盖终点"
    )
    max_trajectory_points: int = Field(
        300, ge=10, le=2000, description="每条轨迹在 JSON 中返回的最大采样点数"
    )

    @model_validator(mode="after")
    def _check_scenarios(self) -> "SimulationRequest":
        names = [s.name for s in self.scenarios]
        if len(names) != len(set(names)):
            raise ValueError("工况名称必须唯一")
        n_base = sum(1 for s in self.scenarios if s.is_baseline)
        if n_base > 1:
            raise ValueError("最多只能有一个基准工况 (is_baseline=true)")
        return self
