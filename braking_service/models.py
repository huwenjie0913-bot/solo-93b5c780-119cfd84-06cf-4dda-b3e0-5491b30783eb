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


class BrakeGroup(BaseModel):
    """车辆组：长编组列车按前后顺序分组描述制动指令传播与建立。

    制动指令沿列车由前向后传播，因此各组 delay_s 应单调不减；
    各组质量合计必须等于 vehicle.mass_t。每组独立提交制动力曲线，
    数值积分按各组实际生效时刻汇总全列制动力，各组分别受
    mu(s) * m_group * g 的黏着上限约束。
    """

    name: str = Field(..., min_length=1, max_length=64,
                      description="组名（在输出中唯一标识该组）")
    mass_t: float = Field(..., gt=0, le=30000, description="该组质量 (t)")
    delay_s: float = Field(
        ..., ge=0, le=60,
        description="制动指令传播至该组的延迟 (s)；沿列车由前向后应单调不减")
    buildup_s: float = Field(
        0.0, ge=0, le=60, description="该组制动力线性建立时间 (s)")
    service_brake: BrakeCurve = Field(..., description="该组常用制动力曲线")
    emergency_brake: BrakeCurve = Field(..., description="该组紧急制动力曲线")


class SupervisionThreshold(BaseModel):
    """单级监督阈值的误差与接管裕量配置。

    三级阈值分别为：告警（warning）、常用制动介入（service）、
    紧急制动介入（emergency）。每级可独立配置：
    - speed_error：速度测量误差（按 speed_unit，按不利方向取测速偏低）；
    - position_error_m：里程测量误差 (m)，按不利方向取显示里程偏小
      （列车实际位置比显示更靠近限速点）；
    - trigger_delay_s：触发延迟 (s)（系统反应/指令建立附加延迟）；
    - min_takeover_distance_m / min_takeover_time_s：与更内一级曲线之间
      的最小接管裕量（距离 m / 时间 s）；紧急级的裕量指紧急介入曲线到
      限速点之间的最小余量。
    """

    speed_error: float = Field(
        0.0, ge=0, le=50, description="速度测量误差（按 speed_unit，按测速偏低计入）")
    position_error_m: float = Field(
        0.0, ge=0, le=500, description="里程测量误差 (m)，按定位偏大计入")
    trigger_delay_s: float = Field(
        0.0, ge=0, le=60, description="触发延迟 (s)，额外空走时间按接近速度折算距离")
    min_takeover_distance_m: float = Field(
        0.0, ge=0, le=5000, description="与更内一级曲线的最小接管距离裕量 (m)")
    min_takeover_time_s: float = Field(
        0.0, ge=0, le=300, description="与更内一级曲线的最小接管时间裕量 (s)")


class SupervisionConfig(BaseModel):
    """限速监督包络配置：三级阈值 + 名称（用于双配置对比）。

    告警曲线在常用全制动反推曲线基础上额外前置 warning.trigger_delay_s；
    常用/紧急曲线分别以常用/紧急全制动反推曲线为基准，各自叠加本级
    触发延迟。误差一律按不利方向计入：测速偏低使有效触发速度抬高，
    里程显示偏小与触发延迟使触发位置向限速点方向后移。
    """

    name: str = Field(
        "default", min_length=1, max_length=64,
        description="配置名称（对比两套配置时据此标识）")
    warning: SupervisionThreshold = Field(
        default_factory=SupervisionThreshold,
        description="告警阈值（驾驶员接管提示）")
    service: SupervisionThreshold = Field(
        default_factory=SupervisionThreshold,
        description="常用制动介入阈值")
    emergency: SupervisionThreshold = Field(
        default_factory=SupervisionThreshold,
        description="紧急制动介入阈值")
    max_curve_points: int = Field(
        80, ge=10, le=500,
        description="每条触发曲线在 JSON 中返回的最大采样点数")


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
        None, ge=0, le=60,
        description=(
            "制动延迟覆盖 (s)；缺省用请求级 brake_delay_s。"
            "提交 brake_groups 时，该值作为附加的统一指令延迟叠加到各组传播延迟上"
        ),
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

    brake_groups: Optional[list[BrakeGroup]] = Field(
        None, min_length=1,
        description=(
            "车辆组（按列车前后顺序）：分组制动传播模型。提交后数值积分按各组"
            "实际生效时刻汇总制动力，并以现有统一延迟模型为基线输出对比；"
            "缺省或为空时保持统一延迟模型行为，与旧请求兼容"
        ),
    )

    adhesion: float = Field(..., gt=0, le=1.0, description="基准黏着系数")
    rolling_resistance: RollingResistance = Field(default_factory=RollingResistance)

    scenarios: list[Scenario] = Field(..., min_length=1, description="工况列表，一次提交多个")

    target_stop_m: Optional[float] = Field(
        None, description="目标停车里程 (m)；缺省取线路覆盖终点"
    )
    max_trajectory_points: int = Field(
        300, ge=10, le=2000, description="每条轨迹在 JSON 中返回的最大采样点数"
    )

    supervision: Optional[SupervisionConfig] = Field(
        None,
        description=(
            "限速监督包络配置：分别配置告警/常用/紧急三级阈值的速度测量误差、"
            "里程误差、触发延迟与最小接管裕量；沿每个限速收紧点反推生成三条"
            "速度—里程触发曲线，按不利方向计入误差并校核曲线次序、交叉与"
            "接管裕量。缺省（null）时不输出监督结果，与旧请求完全兼容"
        ),
    )
    alternative_supervision: Optional[SupervisionConfig] = Field(
        None,
        description=(
            "第二套监督阈值配置，用于与 supervision 对比触发区间与接管空间的"
            "变化；仅在同时提交 supervision 时生效，且两者 name 不得相同"
        ),
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
