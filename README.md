# 列车制动仿真服务

基于 FastAPI 的列车制动分析接口：输入按里程排列的坡度/限速区段、车辆与制动参数，
用分段 RK4 数值积分生成**常用制动**与**紧急制动**的速度—里程轨迹，自动完成：

- **输入校验**：区段断裂（gap）、里程重叠（overlap）、单位冲突（km/h 与 m/s、
  ‰ 与 % 混淆）、非物理参数（负质量、负制动力、超物理黏着/坡度等）；
- **局部低黏着**：每个工况可提交按里程连续的 `adhesion_segments`
  （起止里程 + 黏着系数），表达秋季落叶、隧道渗水等仅覆盖部分里程的低黏着区；
  未提交时沿用标量 `adhesion`，旧请求完全兼容。前向 RK4 与限速点反向积分均按
  当前位置选取黏着上限；汇总列车进出低黏着区的速度、区内最低减速度、受影响
  限速点，以及相对标量基线的最晚制动位置前移量；
- **限速点分析**：每个限速收紧点的最晚制动位置（含制动延迟与建立期折算距离）、
  接近速度、轨迹实际通过速度、可行性标记；
- **指标**：停车余量（相对目标停车点）、最大减速度及其位置；
- **违规区间**：轨迹中超速的连续里程区间，以及错过停车目标的区间；
- **多工况对比**：一次提交干轨/湿轨/部分制动力失效等多个工况，
  按停车余量升序返回（无法停车者排最前），并给出相对基准工况的制动距离变化；
- **失败保护**：数值积分终止（如制动力不足、超出计算域）时保留已算轨迹，
  并报告终止位置与原因；
- **双格式输出**：结构化 JSON + 可下载 CSV（轨迹明细 / 工况汇总）。

## 运行

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn braking_service.main:app --reload --port 8000
```

- Swagger UI（含可直接试用的示例请求）：http://localhost:8000/docs
- 示例请求体：`GET /api/example`

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/simulate` | 运行仿真，返回结构化 JSON |
| POST | `/api/simulate/csv?kind=trajectory` | 下载全部轨迹点 CSV |
| POST | `/api/simulate/csv?kind=summary` | 下载工况汇总 CSV |
| GET | `/api/example` | 返回示例请求体 |
| GET | `/api/health` | 健康检查 |

## 快速试用

```bash
curl -s localhost:8000/api/example > req.json
curl -s -X POST localhost:8000/api/simulate -H 'Content-Type: application/json' -d @req.json | python3 -m json.tool
curl -OJ -X POST "localhost:8000/api/simulate/csv?kind=summary" -H 'Content-Type: application/json' -d @req.json
```

## 物理模型

运动方程（时间域 RK4，步长 0.05 s）：

```
m_eff · dv/dt = -( F_brake(v,t) + F_rr(v) + F_grade(s) )
```

- `F_brake`：制动力曲线（速度→kN，分段线性）× 工况系数，受黏着上限 `μ(s)·m·g`
  约束；`μ(s)` 默认取标量 `adhesion`，提交 `adhesion_segments` 时按当前里程
  从分段表选取；`t < 延迟` 时为 0，建立期内线性爬坡；
- `F_rr = a + b·v + c·v²`（Davis 滚动阻力）；
- `F_grade = m·g·slope(s)`，坡度分段恒定，上坡为正；
- `m_eff = m · 回转质量系数`。

**最晚制动位置**：从限速点以全制动曲线反向积分至接近速度
（前区段限速与初速度的较小者），再减去 `v·(延迟 + 建立期/2)` 的走行距离。
反推同样按当前里程选取黏着上限；提交黏着分段时，每个限速点额外给出
标量基线模型的最晚制动位置 `latest_brake_m_scalar_baseline` 与前移量
`latest_brake_advance_m`（正值表示低黏着要求更早下闸）。

### 局部低黏着输入与输出

`scenarios[].adhesion_segments` 为按里程升序、首尾相接（不允许空缺/重叠）的
连续分段表，区段外沿用标量 `adhesion`；如需在两个低黏着点之间保留正常黏着，
用一段标量黏着显式补齐。分段超出线路覆盖时整段在外将被拒绝（422），
跨界部分给出警告。

- 每个轨迹点含 `adhesion` 字段（该点实际黏着系数）；
- 每个制动模式含 `low_adhesion` 汇总：`active`、`baseline_adhesion` 与 `zones`；
  低于标量基线的相邻分段合并为一个低黏着区，逐区给出 `entry_speed_kmh` /
  `exit_speed_kmh`（未走到为 null）、`min_deceleration_mps2` 及其里程、
  `stopped_inside`、`affected_limit_points_m`（含最晚制动前移量）。
- CSV 轨迹增加 `adhesion` 列；汇总 CSV 增加黏着分段描述、低黏着区、
  进出速度、区内最低减速度、受影响限速点与最大前移量等列。

## 项目结构

```
braking_service/
  models.py      # Pydantic 请求模型（单位、区段、曲线、工况）
  validation.py  # 断裂/重叠/单位冲突/非物理参数校验
  physics.py     # RK4 积分、反向制动曲线、超速扫描
  simulator.py   # 工况编排、限速点分析、余量排序、基准对比
  main.py        # FastAPI 入口（JSON / CSV / OpenAPI 示例）
tests/test_api.py
```

## 测试

```bash
.venv/bin/python -m pytest tests/ -q   # 26 个用例
```
