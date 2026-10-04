# Concatemer Decode Service

从滚环扩增（RCA）产生的串联条码读段中，在插入、缺失和替换噪声下恢复共同切点。

联合选择：参考序列的一个**循环移位**、恰好 `copies` 个覆盖整条读段的**连续非空分段**，
以及每段的一次**全局比对**（Needleman–Wunsch，M/D/I 单碱基代价均为 1），按

1. 总编辑数（`total_edits`）
2. 单段最大编辑数（`max_segment_edits`）

字典序最小化；最优解释按 `(shift, boundaries, CIGAR)` 稳定排序
（`partial` 模式下还纳入两个端部参考范围）。

- 唯一最优 → `unique`：唯一移位、各段边界、编辑数、CIGAR 及可回放对齐双行串。
- 多个最优 → `ambiguous`：返回稳定排序后的前两份见证及 `more_witnesses`。
- 无可行解释 → HTTP 422 `constraint_failed`，区分
  `segment_length`（结构性长度不可能）与 `per_segment_edit_budget`（预算超限），
  后者给出无预算下最近分段及每个超限分段的位置、所需编辑数与超出量。

## 采集窗口截断：`terminal_mode=partial`

质控截取滚环扩增读段时，窗口常从一个重复单元中部开始、在另一个单元中部结束。
请求可携带可选字段 `"terminal_mode": "partial"`（省略时行为与旧版完全一致，
即 `full`）：

- `copies` 仍表示**连续观测片段数**（partial 下允许 2–8）；
- 首段必须对齐到同一循环移位参考的某个**非空真后缀**（范围 `[L-a, L)`，
  `1 ≤ a ≤ L-1`）；
- 末段必须对齐到其某个**非空真前缀**（范围 `[0, b)`，`1 ≤ b ≤ L-1`）；
- 中间各段仍对完整移位参考做全局比对。

服务联合选择**同一个循环移位**、两个端部参考范围、读段边界与各段比对。
成功响应的每份见证给出 `terminal_ranges: {first, last}`（均为相对该移位参考
的半开区间）和每段可回放的 CIGAR；同分解释继续返回稳定排序后的前两份见证。

截断在数学上是可识别但不一定唯一的：干净数据上切点可沿被覆盖的单元滑动，
因此多帧等优时返回 `ambiguous`；要区分**采集截断**与**噪声超限**：

- 端部无法形成合法前后缀（例如总长度放不下 `copies` 个非空片段）→
  `constraint.name = "terminal_prefix_suffix"`，`nearest` 为空；
- 合法前后缀存在但某段编辑数超出预算 → `per_segment_edit_budget`，
  `nearest.violating_segments` 给出超限片段（含 `role: first/last/middle`、
  `ref_range`、所需编辑数与超出量）。

## 目录

```
app/solver.py     核心算法：带限 NW + 全最优 CIGAR 枚举、前后缀分段 DP、见证重建、失败诊断
                  （含 partial 端部真后缀/真前缀的联合移位-范围-边界选择）
app/main.py       FastAPI 服务：POST /api/concatemers/decode 与 /health
tests/            62 个测试，含 full/partial 两种模式与穷举参考实现的一致性校验
scripts/verify.py 一次性校验：等健康 → pytest → 构建自检 → 旧模式回归 + 端部插缺替解码冒烟
                  + partial 定位失败（预算 / 前后缀结构）
Dockerfile        python:3.11-slim，内置容器健康检查
docker-compose.yml 可配置宿主机端口；verify 一次性服务（依赖 api 健康后启动）
```

## 用 Docker Compose 启动

```bash
# 默认宿主机端口 8000
docker compose up -d --build

# 自定义宿主机端口
HOST_PORT=9090 docker compose up -d --build
```

API 自带 `/health` 健康检查；compose 也配置了 healthcheck。

## 一次性校验（退出码报告结果）

```bash
docker compose run --build --rm verify
echo $?      # 0 通过，非 0 失败
```

`verify` 服务通过 `depends_on: condition: service_healthy` 等待 API 健康，然后依次：

1. 轮询 `/health`；
2. 在容器内执行全部代码测试（pytest）；
3. 构建自检（依赖版本、应用可导入、路由数）；
4. 对**运行中的 API** 发起解码冒烟，覆盖同一请求中的替换、缺失与插入，
   并额外核对 unique / ambiguous / infeasible 三种响应；
5. 对 `terminal_mode=partial` 发起实时冒烟：干净真后缀/真前缀分帧、
   端部插/缺/替解码、可定位的预算超限与前后缀结构失败，并回归校验
   省略字段时仍为 `full` 行为。

## 请求示例

```bash
curl -s -X POST http://localhost:8000/api/concatemers/decode \
  -H 'Content-Type: application/json' \
 -d '{
       "reference": "ACGTACGATC",
       "read": "ATGTACGATCACGTACGATACGTACGATCA",
       "copies": 3,
       "max_edits": 1
     }'
```

上面的读段三段分别含 1 个替换、1 个缺失、1 个插入，响应（节选）：

```json
{
  "status": "unique",
  "objective": {"total_edits": 3, "max_segment_edits": 1},
  "witness": {
    "shift": 0,
    "boundaries": [[0, 10], [10, 19], [19, 30]],
    "segments": [
      {"start": 0, "end": 10, "reference": "ACGTACGATC", "read": "ATGTACGATC",
       "edits": 1, "cigar": "10M",
       "aligned_reference": "ACGTACGATC", "marker": " ^        ", "aligned_read": "ATGTACGATC"},
      {"start": 10, "end": 19, "reference": "ACGTACGATC", "read": "ACGTACGAT",
       "edits": 1, "cigar": "9M1D",
       "aligned_reference": "ACGTACGATC", "marker": "         ^", "aligned_read": "ACGTACGAT-"},
      {"start": 19, "end": 30, "reference": "ACGTACGATC", "read": "ACGTACGATCA",
       "edits": 1, "cigar": "10M1I",
       "aligned_reference": "ACGTACGATC-", "marker": "          ^", "aligned_read": "ACGTACGATCA"}
    ]
  }
}
```

歧义（如同聚物导致移位不可区分）返回：

```json
{
  "status": "ambiguous",
  "objective": {"total_edits": 0, "max_segment_edits": 0},
  "witnesses": [ {…第一份…}, {…第二份…} ],
  "more_witnesses": true
}
```

### 截断读段（partial）请求示例

采集窗口在单元中部起、止。单碱基的端部片段使切点在干净数据上也唯一可判：

```bash
curl -s -X POST http://localhost:8000/api/concatemers/decode \
  -H 'Content-Type: application/json' \
 -d '{
       "reference": "ACGTACGATCGTACGATCAT",
       "read": "TACGTACGATCGTACGATCATACGTACGATCGTACGATCATA",
       "copies": 4,
       "max_edits": 0,
       "terminal_mode": "partial"
     }'
```

成功响应给出共同移位下的两个**端部范围**与可回放 CIGAR：

```json
{
  "status": "unique",
  "objective": {"total_edits": 0, "max_segment_edits": 0},
  "witness": {
    "shift": 0,
    "boundaries": [[0, 1], [1, 21], [21, 41], [41, 42]],
    "terminal_ranges": {"first": [19, 20], "last": [0, 1]},
    "segments": [
      {"start": 0, "end": 1, "reference": "T", "read": "T",
       "ref_range": [19, 20], "role": "first", "edits": 0, "cigar": "1M"},
      {"start": 1, "end": 21, "reference": "ACGTACGATCGTACGATCAT",
       "read": "ACGTACGATCGTACGATCAT", "ref_range": [0, 20],
       "edits": 0, "cigar": "20M"},
      {"start": 21, "end": 41, "reference": "ACGTACGATCGTACGATCAT",
       "read": "ACGTACGATCGTACGATCAT", "ref_range": [0, 20],
       "edits": 0, "cigar": "20M"},
      {"start": 41, "end": 42, "reference": "A", "read": "A",
       "ref_range": [0, 1], "role": "last", "edits": 0, "cigar": "1M"}
    ]
  }
}
```

端部片段更长时，切点可沿被覆盖的单元滑动，这类等优解释以 `ambiguous` 返回
稳定排序的前两份见证（排序键含边界、端部范围与 CIGAR）。

失败时（422）质控人员可区分截断与噪声。预算超限（下例 39 nt 的读段迫使
某个端部片段覆盖完整 20 碱基，超出零预算）：

```json
{
  "status": "infeasible",
  "error": "constraint_failed",
  "terminal_mode": "partial",
  "constraint": {
    "name": "per_segment_edit_budget",
    "feasible_read_length": [2, 38]
  },
  "nearest": {
    "shift": 0,
    "terminal_ranges": {"first": [1, 20], "last": [0, 19]},
    "violating_segments": [
      {"segment_index": 0, "role": "first", "ref_range": [1, 20],
       "required_edits": 1, "budget": 0, "over_by": 1, "cigar": "1I19M"}
    ]
  }
}
```

端部连合法前后缀都无法形成（长度放不下 `copies` 个非空片段）时：

```json
{
  "status": "infeasible",
  "error": "constraint_failed",
  "terminal_mode": "partial",
  "constraint": {"name": "terminal_prefix_suffix"},
  "nearest": null
}
```

## 本地开发（无 Docker）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 算法说明

- **全局比对**：参考长 ≤20、单段预算 ≤3，使用带宽为 `max_edits` 的带限
  Needleman–Wunsch；随后回溯枚举该距离下的**全部**最优 CIGAR（M/D/I），
  按“操作数优先、再按操作与计数”的确定性顺序排列。结果经 `lru_cache` 复用。
- **分段**：对每个循环移位做前缀 DP `(段数, 读段偏移) → (总编辑, 单段最大)`，
  并维护镜像后缀表；只沿“前缀 + 后缀 == 全局最优”的边做 DFS 重建，
  按段长升序、CIGAR 已排序的顺序产出，天然得到稳定排序的见证。
- **失败诊断**：主流程在预算内无解时，对每个移位计算参考到任意相关子串的
  编辑距离矩阵，再做一次**无单段预算**的分段 DP，找到全局最近分段，
  报告每个超限分段；若连无预算下都无法覆盖（结构性长度约束），直接定位为
  `segment_length`。
- **partial 端部处理**：端部段在枚举可行片段时**联合枚举**“观测长度 ×
  真后缀/真前缀跨度（1..L-1）”，复用同一套前缀/后缀 DP 与见证 DFS；
  稳定排序键逐段为 `(边界终点, 端部参考范围, CIGAR)`，与枚举顺序一致，
  因此每移位截取的前两份见证即全局稳定前两份。无预算诊断时端部同样允许
  选取任意非空真后缀/真前缀（对每个起点/每个跨度各做一次完整 NW），
  从而把 `terminal_prefix_suffix`（前后缀根本无法成帧）与
  `per_segment_edit_budget`（成帧合法但噪声超预算）分开报告。
