# Concatemer Decode Service

从滚环扩增（RCA）产生的串联条码读段中，在插入、缺失和替换噪声下恢复共同切点。

联合选择：参考序列的一个**循环移位**、恰好 `copies` 个覆盖整条读段的**连续非空分段**，
以及每段的一次**全局比对**（Needleman–Wunsch，M/D/I 单碱基代价均为 1），按

1. 总编辑数（`total_edits`）
2. 单段最大编辑数（`max_segment_edits`）

字典序最小化；最优解释按 `(shift, boundaries, CIGAR)` 稳定排序。

- 唯一最优 → `unique`：唯一移位、各段边界、编辑数、CIGAR 及可回放对齐双行串。
- 多个最优 → `ambiguous`：返回稳定排序后的前两份见证及 `more_witnesses`。
- 无可行解释 → HTTP 422 `constraint_failed`，区分
  `segment_length`（结构性长度不可能）与 `per_segment_edit_budget`（预算超限），
  后者给出无预算下最近分段及每个超限分段的位置、所需编辑数与超出量。

## 目录

```
app/solver.py     核心算法：带限 NW + 全最优 CIGAR 枚举、前后缀分段 DP、见证重建、失败诊断
app/main.py       FastAPI 服务：POST /api/concatemers/decode 与 /health
tests/            35 个测试，含与穷举参考实现的一致性校验
scripts/verify.py 一次性校验：等健康 → pytest → 构建自检 → 含插/缺/替的实时解码冒烟
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
   并额外核对 unique / ambiguous / infeasible 三种响应。

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
