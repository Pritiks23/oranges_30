# ⚡ Inference Router

A unified control plane for AI inference that routes every request to the
**cluster that minimises effective cost** across 10 inference clusters
spanning AWS, GCP, Azure, CoreWeave, Lambda, and other providers:

```
effective_cost = compute_cost($) + λ × latency(s)
```

- **Cluster-based routing** — 10 pre-configured inference clusters (D–M) with different cost/latency trade-offs
- **Compute cost** — actual token cost based on cluster pricing (midpoint of observed range)
- **Latency penalty** — configurable weight λ ($/s) that converts response time into a dollar figure, letting you tune the speed/cost trade-off
- **Multi-cloud providers** — AWS (on-demand, spot), GCP (spot, TPU), Azure (reserved), CoreWeave, Lambda, Crusoe, and private infrastructure
- **Mock mode** — works instantly with zero credentials; every cluster returns a realistic simulated response with authentic cost + latency numbers
- **Live dashboard** — visual routing decision for every request, cost breakdown chart, request history, aggregate metrics

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Browser  ──  GET /  →  frontend/index.html                │
│           ──  POST /api/complete  →  router.py             │
│               ┌──────────────────────────────────────────┐ │
│               │  for each of 10 clusters (D–M):         │ │
│               │    effective_cost = compute_cost +      │ │
│               │                    λ × latency          │ │
│               │  select cluster with min(effective_cost) │ │
│               └──────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Cluster D (AWS H100)   Cluster E (AWS Spot A100)    │  │
│  │  Cluster F (GCP H100)   Cluster G (Azure A100)       │  │
│  │  Cluster H (CoreWeave)  Cluster I (Lambda)           │  │
│  │  Cluster J (Crusoe)     Cluster K (Private)          │  │
│  │  Cluster L (GCP TPU)    Cluster M (Multi-Region)     │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

```
inference-router/
├── app/
│   ├── main.py        FastAPI app + API routes + frontend serving
│   ├── router.py      Cluster-based cost-optimal routing engine
│   ├── schema.py      Pydantic request / response models
│   └── metrics.py     In-memory metrics store
├── adapters/
│   ├── base.py        Abstract adapter interface
│   ├── aws.py         AWS Bedrock  (mock + live)
│   ├── gcp.py         GCP Vertex AI (mock + live)
│   └── azure.py       Azure OpenAI  (mock + live)
├── config/
│   ├── config.py      Provider cost tables + routing parameters
│   └── clusters.py    10 inference clusters (D–M) with specs
├── frontend/
│   └── index.html     Single-page dashboard (no build step)
├── infra/
│   ├── Dockerfile
│   └── k8s.yaml
├── .env.example
└── requirements.txt
```

---

## Quick Start

### Option A — run locally (recommended for first try)

```bash
# 1. Clone & install
git clone <repo>
cd inference-router
pip install -r requirements.txt

# 2. Configure (mock mode is on by default — no credentials needed)
cp .env.example .env

# 3. Start
uvicorn app.main:app --reload --port 8000

# 4. Open
open http://localhost:8000
```

### Option B — Docker

```bash
# Build from repo root
docker build -f infra/Dockerfile -t inference-router .

# Run in mock mode
docker run -p 8000:8000 -e MOCK_MODE=true inference-router

open http://localhost:8000
```

### Option C — Kubernetes

```bash
# Fill in credentials in infra/k8s.yaml (Secret section)
kubectl apply -f infra/k8s.yaml
```

---

## Connecting Real Cloud Providers

Set `MOCK_MODE=false` in `.env`, then configure the provider(s) you want.
You can enable any combination; un-configured providers are skipped.

### AWS Bedrock

1. AWS Console → **Bedrock → Model Access** → request access for
   *Claude 3 Haiku* in `us-east-1` (approves in ~1 min)
2. **IAM → Users → Create user** → attach `AmazonBedrockFullAccess`
3. **Security credentials → Create access key** → copy both values
4. Add to `.env`:
   ```
   AWS_ACCESS_KEY_ID=AKIA…
   AWS_SECRET_ACCESS_KEY=…
   AWS_REGION=us-east-1
   ```

### GCP Vertex AI

1. GCP Console → **APIs & Services → Enable** → *Vertex AI API*
2. **IAM → Service Accounts → Create** → grant role **Vertex AI User**
3. **Keys → Add Key → JSON** → download the file
4. Add to `.env`:
   ```
   GCP_PROJECT_ID=my-project
   GCP_REGION=us-central1
   GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/key.json
   ```

### Azure OpenAI

1. Azure Portal → **Create resource → Azure OpenAI** (takes ~2 min)
2. Resource → **Keys and Endpoint** → copy KEY 1 + Endpoint URL
3. **Azure OpenAI Studio → Deployments → Create** → deploy `gpt-35-turbo`
4. Add to `.env`:
   ```
   AZURE_OPENAI_API_KEY=…
   AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
   AZURE_OPENAI_DEPLOYMENT=gpt-35-turbo
   ```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/complete` | Route prompt to best cluster, return completion + cost breakdown |
| `GET`  | `/api/clusters` | Cluster status (provider, GPU, cost range, latency) |
| `GET`  | `/api/metrics` | Aggregate stats across all requests |
| `GET`  | `/api/history?n=20` | Last N requests (newest first) |
| `GET`  | `/api/pricing` | Current token prices (legacy provider data) |

### POST /api/complete

```json
{
  "prompt": "Explain quantum entanglement in one paragraph.",
  "max_tokens": 256,
  "latency_weight": 0.001
}
```

Response includes `candidates[]` with the per-cluster cost breakdown and
`selected: true` on the winning cluster.

Example response:
```json
{
  "text": "Quantum entanglement is...",
  "cluster_id": "F",
  "cluster_provider": "GCP",
  "cluster_gpu": "H100",
  "model": "70B",
  "actual_latency_ms": 52,
  "input_tokens": 12,
  "output_tokens": 45,
  "compute_cost": 0.000057,
  "latency_cost": 0.000052,
  "effective_cost": 0.000109,
  "candidates": [
    {
      "cluster_id": "F",
      "cluster_provider": "GCP",
      "cluster_gpu": "H100",
      "model": "70B",
      "effective_cost": 0.000109,
      "selected": true
    },
    {
      "cluster_id": "I",
      "cluster_provider": "Lambda",
      "cluster_gpu": "H100",
      "model": "70B",
      "effective_cost": 0.000125,
      "selected": false
    }
  ]
}
```

### GET /api/clusters

Returns all 10 available clusters ranked by effective cost potential:

```json
[
  {
    "cluster_id": "F",
    "provider": "GCP",
    "gpu_type": "H100",
    "model_support": ["70B"],
    "is_available": true,
    "effective_cost_range": "$0.000035–$0.000075",
    "typical_latency_ms": 55
  },
  {
    "cluster_id": "L",
    "provider": "GCP",
    "gpu_type": "TPU v5",
    "model_support": ["70B"],
    "is_available": true,
    "effective_cost_range": "$0.000035–$0.000070",
    "typical_latency_ms": 62
  }
]
```

---

## Cluster Specifications

10 clusters span different cost/latency trade-offs:

| Cluster | Provider | GPU | Model | Cost/Token | Latency | Effective Cost | Notes |
|---------|----------|-----|-------|-----------|---------|---------------|-------|
| **D** | AWS | H100 | 70B | $0.000055–$0.000095 | 35–55ms | $0.000060–$0.000110 | On-demand, stable |
| **E** | AWS | A100 | 13B–70B | $0.000035–$0.000070 | 45–90ms | $0.000040–$0.000090 | Spot, unstable |
| **F** | GCP | H100 | 70B | $0.000030–$0.000065 | 40–70ms | **$0.000035–$0.000075** | **Best case spot** |
| **G** | Azure | A100 | 13B–70B | $0.000055–$0.000110 | 30–60ms | $0.000060–$0.000120 | Reserved, predictable |
| **H** | CoreWeave | H100 | 70B | $0.000040–$0.000075 | 30–50ms | $0.000045–$0.000080 | Excellent efficiency |
| **I** | Lambda | H100 | 13B–70B | $0.000050–$0.000085 | 25–45ms | $0.000055–$0.000095 | **Lowest latency** |
| **J** | Crusoe | H100 | 70B | $0.000045–$0.000080 | 35–60ms | $0.000050–$0.000090 | Balanced |
| **K** | Private | A100 | 13B–70B | $0.000080–$0.000160 | 50–120ms | $0.000090–$0.000180 | Highest cost/latency |
| **L** | GCP | TPU v5 | 70B | $0.000030–$0.000060 | 45–80ms | **$0.000035–$0.000070** | **Competitive cost** |
| **M** | AWS | H100 | 70B | $0.000060–$0.000110 | 55–120ms | $0.000070–$0.000130 | Multi-region failover |

---

## Tuning the Latency Weight

| λ ($/s) | Behaviour |
|---------|-----------|
| `0`     | Pure cost minimisation — pick the cheapest cluster regardless of speed |
| `0.001` | **Default** — 1 second of extra latency ≈ $0.001 penalty |
| `0.01`  | Strongly prefer fast clusters; latency matters as much as token cost |
| `0.1`   | Always pick the fastest cluster |
