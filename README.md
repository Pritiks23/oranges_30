# ⚡ Inference Router

A unified control plane for AI inference that routes every request to the
provider that **minimises effective cost**:

```
effective_cost = compute_cost($) + λ × latency(s)
```

- **Compute cost** — actual token cost at the provider's on-demand rate  
- **Latency penalty** — configurable weight λ ($/s) that converts response
  time into a dollar figure, letting you tune the speed/cost trade-off
- **Three providers out-of-the-box** — AWS Bedrock, GCP Vertex AI, Azure OpenAI
- **Mock mode** — works instantly with zero cloud credentials; every provider
  returns a realistic simulated response with authentic cost + latency numbers
- **Live dashboard** — visual routing decision for every request, cost
  breakdown chart, request history, aggregate metrics

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Browser  ──  GET /  →  frontend/index.html          │
│           ──  POST /api/complete  →  router.py       │
│               ┌──────────────────────────────────┐   │
│               │  for each provider:              │   │
│               │    effective_cost = compute +    │   │
│               │                    λ × latency   │   │
│               │  select min(effective_cost)       │   │
│               └──────────────────────────────────┘   │
│               ┌─────────┬──────────┬──────────────┐  │
│               │ AWS     │ GCP      │ Azure        │  │
│               │ Bedrock │ Vertex AI│ OpenAI       │  │
│               └─────────┴──────────┴──────────────┘  │
└─────────────────────────────────────────────────────┘
```

```
inference-router/
├── app/
│   ├── main.py        FastAPI app + API routes + frontend serving
│   ├── router.py      Cost-optimal routing engine
│   ├── schema.py      Pydantic request / response models
│   └── metrics.py     In-memory metrics store
├── adapters/
│   ├── base.py        Abstract adapter interface
│   ├── aws.py         AWS Bedrock  (mock + live)
│   ├── gcp.py         GCP Vertex AI (mock + live)
│   └── azure.py       Azure OpenAI  (mock + live)
├── config/
│   └── config.py      Provider cost tables + routing parameters
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
| `POST` | `/api/complete` | Route prompt, return completion + full cost breakdown |
| `GET`  | `/api/providers` | Provider status (configured / mock, latency) |
| `GET`  | `/api/metrics` | Aggregate stats across all requests |
| `GET`  | `/api/history?n=20` | Last N requests (newest first) |

### POST /api/complete

```json
{
  "prompt": "Explain quantum entanglement in one paragraph.",
  "max_tokens": 256,
  "latency_weight": 0.001
}
```

Response includes `candidates[]` with the per-provider cost breakdown and
`selected: true` on the winning provider.

---

## Tuning the Latency Weight

| λ ($/s) | Behaviour |
|---------|-----------|
| `0`     | Pure cost minimisation — pick the cheapest tokens regardless of speed |
| `0.001` | **Default** — 1 second of extra latency ≈ $0.001 penalty |
| `0.01`  | Strongly prefer fast providers; latency matters as much as token cost |
| `0.1`   | Always pick the fastest provider |
