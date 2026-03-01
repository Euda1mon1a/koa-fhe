# koa-fhe

<!-- mcp-name: io.github.euda1mon1a/koa-fhe -->

Confidential coprocessor client — compute on encrypted data via Koa's FHE service.

The server **never sees plaintext**. Your data is encrypted locally, evaluated under Fully Homomorphic Encryption, and decrypted locally. Every query costs $0.001–$0.10 USDC via [x402](https://www.x402.org/).

## Install

```bash
pip install koa-fhe            # core (compare, add, multiply, schedule analysis)
pip install koa-fhe[ml]        # + encrypted ML inference (XGBoost workload scoring)
```

## Quickstart

```python
import koa_fhe

client = koa_fhe.Client()  # default: localhost:3410

# Encrypted comparison — server never sees 85 or 80
result = client.compare(85, 80)
print(result.greater)  # True

# Encrypted addition
result = client.add(500, 300)
print(result.value)  # 800

# Encrypted multiplication
result = client.multiply(7, 13)
print(result.value)  # 91
```

## API Reference

### `Client(server, *, payment_proof)`

| Method | Returns | Latency | Price |
|--------|---------|---------|-------|
| `compare(a, b)` | `CompareResult(greater, stats)` | ~625ms | $0.002 |
| `add(a, b)` | `ArithmeticResult(value, stats)` | ~260ms | $0.001 |
| `multiply(a, b)` | `ArithmeticResult(value, stats)` | ~685ms | $0.002 |
| `measure_rigidity(old, new)` | `RigidityResult(hamming_distance, rigidity_score, severity, stats)` | ~3.5s | $0.05 |
| `analyze_schedule(old, new, *, constraints, alpha, beta)` | `ScheduleAnalysis(objective, ...)` | ~5s | $0.10 |
| `predict_workload(features)` | `WorkloadPrediction(predictions, stats)` | ~7s | $0.01 |
| `health()` | `ServiceHealth(status, circuits, ...)` | instant | free |
| `circuits()` | `dict` — circuit catalog with hashes, pricing | instant | free |

### Warm-up & Readiness

Key generation is the slowest step (~2-30s per circuit). Use `warm_up()` to pre-generate keys in background threads:

```python
client = koa_fhe.Client()
client.warm_up(["add", "threshold"])  # keygen in background

# ... do other work ...

print(client.ready)  # {"add": True, "threshold": True}
result = client.add(1, 2)  # instant — keys already ready
```

### Key Deduplication

Evaluation keys are content-addressed (SHA-256). If you create multiple `Client` instances with the same circuits, the SDK detects duplicate keys on the server via `HEAD /keys/{circuit}?hash=...` and reuses them — no redundant uploads.

### How it works

1. On first call, the SDK downloads circuit specs from the server
2. Keys are generated **locally** — the server never sees your secret key
3. Key hash is checked against server — if a matching key exists, it's reused
4. Otherwise, keys are uploaded (with integrity verification via `X-Content-Hash`)
5. Inputs are encrypted locally, sent as ciphertext
6. Server evaluates the FHE circuit on ciphertext
7. Encrypted result returns, SDK decrypts locally

Key setup is cached per circuit. Subsequent calls skip steps 1-4. Server-side key management enforces TTL (24h default) and per-circuit caps (10 keys max).

## LangChain Integration

```bash
pip install koa-fhe[langchain]
```

```python
from koa_fhe.langchain import koa_tools
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

agent = initialize_agent(
    tools=koa_tools("http://your-koa-server:3410"),
    llm=ChatOpenAI(),
    agent=AgentType.OPENAI_FUNCTIONS,
)

agent.run("Is 85 greater than 80? Use encrypted comparison.")
```

Tools: `koa_encrypted_compare`, `koa_encrypted_add`, `koa_encrypted_multiply`, `koa_encrypted_rigidity`, `koa_encrypted_schedule_analysis`, `koa_fhe_health`.

## Discovery

The FHE service exposes machine-readable metadata for agent-to-agent integration:

- **OpenAPI spec:** `GET /openapi.yaml`
- **Service manifest:** `GET /.well-known/x402` — x402 payment info, service catalog, ERC-8004 identity
- **Agent aliases:** `/v1/confidential/{predicate,aggregate,arithmetic,rigidity,analysis,predict}`

On-chain identity: [ERC-8004 Agent #21648](https://basescan.org/tx/0x3ecbd7e26723433cea3ed0b361becb0859d6117c89041e4eb33a9e6b4812e427) on Base mainnet.

## Dependencies

**Core** (`pip install koa-fhe`): `concrete-python` + `numpy`. Handles compare, add, multiply, rigidity, and schedule analysis.

**ML** (`pip install koa-fhe[ml]`): adds `concrete-ml` for encrypted XGBoost inference. This pulls in PyTorch, scikit-learn, etc. — only install if you need `predict_workload()`.

## License

MIT
