# Benchmark Results — DeepSeek-V4-Flash-DSpark NVFP4-KV on 2× DGX Spark (GB10, TP=2)

All numbers measured on **2× NVIDIA DGX Spark (GB10, sm_121a), TP=2** over RoCE/IB
(head `.4` + worker `.3`), one replica. Model `DeepSeek-V4-Flash-DSpark`, DSpark
speculative decoding (`MTP_NUM_TOKENS=5`), `kv_cache_dtype=nvfp4_ds_mla`,
deterministic sampling (`temperature=0`, `top_p=1.0`, `thinking=false`).

Serving stack: `vllm-dspark-runtime:dspark-nvfp4-stage-c` (aidendle94 GB10 base +
rafaelcaricio DSpark overlay + tonyd2wild NVFP4 stage A/B/C). See `docs/BUILD.md`.

---

## 1. Boot / KV pool (memory profiles)

| Profile | GPU util | max_num_seqs | Available KV | **KV pool (tokens)** | Max concurrency |
|---|---|---|---|---|---|
| 1M         | 0.80 | 6  | 6.26 GiB  | ~0.78M | ❌ fails (1M needs 7.3 GiB) |
| 1M         | 0.86 | 1  | 28.94 GiB | 3.19M  | 3.04× @1M |
| 1M         | 0.86 | 16 | 23.99 GiB | 3.13M  | 2.98× @1M |
| **1.5M** (standing) | **0.85** | **12** | **21.25 GiB** | **3,231,736** | **2.15× @1.5M** (~3.08× @1M) |

**1.5M context window** (`max_model_len=1500000`) backed by a **3,231,736-token KV pool**.
The model's native trained window is `max_position_embeddings=1048576`; the 1M–1.5M range is
RoPE-extended (enabled via `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`). Coherence validated by needle
retrieval to **512k** (§6); treat 1M–1.5M as the extended-context range (standard RoPE-extension
caveat applies past the native 1M). The 3.23M-token pool gives ~2.15× concurrency at full 1.5M.

---

## 2. Single-stream (warm, uniform 256 tok, `ignore_eos`)

| Config | Single-stream tok/s | Acceptance | Mean accept length |
|---|---|---|---|
| WO=0, seqs≤16 | 50–56 (peak 56.5) | 61–66% | ~4.1 |
| WO=1, seqs≤16 | 50–52 | 61–67% | ~4.1 |

`max_num_seqs` does **not** affect single-stream (1 == 6 == 16 all ~50–56 tok/s).
NVFP4-1M single-stream ceiling ≈ 55–57 tok/s. (fp8/262k reaches ~64 — a context-vs-speed tradeoff.)

---

## 3. Concurrency sweep — WO=0 vs WO=1 (seqs=16, uniform 256 tok)

| Concurrency | WO=0 agg tok/s | WO=0 /req | WO=1 agg tok/s | WO=1 /req |
|---|---|---|---|---|
| 1  | 56.5  | 56.5 | 51.0  | 51.0 |
| 2  | 83.0  | 41.5 | 76.1  | 38.0 |
| 4  | 118.1 | 29.5 | 121.5 | 30.4 |
| 6  | 167.9 | 28.0 | 161.2 | 26.9 |
| 8  | 200.4 | 25.0 | 186   | 23.3 |
| 12 | 236.3 | 19.7 | 261.2 | 21.8 |
| 16 | 318.3 | 19.9 | 318.8 | 19.9 |
| **peak engine decode** | 312 | | **343** | |
| **acceptance @ load**  | 58% | | **67%** | |

- **WO=0** wins single-user / low concurrency (≤8): faster single-stream, better per-req latency.
- **WO=1** wins high concurrency (12+): higher acceptance under load (67% vs 58%) and higher peak decode (343 vs 312).
- Both converge to **~318 tok/s @ C16** (matches tonyd2wild's reported 315).

**Design decision:** the standing config uses **WO_PROJECTION=1** — at high concurrency we value
**sustained acceptance (67%) over raw single-stream speed**.

---

## 4. Standing config — 1.5M / util 0.85 / seqs 12 / WO=1 (uniform 256 tok)

| Concurrency | Aggregate tok/s | Per-req tok/s |
|---|---|---|
| 1  | 51.4  | 51.4 |
| 2  | 84.2  | 42.1 |
| 4  | 119.6 | 29.9 |
| 6  | 170.9 | 28.5 |
| 8  | 209.4 | 26.2 |
| **12** | **255.4** | 21.3 |
| 16 (4 queue) | 209.8 ↓ | 13.1 ↓ |

- Single-stream 51.4 tok/s, **acceptance 61.6%**, peak engine decode 272 tok/s.
- **Peak aggregate at C12 = 255 tok/s** (the seqs=12 saturation point).
- Beyond 12, requests queue → throughput **regresses** (C16 drops to 210, per-req to 13). **Do not exceed C=12.**

---

## 5. Context-length sweep (single-stream, code workload)

Decode throughput vs context size. **Decode stays flat; only TTFT (prefill) grows.**

| Context (tok) | TTFT (prefill) | Decode tok/s | Acceptance |
|---|---|---|---|
| 1,755  | 1.14s | 47.5 | 51% |
| 3,518  | 2.04s | 46.3 | 51% |
| 7,003  | 4.12s | 47.5 | 53% |
| 10,529 | 6.97s | 50.3 | 54% |

Sparse-MLA keeps per-token decode cost ~constant regardless of context length —
the key property that makes 1M serving viable.

---

## 6. Long-context needle sweep (6k → 512k, single-stream)

A fact ("vault access code ZEBRA-4271-OMEGA") is placed at the **start** of a large
context; the model must recall it after N filler tokens. Tests coherence + attention at scale.

| Context (tok) | TTFT (prefill) | Decode tok/s | Needle recall |
|---|---|---|---|
| 6,369   | 3.8s   | 54.4  | ✅ |
| 16,994  | 5.9s   | 59.2  | ✅ |
| 33,994  | 9.8s   | 80.3  | ✅ |
| 67,994  | 19.8s  | 47.1  | ✅ |
| 135,994 | 41.0s  | 63.0  | ✅ |
| 271,994 | 97.6s  | 86.7  | ✅ |
| **543,994** | **263.7s** | **122.0** | ✅ |

**Headline:** coherent, correct needle retrieval at **every** size through **543,994 tokens** —
no garble, no drift, no attention collapse. Decode does **not** degrade with context
(high values at large context = highly predictable retrieval answer → near-100% DSpark accept).
TTFT scales ~linearly with context (extrapolated ~1M prefill ≈ 9 min; decode stays flat once generating).

---

## Reproduce

```bash
# concurrency sweep (server on :8000)
benchmarks/concurrency_sweep.sh <tag> 1 2 4 6 8 12 16
# context sweep (code workload)
benchmarks/context_sweep.sh
# long-context needle sweep
benchmarks/longctx_needle_sweep.sh 6000 16000 32000 64000 128000 256000 512000
```
