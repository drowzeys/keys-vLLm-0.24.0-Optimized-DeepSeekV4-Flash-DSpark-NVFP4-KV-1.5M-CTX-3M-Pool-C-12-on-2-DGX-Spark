# Keys — vLLM 0.24.0-Optimized DeepSeek-V4-Flash-DSpark · NVFP4-KV · 1.5M CTX · 3M Pool · C=12 · 2× DGX Spark

Frontier serving recipe for **DeepSeek-V4-Flash-DSpark** with **DSpark speculative decoding**
and an experimental **`nvfp4_ds_mla` 4-bit KV cache**, tuned for a **1.5M-token context window**
on **2× NVIDIA DGX Spark (GB10, sm_121a)** at tensor-parallel 2.

> **Headline result:** a **3.23M-token KV pool** at a **1.5M context window**, coherent
> needle retrieval validated to **543,994 tokens**, **C=12 concurrency at ~255 tok/s aggregate**,
> and **61–67% DSpark acceptance** — on two Spark boxes over RoCE/IB.

This repo publishes **our original artifacts** (launcher, ops tooling, benchmark harness +
full results, and the vLLM 0.24.0 sparse-MLA port fix). The upstream serving overlay is
**not** redistributed here — see [`docs/BUILD.md`](docs/BUILD.md) for how to build the image
from its upstream sources, with credit.

📦 **Model / recipe card on Hugging Face:**
[`drowzeys/DeepSeek-V4-Flash-DSpark-NVFP4-KV-1.5M-CTX-2xDGX-Spark`](https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-NVFP4-KV-1.5M-CTX-2xDGX-Spark)
— includes the launcher, benchmarks, and a two-command download-and-serve guide (pulls the
base model `deepseek-ai/DeepSeek-V4-Flash-DSpark`).

---

## Standing configuration

| Parameter | Value |
|---|---|
| Model | `DeepSeek-V4-Flash-DSpark` (deepseek-ai) |
| KV cache | `nvfp4_ds_mla` (4-bit MLA KV) |
| Context window | **1,500,000** (`max_model_len`) |
| KV pool | **3,231,736 tokens** (21.25 GiB @ util 0.85) |
| Max concurrency | **2.15× @1.5M** (~3.08× @1M) |
| `max_num_seqs` | **12** |
| `gpu_memory_utilization` | **0.85** |
| Speculative | DSpark, `MTP_NUM_TOKENS=5` |
| `VLLM_USE_B12X_WO_PROJECTION` | **1** |
| Topology | 2× DGX Spark GB10, TP=2, RoCE/IB |

### Why `WO_PROJECTION=1`

We benchmarked both settings head-to-head (see [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) §3).
`WO=0` is marginally faster single-stream; `WO=1` sustains **higher DSpark acceptance under
concurrency (67% vs 58%)** and a higher peak decode rate (343 vs 312 tok/s). **The standing
config chooses `WO=1`: at high concurrency we value sustained acceptance over raw single-stream speed.**

---

## Results at a glance

- **Single-stream:** ~51 tok/s, 61.6% acceptance.
- **Concurrency (this config):** C1 51 → C6 171 → **C12 255 tok/s** aggregate (seqs=12 saturation; do not exceed 12).
- **WO=0 vs WO=1 @ C16 (seqs=16):** both ~318 tok/s aggregate.
- **Context sweep:** decode **flat ~47–50 tok/s** from 1.7k→10.5k; only TTFT (prefill) grows.
- **Long-context needle (6k→512k):** **correct recall at every size through 543,994 tokens**, no garble.

Full tables: **[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)**.

---

## Model

This recipe serves the **stock** base model (unmodified — NVFP4 is a runtime KV-cache setting):

- **[`deepseek-ai/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)**

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-DSpark --local-dir ./DeepSeek-V4-Flash-DSpark
```

## Quickstart

```bash
# 0. Get the stock model (above), point the launcher's model path at it.

# 1. Build the serving image (see docs/BUILD.md for the upstream stage chain)
#    -> vllm-dspark-runtime:dspark-nvfp4-stage-c on both nodes

# 2. Edit ops/dspark-nvfp4-serve.sh for your fabric (MASTER, IF, HCA, GID) and model path.

# 3. ALWAYS clear + verify both nodes before (re)launch (see ops/gpu-clear.sh):
bash ops/gpu-clear.sh

# 4. Launch worker first, then head:
ssh <worker> '~/dspark-nvfp4-serve.sh 1'   # rank 1 (worker)
~/dspark-nvfp4-serve.sh 0                    # rank 0 (head)  ->  serves :8000

# Overridable profile knobs (defaults = the standing 1.5M config):
MAX_MODEL_LEN=1500000 MAX_NUM_SEQS=12 GPU_MEM_UTIL=0.85 ~/dspark-nvfp4-serve.sh 0
```

### Ops rule (baked in)
A dead head leaves the TP worker stuck in an NCCL retry loop **holding ~77 GB of GPU** —
relaunching on that stale state fails the 2-node NCCL handshake. **Always** `gpu-clear` and
**verify no compute-apps on both nodes** before relaunch, then bring up **worker → head**.

---

## Repository layout

```
ops/
  dspark-nvfp4-serve.sh     # the launcher (1.5M/seqs12/util0.85/WO1 + RoCE/IB networking)
  gpu-clear.sh              # force-clear + verify-free GPU on a node (run before every relaunch)
benchmarks/
  RESULTS.md                # ALL data tables + long-context sweep
  concurrency_sweep.sh      # C1..N aggregate/per-req throughput (uniform, ignore_eos)
  context_sweep.sh          # decode tok/s + TTFT vs context (code workload)
  longctx_needle_sweep.sh   # 6k..512k needle retrieval (coherence at scale)
vllm-0.24-port/
  flashinfer_sparse_mla.py  # our vLLM 0.24.0 compiled sparse-MLA drop-in (GB10 fixes)
  PORT_NOTES.md             # the 0.24.0 port + the two kernel-integration fixes
docs/
  BUILD.md                  # build the nvfp4 stage-c image from upstream (with credit)
```

## Credits & license

**Special thanks to tonyd2wild, MiaAI-Lab, Rafael Caricio, and the vLLM project** — see
**[`CREDITS.md`](CREDITS.md)**. Apache-2.0 (`LICENSE`). This work stands on:
- [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0)
- Rafael Caricio's DSpark vLLM integration (`rafaelcaricio/vllm`)
- aidendle94's GB10 `sparkrun-vllm-ds4` runtime base
- DSpark packaging by **MiaAI-Lab** and **tonyd2wild** (the `nvfp4_ds_mla` stage A/B/C recipe)
- `deepseek-ai/DeepSeek-V4-Flash-DSpark` (the model)

See [`NOTICE`](NOTICE). This repo distributes original launcher/benchmark/port artifacts and
build instructions — not the upstream overlay sources.

## Support / Donations

If this frontier work is useful to you, donations are appreciated and help fund more
open GB10 / DGX Spark serving research:

- **Solana:** `drkeys.sol`

Thank you 🙏

