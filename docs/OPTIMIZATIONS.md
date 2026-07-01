# Optimizations & Engineering Notes

How this build reaches a **1.5M context window**, a **3.23M-token KV pool**, **>60% DSpark
acceptance sustained under concurrency**, and **C=12 at ~255 tok/s** on two DGX Spark (GB10) boxes.

Full numbers: [`../benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).
Attribution for every upstream/transplant component: [`../CREDITS.md`](../CREDITS.md).

---

## 1. KV token-pool optimization (1.5M context)

The enabler is the **`nvfp4_ds_mla` 4-bit MLA KV cache** — it packs far more tokens per GiB
than fp8, which is what makes a multi-million-token pool (and 1M+ context) fit on 2× GB10.

Measured per-token KV footprint ≈ **7.7 KB/token**. The pool then scales almost entirely with
`gpu_memory_utilization`; `max_num_seqs` barely moves it (DSpark's per-slot ring is small):

| Profile | util | max_num_seqs | Available KV | KV pool (tokens) | Result |
|---|---|---|---|---|---|
| 1M   | 0.80 | 6  | 6.26 GiB  | ~0.78M | ❌ fails (1M needs 7.3 GiB) |
| 1M   | 0.86 | 16 | 23.99 GiB | 3.13M  | 2.98× @1M |
| 1M   | 0.86 | 1  | 28.94 GiB | 3.19M  | 3.04× @1M |
| **1.5M** | **0.85** | **12** | **21.25 GiB** | **3,231,736** | **2.15× @1.5M** |

**Tuning takeaways**
- util 0.80 is **too low** — KV starves and the engine can't even guarantee one 1M request. util
  **0.85–0.86** is the sweet spot on GB10's 128 GB unified memory (leaves headroom, avoids OOM).
- The 1.5M setting is **KV-reservation headroom** (`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`). Native trained
  context is 1,048,576; 1M–1.5M is RoPE-extended. **Coherence validated by needle retrieval to 512k.**
- At 1.5M the pool gives **~2× full-length concurrency**; `max_num_seqs=12` caps concurrent sequences
  so 12 requests share ~3.23M tokens (avg ≤ ~269k each).

---

## 2. Custom sm_121a (GB10) patches

GB10 is compute-capability **sm_121a** (`TORCH_CUDA_ARCH_LIST=12.1a`), which stock kernels don't
target. The serving path needs GB10-specific work at three layers:

**a) NVFP4 KV dtype plumbing (stages A/B/C).** Registers `nvfp4_ds_mla` as a KV dtype, maps it to
`KVQuantMode.NVFP4`, and — for DeepSeek-V4 — settles on the **padded 584-byte** page envelope
(stage-C) that boots stably at 1M+ (the true-layout 416-byte kernel fails past ~411 real tokens).
*Recipe by tonyd2wild; see [`BUILD.md`](BUILD.md).*

**b) Compiled sparse-MLA on sm_121a — vLLM 0.24.0 port (original work here).**
Stock vLLM 0.24.0's sparse-MLA verify on GB10 falls back to slow Triton (~375 ms/verify).
[`../vllm-0.24-port/flashinfer_sparse_mla.py`](../vllm-0.24-port/flashinfer_sparse_mla.py) wraps
aidendle94's compiled `sparse_mla_sm120_decode_dsv4` kernel and fixes two integration bugs that
only surface on GB10 at real batch sizes:
  1. **Dual-cache split-K scratch** — the kernel tiles *both* the SWA and global-topk index sets,
     so `mid_out`/`mid_lse` splits must be `cdiv(prim,64) + cdiv(extra,64)`, not `max()`.
  2. **`out_lse` contiguity** — build `BatchSparseMLAPagedAttentionWrapper` with
     `max_num_heads = num_heads` **exact** (not padded to 128), else the internal `out_lse` slice
     is non-contiguous and the kernel's contiguity check fails past the old batch cap.
  → verify ~375 ms → compiled; single-stream 3.4 → ~27 tok/s (~8×) on stock 0.24.0. Kernel = aidendle94.

**c) Transplanted GB10 kernels.** DeepGEMM **sm_121a** `.so` and the compiled flashinfer sparse-MLA
(both **aidendle94**), the `sm12x` Triton sparse-MLA fallback (**CosmicRaisins**), and the GB10
indexer fallbacks (**hazyumps**). Full credit in [`../CREDITS.md`](../CREDITS.md).

---

## 3. Keys concurrency patches

**Stock DSpark forces `max_num_seqs=1`** — the speculative proposer stalls/corrupts at batch > 1
(it indexes its persistent draft KV by *batch row*, which is unstable across scheduler churn).
The **Keys concurrency patch** unlocks correct in-server concurrency:

- **Patch 1 — stable req-id slotting.** Key the persistent `main_kv_cache` by a **stable
  req-id slot** instead of the batch row, so draft KV survives batch reordering / preemption.
- **Patch 2 — ragged context path.** Use a ragged `query_start_loc` (mixed prefill+decode)
  instead of a rectangular `[B, seq, H]` layout on the rejected-context path
  (`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`).

Result: **byte-identical output under churn**, 0 errors staggered, and real concurrent throughput
(C=12 → ~255 tok/s aggregate; up to ~318 tok/s at C16 on the seqs=16 profile). Without it,
DSpark is single-sequence only. The patch is carried in the serving overlay and applies cleanly on top.

---

## 4. Acceptance-rate optimization (>60%)

DSpark speculative decoding runs `MTP_NUM_TOKENS=5` (5 draft tokens per step). Throughput ≈
`decode_step_rate × mean_accept_length`, so **acceptance is the primary speed lever**.

**Levers that get us to 60–67%:**
- **Compiled draft/MHC kernels** (aidendle94 base) — the biggest factor. The Triton draft path on
  stock 0.24.0 caps ~40%; the compiled path reaches 60–67%.
- **Deterministic serving defaults** — `temperature=0`, `top_p=1.0`, `thinking=false`,
  `--generation-config vllm` (also the agent-garble fix).
- **Warmup.** Acceptance **climbs as kernels JIT** over the first few requests:
  `31% → 34% → 65% → 67%`, mean accept length `2.6 → 4.6`, with a **shallow** per-position decay
  (`0.93 / 0.85 / 0.75 / 0.58 / 0.48`). Warm the endpoint before benchmarking or pointing agents at it.

Steady state: **61–67% draft acceptance**, mean accept length ~4.1.

---

## 5. `WO_PROJECTION=1` — sustaining acceptance at high concurrency

`VLLM_USE_B12X_WO_PROJECTION` was benchmarked head-to-head (RESULTS §3):

| | single-stream | acceptance @ concurrency | peak decode |
|---|---|---|---|
| WO=0 | ~56 tok/s (faster) | 58% | 312 tok/s |
| **WO=1** | ~51 tok/s | **67%** | **343 tok/s** |

WO=0 is marginally faster single-stream; **WO=1 sustains meaningfully higher DSpark acceptance
under concurrency (67% vs 58%)** and a higher peak decode rate. Because acceptance is what keeps
per-user latency reasonable when many sessions share the batch, **the standing config uses WO=1:
at high concurrency we value sustained acceptance over raw single-stream speed.**

---

## 6. Results

- **Concurrency sweep** (WO=0 vs WO=1, and the standing 1.5M/seqs12 config): RESULTS §3–4.
- **Context sweep** (decode flat ~47–50 tok/s vs context; TTFT grows): RESULTS §5.
- **Long-context needle sweep** (coherent recall 6k → **543,994 tokens**): RESULTS §6.
