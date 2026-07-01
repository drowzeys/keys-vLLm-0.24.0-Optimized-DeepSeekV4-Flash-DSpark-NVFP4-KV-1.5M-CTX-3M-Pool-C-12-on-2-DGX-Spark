# vLLM 0.24.0 sparse-MLA port for GB10 (DeepSeek-V4-Flash-DSpark)

Stock vLLM **0.24.0** ships DeepSeek-V4 but **no DSpark**, and its sparse-MLA verify on GB10
(sm_121a) falls back to a slow Triton multi-query path (~375 ms/verify, ~3.4 tok/s). This
directory carries the drop-in that replaces that with aidendle94's **compiled** flashinfer
`sparse_mla_sm120_decode_dsv4` kernel, plus the two integration fixes that made it work.

`flashinfer_sparse_mla.py` — env-gated by `VLLM_USE_FLASHINFER_SPARSE_MLA=1`, signature-compatible
with the Triton `flash_mla_with_kvcache` so `deepseek_v4/nvidia/flashmla.py` swaps 1:1 (Triton
kept as fallback). Result on 2× GB10: verify ~375 ms → compiled, single-stream 3.4 → ~27 tok/s
(~8×), coherent, C16 ≈ 77 tok/s aggregate.

## The two fixes (both in `flashinfer_sparse_mla.py`)

**1. `mid_out` scratch split count (dual-cache).**
The kernel tiles **both** the SWA index set (`indices`) and the global-topk set
(`extra_indices_in_kvcache`), so the split-K scratch must be the **sum** of each set's tiles,
not `max()`:
```python
nsplit = cdiv(prim_topk, 64) + cdiv(extra_topk, 64)   # not max(prim, extra)
```
Single-cache (extra==0) reduces to the reference `cdiv(width, 64)`. Oversizing the bf16
scratch is correctness-safe (the kernel reads only the prefix it plans). External `mid_out`/
`mid_lse` are supplied only for `num_tokens <= 64` (the reference decode/prefill split).

**2. `out_lse` contiguity (wrapper head sizing).**
`BatchSparseMLAPagedAttentionWrapper` allocates its internal `_out_lse` as
`(max_num_tokens, max_num_heads)` and slices `[:num_tokens, :num_heads]` on `run()`. Building
it with `max_num_heads=128` while the model has 64 heads makes that last-dim slice
**non-contiguous** → the kernel's `out_lse.IsContiguous()` check fails (surfaces only once the
decode batch exceeds the old cap). Fix: construct with `max_num_heads = num_heads` **exact**
(matches the reference), and key the wrapper cache by `(d_v, num_heads)`.

## Status

The 0.24.0 port serves DSpark coherently on GB10 with the compiled verify kernel. It reaches
~27 tok/s single-stream and correct concurrency, but its **draft path** (Triton/fp8-einsum on
0.24) caps acceptance at ~40% — so for the >60-tok/s / 60–67%-acceptance production target we
run the aidendle94-based `dspark-nvfp4-stage-c` image (compiled draft/MHC kernels). This file
is published as the reference 0.24.0 integration and the two kernel fixes.
