"""FlashInfer compiled sparse-MLA (sparse_mla_sm120_decode_dsv4) drop-in for GB10.
Replaces the slow Triton sm12x multi-query verify with aidendle94's compiled
flashinfer kernel (SM120/SM121). Env-gated by VLLM_USE_FLASHINFER_SPARSE_MLA.
Signature matches flash_mla_with_kvcache_triton so flashmla.py can swap 1:1."""
import torch

_DECODE_SPLIT_TILE = 64
# Match the reference (sparse_mla_sm120.py): only supply external mid_out/mid_lse
# scratch on the small-batch decode path. Above this the kernel takes its own
# (contiguous) internal path — passing our tensors trips "out_lse must be
# contiguous". Spec-verify batches can exceed 64 tokens at higher max_num_seqs.
_DECODE_MAX_TOKENS = 64
_wrapper_cache = {}


def _cdiv(a, b):
    return (a + b - 1) // b


def _get_wrapper(max_tokens, num_heads, d_v):
    from flashinfer import BatchSparseMLAPagedAttentionWrapper
    # max_num_heads MUST equal the actual num_heads (not padded to 128): the
    # wrapper allocates its internal out_lse as (max_num_tokens, max_num_heads)
    # and slices out_lse[:num_tokens, :num_heads] on run(). Padding max_num_heads
    # above num_heads makes that last-dim slice non-contiguous, tripping the
    # kernel's "out_lse must be contiguous" check. Matches reference
    # sparse_mla_sm120.py:306 (max_num_heads=num_heads).
    key = (int(d_v), int(num_heads))
    w = _wrapper_cache.get(key)
    if w is None:
        w = BatchSparseMLAPagedAttentionWrapper(
            max_num_tokens=max(int(max_tokens), 8192),
            max_num_heads=int(num_heads),
            d_v=int(d_v),
        )
        _wrapper_cache[key] = w
    return w


def flash_mla_with_kvcache_flashinfer(
    q, k_cache, block_table, cache_seqlens=None, head_dim_v=512,
    tile_scheduler_metadata=None, num_splits=None, softmax_scale=None,
    causal=False, is_fp8_kvcache=False, indices=None, attn_sink=None,
    extra_k_cache=None, extra_indices_in_kvcache=None, topk_length=None,
    extra_topk_length=None, out=None,
):
    output = out
    if output is not None and output.dim() == 4:
        output = output.squeeze(1)
    # output is [num_tokens, num_heads, d_v]
    num_tokens = output.shape[0]
    num_heads = output.shape[1]
    d_v = output.shape[2]
    prim_topk = indices.shape[-1] if indices is not None else 0
    extra_topk = extra_indices_in_kvcache.shape[-1] if extra_indices_in_kvcache is not None else 0
    # Dual-cache: the kernel tiles the SWA index set AND the extra (global topk)
    # index set, so the required split count is the SUM of each set's own
    # cdiv-tiles, not max(). Single-cache (extra_topk==0) reduces to the
    # reference's cdiv(indices_width, 64). Oversizing this bf16 scratch is
    # correctness-safe: the kernel only writes/reads the prefix it plans.
    nsplit_prim = _cdiv(prim_topk, _DECODE_SPLIT_TILE) if prim_topk else 0
    nsplit_extra = _cdiv(extra_topk, _DECODE_SPLIT_TILE) if extra_topk else 0
    nsplit = max(1, nsplit_prim + nsplit_extra)
    # Only the small-batch decode path takes external scratch (matches reference).
    # For larger batches the kernel allocates its own; supplying ours fails the
    # internal out_lse contiguity check.
    if num_tokens <= _DECODE_MAX_TOKENS:
        mid_out = torch.empty((num_tokens, num_heads, nsplit, d_v), dtype=torch.bfloat16, device=q.device)
        mid_lse = torch.empty((num_tokens, num_heads, nsplit), dtype=torch.float32, device=q.device)
    else:
        mid_out = None
        mid_lse = None
    w = _get_wrapper(max(num_tokens, 8192), num_heads, d_v)
    w.run(
        q=q, kv_cache=k_cache, indices=indices, output=output, sm_scale=softmax_scale,
        topk_length=topk_length, attn_sink=attn_sink,
        extra_kv_cache=extra_k_cache, extra_indices=extra_indices_in_kvcache,
        extra_topk_length=extra_topk_length, mid_out=mid_out, mid_lse=mid_lse,
    )
    return out, None
