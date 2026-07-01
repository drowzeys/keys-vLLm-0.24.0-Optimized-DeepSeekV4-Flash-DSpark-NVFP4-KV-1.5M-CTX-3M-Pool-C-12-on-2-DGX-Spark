# Building the serving image

The runtime is `vllm-dspark-runtime:dspark-nvfp4-stage-c`, built as a chain on top of the
aidendle94 GB10 base. We do **not** redistribute the upstream overlay sources here; build
them from their public repos (all Apache-2.0 / public), with credit.

## Chain

```
ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready   (== aidendle94/sparkrun-vllm-ds4-gb10:production-ready)
        │  + rafaelcaricio DSpark overlay  (recipe/Dockerfile.dspark-runtime-overlay)
        ▼
vllm-dspark-runtime:mia-raf-pr1
        │  + nvfp4 stage-A  (register nvfp4_ds_mla dtype: cache.py / torch_utils.py / kv_cache_interface.py)
        ▼
…-nvfp4-a
        │  + nvfp4 stage-B  (DeepSeek-V4 probe: attention.py / nvidia/flashmla.py -> 416B)
        ▼
…-nvfp4-b
        │  + nvfp4 stage-C  (padded 584B envelope; deepseek_v4 page = storage_block_size * 584)
        ▼
vllm-dspark-runtime:dspark-nvfp4-stage-c   <-- serve this
```

The overlay + nvfp4 stage A/B/C Dockerfiles are published by **tonyd2wild**:
`tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark`. Clone that repo and run its
`build-dspark-vllm-runtime.sh` on each node, or the equivalent:

```bash
BASE=vllm-dspark-runtime:mia-raf-pr1
docker build -f recipe/Dockerfile.dspark-runtime-overlay -t $BASE recipe/overlay
docker build --build-arg BASE_IMAGE=$BASE          -f recipe/nvfp4/Dockerfile.stage-a -t $BASE-nvfp4-a .
docker build --build-arg BASE_IMAGE=$BASE-nvfp4-a  -f recipe/nvfp4/Dockerfile.stage-b -t $BASE-nvfp4-b .
docker build --build-arg BASE_IMAGE=$BASE-nvfp4-b  -f recipe/nvfp4/Dockerfile.stage-c -t vllm-dspark-runtime:dspark-nvfp4-stage-c .
```

Build on **both** nodes (or build once and `docker save | ssh … docker load`).

## What `nvfp4_ds_mla` actually is (stage-C)

Stage-C keeps DeepSeek-V4's proven **584-byte** cache envelope (same page bytes as `fp8_ds_mla`)
but routes through the `nvfp4_ds_mla` path. This is the **padded** NVFP4 probe — not the
unresolved true-layout 416-byte kernel (which fails past ~411 real prompt tokens). The padded
path is what boots stably at 1M+ and delivers the large KV pool in this repo's results.

## vLLM 0.24.0 note

The serving image above is the aidendle94 base (vLLM ~0.21/0.22 lineage, which carries the
compiled DSpark draft/MHC kernels → 60–67% acceptance). A **separate** port of DSpark onto
**stock vLLM 0.24.0** for GB10 is included under `vllm-0.24-port/` (the compiled sparse-MLA
drop-in + the two kernel-integration fixes). See `vllm-0.24-port/PORT_NOTES.md`.
