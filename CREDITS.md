# Credits & Acknowledgments

This recipe would not exist without the frontier work of the following people and
projects. **Special thanks** — this repo builds directly on top of their contributions.

## ⭐ Special credits

- **tonyd2wild** — the `nvfp4_ds_mla` 1M NVFP4-KV stage recipe (stage A/B/C), the two-node
  DGX Spark packaging, the Keys-concurrency integration, and the agent-garble-fix serving
  defaults. This entire 1.5M-context NVFP4 profile is built on tonyd2wild's stage-C padded
  NVFP4 path.
  → `tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark`

- **MiaAI-Lab** — DSpark packaging for DGX Spark and the DSpark 2× DGX Spark recipe that the
  overlay and NVFP4 stages build upon.
  → `MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`

- **Rafael Caricio** (`rafaelcaricio`) — the DSpark speculative-decoding vLLM integration /
  overlay (`rafaelcaricio/vllm`) that provides the DSpark proposer and draft path.

- **vLLM** (the vLLM project) — the inference engine everything runs on. Licensed under
  Apache-2.0. → https://github.com/vllm-project/vllm

## Also

- **aidendle94** — the GB10 (sm_121a) `sparkrun-vllm-ds4` runtime base image carrying the
  compiled DSpark draft/MHC and sparse-MLA kernels that make 60–67% acceptance possible.
- **deepseek-ai** — `DeepSeek-V4-Flash-DSpark`, the model.
- **hazyumps** — GB10 sm12x indexer fallbacks referenced during the vLLM 0.24.0 port.

## What this repo adds

Original artifacts only: the 1.5M/util-0.85/seqs-12/WO=1 serving launcher with RoCE/IB
networking, the GPU-clear/verify ops tooling, an independent benchmark harness with the full
concurrency / context / long-context (6k→512k) results, and the vLLM **0.24.0** compiled
sparse-MLA port drop-in with its two kernel-integration fixes.

All upstream components are Apache-2.0 / publicly published; this repo does not redistribute
their overlay sources or model weights — see `docs/BUILD.md` and `NOTICE`.
