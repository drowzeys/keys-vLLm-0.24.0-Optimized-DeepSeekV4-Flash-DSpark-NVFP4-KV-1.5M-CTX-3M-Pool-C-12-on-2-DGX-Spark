#!/usr/bin/env bash
# Concurrency sweep for dspark-nvfp4. Usage: sweep.sh <tag> <C1> <C2> ...
SB="${OUTDIR:-/tmp/dspark-bench}"; mkdir -p "$SB"
TAG="$1"; shift
PROMPT="Implement a thread-safe LRU cache in Python with get and put using OrderedDict, with type hints and a short usage example, then explain the time complexity."
run_conc() {
  local C=$1 start end wall tot; start=$(date +%s.%N); pids=""
  for n in $(seq 1 $C); do
    curl -s -m 200 http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"deepseek-v4-flash-dspark\",\"messages\":[{\"role\":\"user\",\"content\":\"Variant $n: $PROMPT\"}],\"max_tokens\":256,\"min_tokens\":256,\"ignore_eos\":true,\"temperature\":0}" \
      -o "$SB/${TAG}_${C}_$n.json" & pids="$pids $!"
  done
  wait $pids; end=$(date +%s.%N)
  wall=$(python3 -c "print(f'{$end-$start:.2f}')")
  tot=$(python3 -c "import json,glob;print(sum(json.load(open(f)).get('usage',{}).get('completion_tokens',0) for f in glob.glob('$SB/${TAG}_${C}_*.json')))")
  python3 -c "agg=$tot/$wall; print(f'C{$C:>2}: wall={$wall:.1f}s tok={$tot:>5} | aggregate={agg:6.1f} tok/s | per-req={agg/$C:5.1f} tok/s')"
}
for C in "$@"; do run_conc $C; done
