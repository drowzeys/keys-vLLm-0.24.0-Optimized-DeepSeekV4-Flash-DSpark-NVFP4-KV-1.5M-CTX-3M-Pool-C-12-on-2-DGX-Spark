#!/usr/bin/env bash
# Force-clear ALL stale LLM-serve containers + GPU processes on THIS node, then wait until GPU is free.
KNOWN="dspark_gb10 glm_dspark dspark46995 clean_serve abl_serve vllm_unholy glm_qt vllm_node glm_dsv4 dsv4_serve"
for c in $KNOWN; do docker rm -f "$c" >/dev/null 2>&1; done
# also remove any container whose name matches serve patterns (catch-all)
for c in $(docker ps -aq 2>/dev/null); do
  n=$(docker inspect -f '{{.Name}}' "$c" 2>/dev/null | tr -d '/')
  echo "$n" | grep -qiE "dspark|glm_|clean_serve|abl_serve|vllm|dsv4" && docker rm -f "$c" >/dev/null 2>&1
done
# kill any process still holding the GPU (stray pids from dead containers)
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
# wait up to 60s for the GPU to report no compute apps
for i in $(seq 1 30); do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  [ -z "$apps" ] && { echo "GPU_CLEAR ($(hostname))"; exit 0; }
  sleep 2
done
echo "GPU_STILL_BUSY ($(hostname)): $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')"
exit 1
