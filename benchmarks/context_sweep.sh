printf "%-8s %-9s %-9s %-13s %-9s\n" "target" "ctx_tok" "ttft_s" "decode_t/s" "accept%"
for T in 512 1024 2048 4096 6144; do
python3 - "$T" <<'PY'
import sys,urllib.request,json,time,subprocess
T=int(sys.argv[1]); reps=max(1,T//24)
nonce=f"# module build {T*7+13}\n"
code=("def process_item_%d(data, index):\n    total = sum(x*index for x in data)\n    return {'id': index, 'total': total, 'ok': total > 0}\n\n")
prompt=nonce+"".join(code%i for i in range(reps))+"\n# Continue this module: add a class BatchProcessor with methods add, run, and summarize, fully implemented:\n"
body=json.dumps({"model":"deepseek-v4-flash-dspark","messages":[{"role":"user","content":prompt}],
  "max_tokens":200,"min_tokens":200,"ignore_eos":True,"temperature":0,
  "stream":True,"stream_options":{"include_usage":True}}).encode()
req=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
t0=time.time(); ttft=None; ctx=0; ct=0
with urllib.request.urlopen(req,timeout=180) as r:
    for line in r:
        line=line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        d=json.loads(line[5:])
        if d.get("usage"): ctx=d["usage"]["prompt_tokens"]; ct=d["usage"]["completion_tokens"]
        ch=d.get("choices") or []
        if ch and ch[0].get("delta",{}).get("content") and ttft is None: ttft=time.time()-t0
end=time.time()-t0
dec=ct/(end-ttft) if ttft and end>ttft else 0
acc=subprocess.run("docker logs dspark-nvfp4 2>&1 | grep -oE 'Avg Draft acceptance rate: [0-9.]+%' | tail -1",shell=True,capture_output=True,text=True).stdout.strip().split()[-1]
print("%-8d %-9d %-9.2f %-13.1f %-9s"%(T,ctx,ttft or 0,dec,acc))
PY
done
