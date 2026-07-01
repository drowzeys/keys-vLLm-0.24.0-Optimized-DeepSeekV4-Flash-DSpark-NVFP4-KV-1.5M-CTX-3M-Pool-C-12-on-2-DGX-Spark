#!/usr/bin/env bash
# Long-context sweep with needle retrieval: tests TTFT (prefill), decode tok/s,
# AND coherence (does it recall a needle placed early in a huge context).
# Usage: longctx.sh <ctx_tokens...>
printf "%-8s %-9s %-9s %-11s %-8s %s\n" "target" "ctx_tok" "ttft_s" "decode_t/s" "needle" "answer_snip"
for T in "$@"; do
python3 - "$T" <<'PY'
import sys,urllib.request,json,time
T=int(sys.argv[1])
needle="REMEMBER THIS FACT: the vault access code is ZEBRA-4271-OMEGA."
filler="The quick brown fox jumps over the lazy dog near the wide riverbank at dawn. "
reps=max(1,(T-40)//16)
prompt=needle+"\n\n"+filler*reps+"\n\nQuestion: What is the vault access code stated at the very beginning? Answer with ONLY the code."
body=json.dumps({"model":"deepseek-v4-flash-dspark","messages":[{"role":"user","content":prompt}],
  "max_tokens":32,"temperature":0,"stream":True,"stream_options":{"include_usage":True}}).encode()
req=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
t0=time.time(); ttft=None; ctx=0; ct=0; txt=""
with urllib.request.urlopen(req,timeout=900) as r:
    for line in r:
        line=line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        d=json.loads(line[5:])
        if d.get("usage"): ctx=d["usage"]["prompt_tokens"]; ct=d["usage"]["completion_tokens"]
        ch=d.get("choices") or []
        if ch:
            c=ch[0].get("delta",{}).get("content")
            if c:
                if ttft is None: ttft=time.time()-t0
                txt+=c
end=time.time()-t0
dec=ct/(end-ttft) if ttft and end>ttft else 0
hit="YES" if "ZEBRA-4271" in txt.replace(" ","") or "ZEBRA-4271-OMEGA" in txt else "NO"
snip=txt.strip().replace(chr(10)," ")[:40]
print("%-8d %-9d %-9.1f %-11.1f %-8s %s"%(T,ctx,ttft or 0,dec,hit,repr(snip)))
PY
done
