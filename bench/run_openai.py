#!/usr/bin/env python3
"""Engine-neutral benchmark over an OpenAI-compatible /v1/chat/completions endpoint.
Same prompts (prompts/prompts.jsonl) and same GGUF for every engine (tobu, llama.cpp, halogen, ...).
Usage: python3 -I run_openai.py --url http://127.0.0.1:8080 --engine llamacpp-vulkan [--model X] [--runs 2] [--only id1,id2]
Writes results/<engine>-<timestamp>.jsonl and prints a table. Greedy (temperature 0, seed 0).
Metrics: ttft_s (time to first content/reasoning token, approximates prefill), decode_tps (tokens after first / time
after first), prompt_tokens/completion_tokens from usage (if the server sends it), output sha256 for cross-engine
exactness checks."""
import argparse, hashlib, json, os, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "prompts")

def load_prompts(only):
    out = []
    for line in open(os.path.join(PROMPTS, "prompts.jsonl"), encoding="utf-8"):
        if not line.strip():
            continue
        p = json.loads(line)
        if only and p["id"] not in only:
            continue
        if p.get("context_file"):
            ctx = open(os.path.join(PROMPTS, p["context_file"]), encoding="utf-8").read()
            for m in p["messages"]:
                m["content"] = m["content"].replace("{{CONTEXT}}", ctx)
        out.append(p)
    return out

def run_one(url, model, p, api_key, think, no_cache=False, extra=None):
    body = {"model": model, "messages": p["messages"], "max_tokens": p["max_tokens"], "temperature": 0,
            "top_p": 1, "seed": 0, "stream": True, "stream_options": {"include_usage": True}}
    if no_cache:  # llama.cpp-specific field; enable per engine (prefill must be measured cold)
        body["cache_prompt"] = False
    if extra:
        body.update(extra)
    if think is not None:
        body["chat_template_kwargs"] = {"enable_thinking": think}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    t0 = time.perf_counter(); t_first = None; n_chunks = 0; text = []; usage = None; timings = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("timings"):
                timings = ev["timings"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    n_chunks += 1; text.append(piece)
    t_end = time.perf_counter()
    out = "".join(text)
    ctoks = (usage or {}).get("completion_tokens") or n_chunks
    dec_t = (t_end - t_first) if t_first else None
    return {"id": p["id"], "kind": p["kind"], "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": ctoks, "ttft_s": round(t_first - t0, 3) if t_first else None,
            "decode_tps": round((ctoks - 1) / dec_t, 2) if dec_t and ctoks > 1 else None,
            "total_s": round(t_end - t0, 3), "server_timings": timings,
            "sha256": hashlib.sha256(out.encode()).hexdigest()[:16], "text": out}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080"); ap.add_argument("--engine", required=True)
    ap.add_argument("--model", default="default"); ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--only", default=""); ap.add_argument("--api-key-file", default="")
    ap.add_argument("--no-prompt-cache", action="store_true", help="send cache_prompt=false (llama.cpp/tobu)")
    ap.add_argument("--think", choices=["on", "off", "default"], default="off")
    ap.add_argument("--extra-body", default="", help="JSON merged into the request body, e.g. {\"drafter\":\"serial\"} for halogen")
    a = ap.parse_args()
    extra = json.loads(a.extra_body) if a.extra_body else None
    key = open(a.api_key_file).read().strip() if a.api_key_file else ""
    think = None if a.think == "default" else (a.think == "on")
    prompts = load_prompts(set(filter(None, a.only.split(","))))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"{a.engine}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    print(f"{'id':14} {'run':>3} {'p_tok':>6} {'c_tok':>5} {'ttft_s':>7} {'pp_tps':>8} {'tg_tps':>7} sha")
    with open(path, "w") as f:
        for p in prompts:
            for i in range(a.runs):
                r = run_one(a.url, a.model, p, key, think, a.no_prompt_cache, extra); r.update(engine=a.engine, run=i, think=a.think)
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
                pp = round(r["prompt_tokens"] / r["ttft_s"]) if r["prompt_tokens"] and r["ttft_s"] else None
                print(f"{r['id']:14} {i:>3} {str(r['prompt_tokens']):>6} {r['completion_tokens']:>5} "
                      f"{str(r['ttft_s']):>7} {str(pp):>8} {str(r['decode_tps']):>7} {r['sha256']}", flush=True)
    print("->", path)

if __name__ == "__main__":
    sys.exit(main())
