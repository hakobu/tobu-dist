#!/usr/bin/env python3
"""tobu bench: tobu vs. llama.cpp (and halogen, if you run it) on one Strix Halo box, same GGUF, same prompts.

Method (the one behind the numbers in the README):
  * engines run one after another, never at the same time (each needs most of the unified memory);
  * per prompt: one warmup run that is discarded, then --reps runs back-to-back (no pauses), median reported;
  * GPU clock (sclk) and package power are sampled every 50 ms during every run and stored per run;
  * cold prefill: prompt caches off (cache_prompt=false; tobu also with its answer memory off);
  * greedy (temperature 0, seed 0), thinking off, streaming; output sha256 per run -> "same output on repeat".
Output: <out>/bench.json (all runs + medians + system info) and <out>/card.png (needs matplotlib).

  ./tobu bench                                   # tobu (started by the bench) only
  ./tobu bench --llama-server ~/llama.cpp/build/bin/llama-server   # + llama.cpp (Vulkan or ROCm build), started by the bench
  ./tobu bench --llamacpp-url http://127.0.0.1:8081 --halogen-url http://127.0.0.1:8082   # servers you started yourself
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_openai import load_prompts, run_one  # noqa: E402



def load_install_env(path: str = os.path.join(ROOT, ".tobu.env")) -> None:
    """Settings saved by install.sh (TOBU_GGUF, TOBU_MTP, TOBU_IMAGE, ...); variables already set in the environment win."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                k, sep, v = line.rstrip("\n").partition("=")
                if sep and k.startswith("TOBU_") and not os.environ.get(k):
                    os.environ[k] = v
    except FileNotFoundError:
        pass


load_install_env()
DEFAULT_GGUF = os.path.expanduser(os.environ.get(
    "TOBU_GGUF", "~/models/Qwen3.8-Flash-Next-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"))
PREFILL = ("ctx-8k", "ctx-32k")


# ---- GPU telemetry (amdgpu sysfs; missing files just give None) -------------------------------------------------
def _gpu_dev() -> str:
    for d in sorted(glob.glob("/sys/class/drm/card*/device")):
        if os.path.exists(d + "/pp_dpm_sclk"):
            return d
    return ""


DEV = _gpu_dev()
HW = (glob.glob(DEV + "/hwmon/hwmon*") or [""])[0] if DEV else ""


def sclk() -> int | None:
    try:
        for line in open(DEV + "/pp_dpm_sclk"):
            if "*" in line:
                return int(line.split()[1].lower().rstrip("mhz"))
    except Exception:
        return None
    return None


def power() -> float | None:
    for f in ("power1_average", "power1_input"):
        try:
            return int(open(f"{HW}/{f}").read()) / 1e6
        except Exception:
            pass
    return None


def system_info() -> dict:
    def rd(p):
        try:
            return open(p).read().strip()
        except Exception:
            return None
    cpu = None
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    cap = rd(f"{HW}/power1_cap") if HW else None
    return {"cpu": cpu, "kernel": platform.release(), "gpu_sysfs": bool(DEV),
            "power_cap_w": round(int(cap) / 1e6, 1) if cap and cap.isdigit() else None,
            "power_profile": rd(DEV + "/power_dpm_force_performance_level") if DEV else None,
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.halt = False
        self.c: list[int] = []
        self.p: list[float] = []

    def run(self):
        while not self.halt:
            c, p = sclk(), power()
            if c:
                self.c.append(c)
            if p:
                self.p.append(p)
            time.sleep(0.05)


# ---- engines ----------------------------------------------------------------------------------------------------
def healthy(url: str) -> bool:
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(url.rstrip("/") + path, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
    return False


def wait_up(url: str, proc: subprocess.Popen | None, timeout: float) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if healthy(url):
            return
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        time.sleep(2)
    raise RuntimeError(f"{url} not healthy after {timeout:.0f} s")


class Engine:
    """name, url; start()/stop() only if the bench owns the server."""

    def __init__(self, name, url, start_cmd=None, stop_cmd=None, env=None, no_cache=True, bust_cache=False,
                 status_cmd=None, user_env=None):
        self.name, self.url, self.start_cmd, self.stop_cmd = name, url, start_cmd, stop_cmd
        self.env, self.no_cache, self.bust_cache, self.proc = env or {}, no_cache, bust_cache, None
        self.status_cmd, self.user_env, self.restore = status_cmd, user_env or {}, False

    def _own_server_running(self) -> bool:
        """True if scripts/serve.sh reports its own server running on this port (not some other process)."""
        if not self.status_cmd:
            return False
        r = subprocess.run(self.status_cmd, env={**os.environ, **self.env}, capture_output=True, text=True)
        return r.returncode == 0 and r.stdout.startswith("running")

    def start(self, timeout):
        if self.start_cmd is None:
            if not healthy(self.url):
                raise RuntimeError(f"{self.name}: nothing answers at {self.url}")
            return
        if healthy(self.url):
            if not self._own_server_running():
                raise RuntimeError(f"{self.name}: {self.url} is used by another server. Stop it, measure it with "
                                   f"--tobu-running, or let the bench use another port with --tobu-url "
                                   f"http://127.0.0.1:8090")
            print(f"{self.name}: your tobu server on {self.url} is stopped for the bench (it runs with answer memory "
                  f"off for fair decode numbers) and restarted afterwards", flush=True)
            subprocess.run(self.stop_cmd, env={**os.environ, **self.env})
            self.restore = True
        env = {**os.environ, **self.env}
        if self.stop_cmd:          # script that daemonises itself (scripts/serve.sh)
            subprocess.run(self.start_cmd, env=env, check=True)
            wait_up(self.url, None, timeout)
        else:
            self.proc = subprocess.Popen(self.start_cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            wait_up(self.url, self.proc, timeout)

    def stop(self):
        if self.stop_cmd:
            subprocess.run(self.stop_cmd, env={**os.environ, **self.env})
        elif self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        time.sleep(5)              # let the GPU memory go before the next engine loads

    def restart_user_server(self):
        """Bring back the server the bench stopped, with the user's own settings."""
        if self.restore:
            print(f"{self.name}: restarting your server on {self.url}", flush=True)
            env = {**os.environ, **self.user_env}
            subprocess.run(self.start_cmd, env=env)
            self.restore = False


def measure(eng: Engine, prompts, reps: int, emit) -> None:
    for p in prompts:
        for i in range(reps + 1):
            q = json.loads(json.dumps(p))
            if eng.bust_cache and p["id"] in PREFILL:   # engines without a cache switch: new prefix every run
                q["messages"][0]["content"] = f"[run {time.time_ns()}]\n" + q["messages"][0]["content"]
            s = Sampler()
            s.start()
            r = run_one(eng.url, "default", q, "", False, no_cache=eng.no_cache)
            s.halt = True
            s.join()
            r.pop("text", None)
            r.update(engine=eng.name, warmup=(i == 0), cache_busted=eng.bust_cache and p["id"] in PREFILL,
                     prefill_tps=round(r["prompt_tokens"] / r["ttft_s"]) if r.get("prompt_tokens") and r.get("ttft_s") else None,
                     sclk_mean=round(statistics.mean(s.c)) if s.c else None, sclk_min=min(s.c) if s.c else None,
                     sclk_max=max(s.c) if s.c else None,
                     power_mean=round(statistics.mean(s.p), 1) if s.p else None,
                     power_max=round(max(s.p), 1) if s.p else None)
            emit(r)


def summarize(rows, engines, prompts) -> dict:
    out = {}
    for e in engines:
        per = {}
        for p in prompts:
            rs = [r for r in rows if r["engine"] == e and r["id"] == p["id"] and not r["warmup"]]
            if not rs:
                continue
            metric = "prefill_tps" if p["id"] in PREFILL else "decode_tps"
            vals = [r[metric] for r in rs if r.get(metric) is not None]
            shas = {r["sha256"] for r in rs}
            per[p["id"]] = {"metric": metric, "runs": vals, "median": statistics.median(vals) if vals else None,
                            "ttft_s_median": statistics.median(r["ttft_s"] for r in rs if r.get("ttft_s")),
                            "same_output": len(shas) == 1 if not rs[0].get("cache_busted") else None,
                            "sha256": sorted(shas), "published_sha256": p.get("tobu_sha256"),
                            "power_w_mean": [r["power_mean"] for r in rs],
                            "sclk_mhz_mean": [r["sclk_mean"] for r in rs]}
        out[e] = per
    return out


def card(summary, info, path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    engines = list(summary)
    colors = {"tobu": "#2563eb", "llamacpp": "#9ca3af", "halogen": "#f59e0b"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 5]})
    for ax, ids, title in ((axes[0], [i for i in PREFILL], "cold prefill, tok/s"),
                           (axes[1], None, "decode, tok/s (greedy)")):
        ids = ids or [i for i in next(iter(summary.values())).keys() if i not in PREFILL]
        w = 0.8 / max(1, len(engines))
        for k, e in enumerate(engines):
            ys = [(summary[e].get(i) or {}).get("median") or 0 for i in ids]
            xs = [j + (k - (len(engines) - 1) / 2) * w for j in range(len(ids))]
            bars = ax.bar(xs, ys, w, label=e, color=colors.get(e, None))
            for b, y in zip(bars, ys):
                if y:
                    ax.text(b.get_x() + b.get_width() / 2, y, f"{y:,.0f}" if y >= 100 else f"{y:.1f}",
                            ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(ids)), ids, rotation=20)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    same = ", ".join(f"{e}: {sum(1 for v in summary[e].values() if v['same_output'])}/"
                     f"{sum(1 for v in summary[e].values() if v['same_output'] is not None)} same output"
                     for e in engines)
    pw = [w for e in engines for v in summary[e].values() for w in v["power_w_mean"] if w]
    capw = f"power cap {info['power_cap_w']} W, " if info.get("power_cap_w") is not None else ""
    fig.suptitle(f"tobu bench – {info.get('cpu') or 'unknown CPU'}  |  {capw}"
                 f"measured mean {statistics.mean(pw):.0f} W  |  {info['date']}" if pw else
                 f"tobu bench – {info.get('cpu') or 'unknown CPU'}  |  {info['date']}", fontsize=10)
    fig.text(0.01, 0.01, f"median of runs after a discarded warmup; {same}", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=130)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", default=DEFAULT_GGUF, help="model GGUF (first shard); env TOBU_GGUF")
    ap.add_argument("--reps", type=int, default=4, help="measured runs per prompt after one discarded warmup")
    ap.add_argument("--quick", action="store_true",
                    help="quick check: 1 measured run per prompt (same as --reps 1, about 3 min); hashes are still checked")
    ap.add_argument("--only", default="", help="comma-separated prompt ids (default: all 7)")
    ap.add_argument("--tobu-url", default="http://127.0.0.1:8080")
    ap.add_argument("--tobu-running", action="store_true",
                    help="use a tobu server you started yourself (start it with TOBU_FNX_XSTORE_TOKENS=0 for fair decode)")
    ap.add_argument("--no-tobu", action="store_true")
    ap.add_argument("--llama-server", default="", help="path to llama-server; the bench starts it on --llamacpp-url")
    ap.add_argument("--llamacpp-url", default="", help="running llama.cpp server (default with --llama-server: :8081)")
    ap.add_argument("--halogen-url", default="", help="running halogen server (optional; its prompt cache is busted "
                                                      "with a per-run prefix line for the prefill prompts)")
    ap.add_argument("--start-timeout", type=float, default=600)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "bench-" + time.strftime("%Y%m%d-%H%M%S")))
    a = ap.parse_args()
    if a.quick:
        a.reps = 1

    engines: list[Engine] = []
    if not a.no_tobu:
        if a.tobu_running:
            engines.append(Engine("tobu", a.tobu_url))
        else:
            port = a.tobu_url.rsplit(":", 1)[-1].rstrip("/")
            serve = os.path.join(ROOT, "scripts", "serve.sh")
            env = {"TOBU_GGUF": a.gguf, "TOBU_PORT": port, "TOBU_FNX_XSTORE_TOKENS": "0"}
            engines.append(Engine("tobu", a.tobu_url, [serve, "start"], [serve, "stop"], env,
                                  status_cmd=[serve, "status"], user_env={"TOBU_PORT": port, "TOBU_GGUF": a.gguf}))
    if a.llama_server:
        url = a.llamacpp_url or "http://127.0.0.1:8081"
        port = url.rsplit(":", 1)[-1].rstrip("/")
        cmd = [a.llama_server, "-m", a.gguf, "-ngl", "99", "-c", "65536", "-np", "1", "--jinja", "-fa", "on",
               "-ub", "2048", "-b", "2048", "--host", "127.0.0.1", "--port", port]
        engines.append(Engine("llamacpp", url, cmd))
    elif a.llamacpp_url:
        engines.append(Engine("llamacpp", a.llamacpp_url))
    if a.halogen_url:
        engines.append(Engine("halogen", a.halogen_url, bust_cache=True))
    if not engines:
        ap.error("no engine selected")
    if any(e.start_cmd for e in engines) and not os.path.exists(a.gguf):
        ap.error(f"GGUF not found: {a.gguf} (install.sh downloads it; or pass --gguf / TOBU_GGUF)")

    prompts = load_prompts(set(filter(None, a.only.split(","))))
    os.makedirs(a.out, exist_ok=True)
    info = system_info()
    cap = (f"{info['power_cap_w']} W" if info["power_cap_w"] is not None
           else "not exposed by this kernel (package power is logged per run)")
    print(f"power cap {cap}, profile {info['power_profile']}, engines "
          f"{', '.join(e.name for e in engines)}, {len(prompts)} prompts x (1 warmup + {a.reps})", flush=True)
    rows: list[dict] = []
    raw = open(os.path.join(a.out, "runs.jsonl"), "w")

    def emit(r):
        rows.append(r)
        raw.write(json.dumps(r) + "\n")
        raw.flush()
        m = r["prefill_tps"] if r["id"] in PREFILL else r["decode_tps"]
        print(f"{r['engine']:9} {r['id']:14} {'warm' if r['warmup'] else 'meas'} {m!s:>8} tok/s  "
              f"sclk {r['sclk_mean']} MHz  {r['power_mean']} W  {r['sha256']}", flush=True)

    try:
        for e in engines:
            print(f"== {e.name}: starting" if e.start_cmd else f"== {e.name}: {e.url}", flush=True)
            e.start(a.start_timeout)
            try:
                measure(e, prompts, a.reps, emit)
            finally:
                if e.start_cmd:
                    e.stop()
    except RuntimeError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    finally:
        raw.close()
        for e in engines:
            e.restart_user_server()

    summary = summarize(rows, [e.name for e in engines], prompts)
    res = {"tool": "tobu bench", "system": info, "reps": a.reps, "warmup_discarded": 1, "gguf": os.path.basename(a.gguf),
           "summary": summary, "runs": rows}
    with open(os.path.join(a.out, "bench.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\nmedians (prefill: tok/s cold; decode: tok/s)")
    for e, per in summary.items():
        for pid, v in per.items():
            same = "" if v["same_output"] is None else ("same output" if v["same_output"] else "OUTPUT DIFFERS")
            if e == "tobu" and v.get("published_sha256"):
                same += ", matches published hash" if v["sha256"] == [v["published_sha256"]] else ", DIFFERS FROM PUBLISHED HASH"
            print(f"{e:9} {pid:14} {v['median']!s:>8}  runs {v['runs']}  {same}")
    png = os.path.join(a.out, "card.png")
    print("->", os.path.join(a.out, "bench.json"), "and", png if card(summary, info, png) else "(no card: pip install matplotlib)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
