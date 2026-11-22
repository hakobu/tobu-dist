"""tobu server: one process, one engine (tobu-fnx, Qwen3.8-Flash-Next on gfx1151), two interfaces.

    python server/server.py --engine flashnext --gguf <model-00001-of-0000N.gguf> --fnx-mtp <mtp.gguf> --slots 2
    python server/server.py --engine toy --tokenizer <tokenizer.json> --port 8095        # tests without GPU

POST /v1/chat/completions   OpenAI Chat Completions (stream, tools, stop, max_tokens, response_format; thinking via
                            chat_template_kwargs.enable_thinking or reasoning_effort)
                            thinking budget (TOB-32): thinking_budget (top level or in chat_template_kwargs; Anthropic:
                            thinking.budget_tokens), else --default-thinking-budget; think_budget.py
                            TOB-33: effective min(budget, max_tokens - reserve), --thinking-answer-reserve
                            script ban: ban_scripts ["cjk"] (tobu extension, both formats), else
                            --ban-scripts, unless the messages themselves contain CJK (api.ban_scripts_for)
POST /v1/messages           Anthropic Messages (stream, tools, stop_sequences; thinking via thinking.type)
POST /v1/messages/count_tokens   Anthropic: prompt tokens
GET  /v1/models, /props, /health (503 until the warm-up after loading is done; POST requests likewise, TOB-30)
GET  /tobu/metrics          metrics per request (no contents)
GET  /, /chat               browser chat; /chat/info, /chat/ready/tobu for the page
    Header X-Tobu-Arm: A|M|B|C (else --arm; A = without drafts), X-Tobu-Cache: off = cold start for this request only,
    X-Tobu-Tool-Grammar: off = without tool grammar, X-Tobu-Spec-Sampling: off|one (measurement),
    X-Tobu-Sampling: greedy|model, X-Tobu-Harness: name of the client for the log.
    Answers in the original format (api.py), cache hits in the usage, extension "tobu" (arm, hashes of the prompt and
    output tokens, sampling, ignored parameters) and for OpenAI "timings". Sampling: without settings the values of the
    model card per mode (sampling.py), temperature=0 -> greedy (bit-identical, with speculation); with temperature > 0
    the engine drafts at full width and checks exactly in the distribution (speculative sampling, mode A);
    --greedy-harness: harnesses whose requests without temperature run greedy (mode G).
Streaming: text arrives as soon as the engine has it, one packet per token; until the first token SSE comments as
keepalive, for Anthropic ping events afterwards. If the client aborts, generation ends after the current round.
Default only on 127.0.0.1; with --token or TOBU_TOKEN only with session token (Bearer / x-api-key); the browser chat
is opened once as /chat?token=<token>, which redirects to /chat and sets the HttpOnly, SameSite=Strict cookie
tobu_session (an HMAC of the token, not the token); the page never contains the token. Log (--log): per request counters and
token hashes, no contents.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import http.cookies
import secrets
import json
import socket
import os
import queue
import sys
import threading
import time
import collections
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


class AgentHTTPServer(ThreadingHTTPServer):
    """TOB-8: listen backlog for 64+ concurrent connections (default 5: with many parallel clients the
    kernel resets connections, "Connection reset by peer", before a request reaches the server)."""
    request_queue_size = 128


CHAT_HTML = None    # browser chat: None = HERE/"chat.html"
CHAT_COOKIE = "tobu_session"    # chat session cookie with --token (HMAC of the token, HttpOnly, SameSite=Strict)
CHAT_CSP = ("default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; "
            "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
sys.path.insert(0, str(HERE))

from decoder import generate                    # noqa: E402
from hftok import HFTok                         # noqa: E402

import api                                      # noqa: E402
import tool_format                              # noqa: E402
import chat                                     # noqa: E402
import sampling                                 # noqa: E402
import think_budget                             # noqa: E402  TOB-32: thinking budget

# One worker thread for loading, warm-up and (without --slots) the requests.
WORKER = ThreadPoolExecutor(max_workers=1)
STARTING_RETRY_S = 5                            # TOB-30: Retry-After of the 503 answer during loading/warm-up
JOBS = [WORKER]                                 # L49: with --slots N > 1 own pool with N threads
REP_MIN, REP_PMAX = 256, 64                     # repetition: check only from this many tokens on, periods up to REP_PMAX


def parse_ban_scripts(spec: str) -> list[str]:
    """--ban-scripts: comma-separated scripts (known: api.SCRIPTS), empty = none."""
    out = sorted({x.strip() for x in spec.split(",") if x.strip()})
    bad = [x for x in out if x not in api.SCRIPTS]
    if bad:
        raise SystemExit(f"--ban-scripts: unknown script {bad[0]!r} (known: {', '.join(api.SCRIPTS)})")
    return out


def _phase_stats(res, toks: list[int], thinking: bool, ids: dict) -> dict:
    """Per phase of the output (detected by token ids: thinking up to </think> or <tool_call>, tool call between
    <tool_call> and </tool_call>, else answer): rounds, tokens, check time, draft and accepted tokens per source.
    A round counts toward the phase of its first token. Evaluable the same way for all arms."""
    if ids.get("tc") is None:
        return {}
    phase, ph = [], "think" if thinking else "answer"
    for tk in toks:
        phase.append(ph)
        if ph == "think" and tk in (ids["te"], ids["tc"]):
            ph = "tool" if tk == ids["tc"] else "answer"
            if tk == ids["tc"]:
                phase[-1] = "tool"
        elif ph == "answer" and tk == ids["tc"]:
            phase[-1] = ph = "tool"
        elif ph == "tool" and tk == ids["tce"]:
            ph = "answer"
    out, pos = {}, 1
    for r in res.rounds:
        if pos >= len(phase):
            break
        s = out.setdefault(phase[pos], {"rounds": 0, "tokens": 0, "verify_ms": 0.0, "drafted": 0, "accepted": {},
                                        "by_len": {}, "by_pos": {}})
        bp = s["by_pos"].setdefault(r.src or "-", [])    # L38: per source and draft position [reached, accepted]
        for i in range(min(r.n_draft, r.n_acc + 1)):
            if len(bp) <= i:
                bp.append([0, 0])
            bp[i][0] += 1
            bp[i][1] += i < r.n_acc
        b = s["by_len"].setdefault(str(min(r.n_draft, 32)), [0, 0, 0.0])   # draft length: rounds, accepted, ms
        b[0] += 1
        b[1] += r.n_acc
        b[2] = round(b[2] + r.t * 1000, 1)
        s["rounds"] += 1
        s["tokens"] += r.n_acc + 1
        s["verify_ms"] = round(s["verify_ms"] + r.t * 1000, 1)
        s["drafted"] += r.n_draft
        if r.n_acc:
            s["accepted"][r.src or "-"] = s["accepted"].get(r.src or "-", 0) + r.n_acc
        pos += r.n_acc + 1
    return {"phases": out}


def repeating(toks: list[int], n_new: int) -> bool:
    """Endless repetition at the end of the output: the last max(8·p, 128) tokens are periodic with period p
    (p = 1..REP_PMAX), i.e. at least 8 or 128/p equal repetitions. Checked roughly every 16 tokens."""
    n = len(toks)
    if n < REP_MIN or (n // 16) == ((n - n_new) // 16):
        return False
    for p in range(1, REP_PMAX + 1):
        L = max(8 * p, 128)
        if n >= L + p and toks[n - L:] == toks[n - L - p: n - p]:
            return True
    return False
KEEPALIVE_S = 10
ARMS = ("A", "M", "B", "C")



class Backend:
    def __init__(self, a):
        self.arm_default = a.arm
        self.rw_lock = threading.Lock()
        self.pending, self.last_req_t, self.rewarmed = 0, time.time(), True
        self.rewarm_after = float(getattr(a, "rewarm_after", 0) or 0)
        self.recent = collections.deque(maxlen=max(1, int(getattr(a, "rewarm_n", 4))))
        self.rewarm_stats: dict = {}
        self.template = a.template or None
        self.kmax = a.kmax or 8
        style = "qwen35"
        if a.engine == "flashnext":                # Qwen3.8-Flash-Next on gfx1151 (tobu-fnx, engine/)
            from engine_flashnext import FlashNextEngine

            self.eng = FlashNextEngine(a.gguf, exe=getattr(a, "fnx_exe", ""), mtp=getattr(a, "fnx_mtp", ""),
                                       max_len=a.max_len, kmax=self.kmax, sessions=max(1, a.slots))
            self.tok = HFTok(self.eng.hf_tok)
            self.model = a.model or "qwen38-flash-next"
            # template knows reasoning_effort xhigh/medium/low/none (default xhigh = very long thinking): if not given
            # medium, "high" (rejected by the template) -> medium; budget cap per level in api.EFFORT_BUDGET
            api.EFFORT_DEFAULT, api.EFFORT_TEMPLATE = "medium", {"high": "medium"}
            api.VISION = bool(getattr(self.eng, "vision", False))   # TOB-37: images only with vision tower (TOBU_FNX_MMPROJ)
            if api.VISION:
                print("images: on (vision tower, limits " + json.dumps(__import__("media").limits()) + ")", flush=True)
            ti = chat.template_info(self.eng.template)
            if ti["kind"] in ("qwen35", "bonsai"):
                style = ti["kind"]
            if not self.template:                    # template from the GGUF (jinja2); only it knows reasoning_effort
                self.template = chat.TemplateText(self.eng.template)
            if ti.get("default_thinking") is not None:
                api.DEFAULT_THINKING = bool(ti["default_thinking"])
            if ti["tool_format"] != "qwen3-xml" and not getattr(a, "no_tool_grammar", False):
                a.no_tool_grammar = True
        elif a.engine.startswith("script:"):         # test engines without GPU (engines.py)
            from engines import ScriptEngine

            self.tok = HFTok.from_file(a.tokenizer)
            self.eng = ScriptEngine(self.tok, a.engine.split(":", 1)[1], eos=self.tok.t.token_to_id(chat.EOT))
            self.model = a.model or "script"
        elif a.engine.startswith("keyed:"):
            from engines import KeyedScriptEngine

            self.tok = HFTok.from_file(a.tokenizer)
            answers = json.loads(Path(a.engine.split(":", 1)[1]).read_text(encoding="utf-8"))
            self.eng = KeyedScriptEngine(self.tok, answers, eos=self.tok.t.token_to_id(chat.EOT), max_len=a.max_len)
            self.model = a.model or "keyed"
            style = "bonsai"
        elif a.engine == "toy":
            from engines import ToyEngine

            self.eng = ToyEngine(a.tokenizer)
            self.tok = HFTok(self.eng.hf_tok)
            self.model = a.model or "toy"
            style = "bonsai"
        else:
            raise SystemExit(f"--engine {a.engine}?")
        self.style = a.chat if a.chat != "auto" else style
        if getattr(a, "default_thinking", "auto") != "auto":    # L49: default without enable_thinking
            api.DEFAULT_THINKING = a.default_thinking == "on"
        api.DEFAULT_THINKING_BUDGET = max(0, int(getattr(a, "default_thinking_budget", 0) or 0))    # TOB-32
        api.THINKING_ANSWER_RESERVE = max(0, int(getattr(a, "thinking_answer_reserve", 1024)))       # TOB-33
        api.DEFAULT_BAN_SCRIPTS = parse_ban_scripts(getattr(a, "ban_scripts", "") or "")           # script ban
        self.kmax = min(self.kmax or 1 << 30, getattr(self.eng, "kmax", self.kmax))
        t = self.tok.t
        self.stop = {t.token_to_id(chat.EOT), t.token_to_id("<|endoftext|>")}
        self.prompt_history = PromptHistory()
        self.spec_sampling = not getattr(a, "no_spec_sampling", False)
        # harnesses whose requests without temperature run greedy instead of with the model card values (like the header
        # X-Tobu-Sampling: greedy, but as a server default per harness; X-Tobu-Sampling: model overrides it per request)
        self.greedy_harness = {h.strip() for h in (getattr(a, "greedy_harness", "") or "").split(",") if h.strip()}
        self.api_token = getattr(a, "token", "") or os.environ.get("TOBU_TOKEN") or ""
        # additionally allow Host headers (e.g. host.docker.internal for agents in a container); the server stays on 127.0.0.1
        self.allow_hosts = {"127.0.0.1", "localhost", "[::1]"} | {h.strip().lower() for h in getattr(a, "allow_host", None) or [] if h.strip()}
        if not self.api_token and getattr(a, "token_file", ""):     # generate and store the token
            self.api_token = secrets.token_urlsafe(24)
            tf = Path(a.token_file).expanduser()
            tf.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(tf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)     # never readable by others, not even briefly
            with os.fdopen(fd, "w") as fh:
                os.fchmod(fh.fileno(), 0o600)   # an existing file keeps its mode with O_CREAT: tighten it
                fh.write(self.api_token + "\n")
        self.tool_grammar = not getattr(a, "no_tool_grammar", False)
        self.tc_id = self.tok.t.token_to_id("<tool_call>")
        ts, te = self.tok.t.token_to_id("<think>"), self.tok.t.token_to_id("</think>")
        self.think_ids = (ts, {te, self.tc_id}) if ts is not None and te is not None else None
        self.tok_bytes = tool_format.token_bytes(self.tok) if self.tool_grammar else None
        self.log = Path(a.log) if a.log else None
        self.busy = False
        # GET /tobu/metrics + usage.tobu: running requests (rid -> state) and the latest metrics;
        # only assignments in the hot path (no locks), no contents
        self.live: dict[str, dict] = {}
        self.recent_metrics: collections.deque = collections.deque(maxlen=200)
        self.warm = threading.Event()            # main() sets it only after the warm-up (/health, /chat/ready/tobu)
        self.warm.set()
        self.warmup_s = None

    def warmup(self):
        """Before the first request run the whole path once (prefill, decode with drafts, with and without thinking),
        so that GPU programs, buffers and clock are ready. Afterwards clear the prefix memory so that nothing from the
        warm-up affects real requests (the output is independent of it anyway)."""
        t0 = time.perf_counter()
        text = "Continue this list of numbers in words, one per line: " + ", ".join(str(i) for i in range(24))
        for think in (False, True):
            req = api.parse_openai({"messages": [{"role": "user", "content": text}], "max_tokens": 24,
                                    "temperature": 0, "chat_template_kwargs": {"enable_thinking": think}})
            self.run(req, self.prompt(req), self.arm_default.upper(), lambda e: None, {})
            text = "Hi"
        self._clear_mem()
        self.warmup_s = round(time.perf_counter() - t0, 2)

    # --------------------------------------------------------------------------------------------- warm-up after pause
    # L39: after rewarm_after s of idle, send the last prompt (or its beginning) again through the real
    # prefill path (engine.prefill, the same grid/the same chunks as a request). The next turn then hits the cache with a warm GPU. Chunk by chunk per grid,
    # between the chunks abort as soon as a real request is waiting (it always has priority). Bit-identical: saves at
    # grid points are equal to the cold start (L38 point 10), the output of the warm-up is discarded.
    def note_request(self, ids: list[int]):
        with self.rw_lock:
            self.pending += 1
            self.last_req_t = time.time()
            self.rewarmed = False
            if self.rewarm_after > 0:
                self.recent.append(list(ids))

    def request_done(self):
        with self.rw_lock:
            self.pending -= 1
            self.last_req_t = time.time()

    def _rewarm_loop(self):
        while True:
            time.sleep(1.0)
            with self.rw_lock:
                due = (not self.rewarmed and self.pending == 0 and self.recent
                       and time.time() - self.last_req_t >= self.rewarm_after)
                if due:
                    self.rewarmed = True
            if due:
                WORKER.submit(self.rewarm)

    def rewarm(self):
        """Prefill the last prompt up to before its last token in grid steps; aborts as soon as a request is waiting."""
        ids = self.recent[-1]
        G = getattr(self.eng, "G", 512)
        stops = list(range(G, len(ids) - 1, G)) + [len(ids) - 1]
        t0, done = time.perf_counter(), 0
        for c in stops:
            if self.pending > 0:
                break
            self.eng.prefill(ids[: c + 1])          # first token discarded; state/saves as with a request
            done = c
        st = self.rewarm_stats
        st["runs"] = st.get("runs", 0) + 1
        st["aborted"] = st.get("aborted", 0) + (done < stops[-1])
        st["last_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        st["last_tokens"] = done + 1 if done else 0
        print(f"rewarmed after idle: {st['last_tokens']} tokens in {st['last_ms']} ms"
              + (" (aborted: request)" if done < stops[-1] else ""), flush=True)

    def _clear_mem(self):
        """After the warm-up only clear the memory cache (like a new session); do not touch the SSD cache."""
        if not hasattr(self.eng, "clear_cache"):
            return
        import inspect
        if "disk" in inspect.signature(self.eng.clear_cache).parameters:
            self.eng.clear_cache(disk=False)
        else:
            self.eng.clear_cache()

    # --------------------------------------------------------------------------------------------- prompt
    def prompt(self, req: api.Req) -> list[int]:
        """Template + tokenize (in the HTTP thread: error -> 400 before the engine does anything)."""
        ids = self.encode(req)
        max_len = getattr(self.eng, "max_len", 1 << 30)
        if len(ids) + 64 > max_len:     # wording like the Anthropic API: Claude Code then summarises the history
            raise api.ApiError(f"prompt is too long: {len(ids)} tokens > {max_len - 64} maximum")
        room = max_len - len(ids) - 16                  # KV buffer of the engine: prompt + answer + draft
        if req.max_tokens > room:
            req.max_tokens = room
            req.ignored.append(f"max_tokens>{room}")
        return ids

    def encode(self, req: api.Req) -> list[int]:
        """Template + tokenize without length check (count_tokens)."""
        kw = dict(req.kwargs)
        try:
            if self.template:            # other template via jinja2 (reference/comparison)
                if req.effort:
                    kw.setdefault("reasoning_effort", req.effort)
                text = chat.render_jinja(req.messages, req.tools, enable_thinking=req.thinking, template=self.template, **kw)
            elif self.style == "bonsai":
                if req.effort:
                    kw["reasoning_effort"] = req.effort
                kw = {k: v for k, v in kw.items() if k in ("reasoning_effort", "preserve_thinking")}
                text = chat.render_bonsai_py(req.messages, req.tools, enable_thinking=req.thinking, **kw)
            else:
                text = chat.render_py(req.messages, req.tools, enable_thinking=req.thinking)
        except (ValueError, KeyError, TypeError) as e:
            raise api.ApiError(f"chat template: {e}") from e
        req.gen_prompt = text[text.rfind("<|im_start|>assistant\n"):] if "<|im_start|>assistant\n" in text else ""
        if req.prefill:                  # TOB-12: Anthropic prefill after the generation prompt (without: prompt unchanged)
            text += req.prefill
        ids = self.tok.encode(text)
        imgs = api.images_of(req.messages)
        if imgs:
            ids = self._expand_images(ids, imgs, req)
        return ids

    def _expand_images(self, ids: list[int], imgs: list, req: api.Req) -> list[int]:
        """TOB-37: per image one <|image_pad|> from the template -> gh*gw image tokens; remember offsets for VPROMPT.
        More/fewer pads than images (e.g. the text contains the special token itself) -> 400."""
        import media

        try:
            media.check_request(imgs)
        except media.MediaError as e:
            raise api.ApiError(str(e)) from e
        pad = self.tok.t.token_to_id("<|image_pad|>")
        if pad is None or sum(1 for t in ids if t == pad) != len(imgs):
            raise api.ApiError("image placeholders do not match the images (is '<|image_pad|>' in the text?)")
        out, placed, k = [], [], 0
        for t in ids:
            if t == pad:
                placed.append((imgs[k], len(out)))
                out += [pad] * imgs[k].tokens
                k += 1
            else:
                out.append(t)
        req.images = placed
        return out

    # --------------------------------------------------------------------------------------------- generate
    def run(self, req: api.Req, ids: list[int], arm: str, sink, flag: dict) -> None:
        """In the worker thread. sink(("start", info) | ("think"|"text", s) | ("tool", call) | ("done", Done))."""
        self.busy = True
        rid = flag.get("rid") or f"req-{time.time_ns()}"
        if not flag.get("rid"):                  # without HTTP request (warm-up or similar): recognisable for metrics
            flag["harness"] = flag.get("harness") or "intern"
        flag["rid"] = rid
        live = self.live.get(rid) or self.live_entry(flag)
        live["busy_start"] = self._slots_busy()
        self.live[rid] = live
        live["running"] = True

        def tick(n, ph):                         # decode: committed tokens, phase think/text/tool (+ tool)
            if live["t_first"] is None:
                live["t_first"] = time.time()
                live["state"] = "decode"
            live["tokens"], live["phase"], live["tool"] = n, ph[0], ph[1]
        flag["_tick"] = tick

        def sink_live(ev):                       # state per request for /tobu/metrics (prefill -> decode, chunks)
            if ev[0] in ("think", "text", "tool", "tool_open", "tool_args") and live["t_first"] is None:
                live["t_first"] = time.time()
                live["state"] = "decode"
            sink(ev)
        try:
            self._run(req, ids, arm, sink_live, flag)
        finally:
            self.live.pop(rid, None)
            self.busy = False
            if hasattr(self.eng, "leave"):       # L49: release slot (also on error/abort)
                self.eng.leave()

    def live_entry(self, flag: dict) -> dict:
        """Running request for /tobu/metrics: queued (waiting for worker/slot) -> prefill (with progress) -> decode."""
        lv = {"request_id": flag.get("rid") or "", "client_ip": flag.get("client_ip") or "",
              "harness": flag.get("harness") or "", "state": "queued", "t_start": time.time(), "t_first": None,
              "tokens": 0, "busy_start": 0, "prefill_done": 0, "prefill_total": None, "prefill_eta_s": None}
        if flag.get("rid"):
            self.live[flag["rid"]] = lv
        return lv

    def _slots_busy(self) -> int:
        si = getattr(self.eng, "slot_info", None)
        if si is None:
            return 1 if self.busy else 0
        d = si()
        return int(d.get("slots", 1)) - int(d.get("free", 0))

    def metrics_snapshot(self, since: float | None = None, client_ip: str = "", harness: str = "", limit: int = 50,
                         probes: bool = False) -> dict:
        """GET /tobu/metrics: running (slots, requests, tok/s now) and the latest requests (from `since`, per
        `client_ip`/`harness`, at most `limit` ≤ 200). Probes (harness "probe") only with probes=1 or harness=probe."""
        now = time.time()
        act, tps, qpos = [], 0.0, 0
        for x in sorted(list(self.live.values()), key=lambda x: x["t_start"]):
            rate = (x["tokens"] / max(1e-3, now - x["t_first"])) if x["t_first"] else 0.0
            tps += rate
            r = {"request_id": x["request_id"], "client_ip": x["client_ip"], "harness": x["harness"],
                 "state": x["state"], "tokens": x["tokens"], "elapsed_s": round(now - x["t_start"], 2),
                 "tok_s": round(rate, 1), "prefill_done": x.get("prefill_done", 0),
                 "prefill_total": x.get("prefill_total"), "prefill_eta_s": x.get("prefill_eta_s"),
                 "phase": x.get("phase"), "tool": x.get("tool"), "queue_pos": None}
            if x["state"] == "queued":           # 1 = next in line
                qpos += 1
                r["queue_pos"] = qpos
            act.append(r)
        si = getattr(self.eng, "slot_info", None)
        total = int(si().get("slots", 1)) if si else 1
        reqs = [m for m in list(self.recent_metrics) if (since is None or m["t"] > since)
                and (not client_ip or m["client_ip"] == client_ip) and (not harness or m["harness"] == harness)
                and (probes or harness == PROBE or m["harness"] != PROBE)][-max(1, min(limit, 200)):]
        return {"t": round(now, 3), "model": self.model,
                "live": {"slots_busy": self._slots_busy(), "slots_total": total, "active_requests": len(act),
                         "tok_s_now": round(tps, 1), "requests": act},
                "requests": reqs}

    def context_info(self) -> dict:
        """Context line in the browser chat: how many tokens fit (limit max_len)."""
        max_len = getattr(self.eng, "max_len", 0) or 0
        return {"tokens_fit": max_len, "model_max_context": max_len, "memory_pressure": 1}

    def ready(self, name: str) -> bool:
        """Browser chat waits here before it starts the clock (start/warm-up do not count toward the answer time):
        until after the warm-up at start (False after 600 s)."""
        return self.warm.wait(600)

    def _run(self, req, ids, arm, sink, flag):
        diag = _Sampler(threading.get_ident()) if os.environ.get("TOBU_DIAG") else None
        if flag.get("nocache") and hasattr(self.eng, "clear_cache") and not getattr(self.eng, "store", None):
            self.eng.clear_cache()
        info = {}
        fmt = None
        if req.structured and not flag.get("nogram"):          # Structured Output: one grammar from the 1st token
            st = req.structured
            fmt = tool_format.ToolFormat.eager(tool_format.format_gbnf(st.get("schema"), st.get("grammar")),
                                               self.tok_bytes or tool_format.token_bytes(self.tok), self.stop,
                                               prefill=req.gen_prompt if "schema" in st else "")
        elif req.all_tools() and self.tool_grammar and not flag.get("nogram"):     # format grammar like the fork
            try:
                fmt = tool_format.ToolFormat(req.all_tools(), self.tok_bytes, {self.tc_id}, self.stop, req.parallel,
                                             req.force, think=self.think_ids, thinking=req.thinking)
            except tool_format.SchemaError as e:
                raise api.ApiError("JSON schema conversion failed (tools): " + str(e)[:300]) from e
        # sampling with drafts: the engine (flashnext) computes it itself (native_spec_sampling) – engine drafts at
        # full width (arm C/M), exactly in the distribution.
        # Only without seed: with a seed the same request should give the same output. At full width, however, the sequence of
        # draws depends on the copies, and the answer memory changes between two requests (measured:
        # same seed, second run different). With a seed at most one draft per round.
        native_spec = bool(req.sampling) and self.spec_sampling and not flag.get("nospecs") and \
            arm in ("C", "M") and getattr(self.eng, "native_spec_sampling", False) and \
            getattr(self.eng, "mtp", None) is not None and req.sampling.get("seed") is None
        if req.sampling:
            use_self = native_spec
        else:
            use_self = arm in ("C", "M") and getattr(self.eng, "mtp", None) is not None
        v = self.eng.make(ids)
        base_v = v
        if req.images:                           # TOB-37: images + offsets to the engine (IMAGE/VPROMPT in the prefill)
            if not hasattr(base_v, "vision"):
                raise api.ApiError("this engine has no image input")
            base_v.vision = req.images
        lv = self.live.get(flag.get("rid") or "")
        if lv is not None:                       # slot available: prefill starts (progress per chunk, if the engine reports it)
            lv.update(state="prefill", prefill_total=len(ids))
        if hasattr(base_v, "progress"):
            def _progress(done, total, eta, lv=lv, stream=req.stream):
                if lv is not None:
                    lv.update(prefill_done=done, prefill_total=total, prefill_eta_s=eta)
                if stream and done < total:      # SSE comment (standard clients ignore it)
                    sink(("progress", (done, total, eta)))
            base_v.progress = _progress
        if hasattr(base_v, "budget"):    # flashnext: remaining budget per engine round (wide copy check)
            base_v.budget = req.max_tokens
        if getattr(self.eng, "store", None) is not None and flag.get("nocache"):
            # L49/E3: cold start only for this request (the prefix memory of the others stays)
            v.use_cache = False
        if fmt is not None and not hasattr(v, "logits"):      # test engines without logits: without format grammar
            fmt = None
        if hasattr(v, "use_mtp"):        # engine with MTP head: without MTP draft also no head per round (arm A/B)
            v.use_mtp = use_self
        # flashnext: the engine enforces tool grammar, thinking budget and sampling itself (also in MTP rounds);
        # instead of the Python wrappers it gets the settings per request (Structured Output: still via the wrapper).
        # Structured Output with JSON schema also in the engine (grammar per line, MTP stays on); GBNF still via wrapper
        native = bool(getattr(self.eng, "native_constraints", False)) and \
            (not req.structured or "schema" in req.structured)
        if native and req.structured and hasattr(base_v, "probe_config") and \
                not base_v.probe_config(self._native_config(req, flag)):
            native = False                       # the engine does not compile the schema: this request via the Python wrapper
            info["structured"] = "python (engine rejects schema)"
            print("note: JSON schema rejected by the engine, Python wrapper: "
                  + json.dumps(req.structured.get("schema"), ensure_ascii=False)[:400], flush=True)
        if native_spec and not native:           # sampling via the Python wrapper (step mode): no engine drafts
            native_spec = use_self = False
            if hasattr(v, "use_mtp"):
                v.use_mtp = False
        if req.sampling and native and not native_spec and hasattr(base_v, "plain") and \
                (flag.get("nospecs") or not self.spec_sampling or arm not in ("C", "M")) and not flag.get("specone") and \
                getattr(self.eng, "native_spec_sampling", False):
            base_v.plain = True                  # sampling without drafts (switched off): rounds with exactly one token
        if req.penalties and not req.sampling and not native:
            req.ignored.append("presence_penalty/frequency_penalty/repetition_penalty (greedy, engine without sampler)")
        if not native and api.ban_scripts_for(req):     # script ban exists only in the engine (flashnext)
            req.ignored.append("ban_scripts")
        if native:
            base_v.config = self._native_config(req, flag)
            fmt = None
        if req.sampling and not native:
            if not hasattr(v, "logits"):
                raise api.ApiError(f"engine {self.model} cannot sample (only temperature=0)")
            v.sampling = True
            v = sampling.SampledVerifier(v, sampling.Sampler(req.sampling, ids), fmt, spec=False, stop=self.stop,
                                         q_mtp=False)
        elif fmt is not None:
            v = tool_format.FormatVerifier(v, fmt, mask=False)
        tb = None if native else self._think_budget(req, v, fmt)
        if tb is not None:               # TOB-32: outermost wrapper, forces the end of thinking when the budget is reached
            v = tb
        if v is not base_v and hasattr(base_v, "force_step"):
            base_v.force_step()          # flashnext: wrappers need logits per position -> step mode from the start
        rep = {"tok": []}
        # tool arguments in the stream (else the client sees nothing until the end for large write/edit): only streamed and with
        # tool grammar (there form and order of the parameters are guaranteed)
        det, sf = api.Detok(self.tok, self.stop), api.StopFilter(req.stop)
        sp = api.Splitter(req.all_tools(), req.thinking,
                          stream_args=bool(req.stream and self.tool_grammar and not flag.get("nogram")))
        if req.prefill:                  # TOB-12: continuation of the prefill, leading line breaks belong to it
            sp.started = True
        n_tool = [0]

        def put(evs):
            for e in evs:
                n_tool[0] += e[0] == "tool"
                sink(e)

        def on_tokens(new):
            if not flag.get("started"):
                flag["started"] = True
                sink(("start", {"prompt_n": len(ids), "cached": getattr(v, "reuse", 0)}))
            for t in new:                        # one stream packet per token (like the fork), not per round
                put(sp.feed(sf.feed(det.feed([t]))))
            rep["tok"] += new
            if flag.get("_tick") is not None:    # /tobu/metrics: tokens of this answer and phase (also in the middle of a call)
                flag["_tick"](len(rep["tok"]), sp.phase)
            if repeating(rep["tok"], len(new)):          # endless loop: abort like at max_tokens
                flag["repetition"] = True
            return sf.hit is not None or flag.get("cancel", False) or flag.get("repetition", False)

        res = generate(v, ids, n=req.max_tokens, stop=self.stop, use_self=use_self, on_tokens=on_tokens)
        put(sp.feed(sf.feed(det.flush()) + sf.flush()) + sp.finish())
        toks = res.tokens
        if sf.hit is not None:
            finish = "stop_sequence"
        elif n_tool[0]:
            finish = "tool_calls"
        elif toks and toks[-1] in self.stop:
            finish = "stop"
        else:
            finish = "length"
        if finish in ("stop", "tool_calls") and req.all_tools() and sp.malformed:
            raise api.ApiError(sp.malformed, 502, "tool_call_error")
        n = len(toks)
        src_draft, src_acc = Counter(), Counter()
        for r in res.rounds:
            if r.n_draft:
                src_draft[r.src] += r.n_draft
                src_acc[r.src] += r.n_acc
        cached = getattr(v, "reuse", 0)
        lcp_prev = self.prompt_history.lcp(ids) if flag.get("harness") != PROBE else (None, None)
        timings = {"arm": arm, "prompt_n": len(ids), "prompt_cached": cached,
                   "lcp_prev": lcp_prev[0], "lcp_prev_n": lcp_prev[1],
                   "cache_src": getattr(v, "cache_src", None) or ("miss" if not cached else "ram"),   # TOB-10: vram|ram|ssd|miss
                   **({"diag_threads": diag.stop()} if diag is not None else {}),
                   "ttft_ms": round(res.t_first * 1000, 1),
                   "prefill_ms": round(getattr(v, "t_prefill", res.t_first) * 1000, 1),
                   "predicted_n": n, "decode_ms": round(res.t_decode * 1000, 1),
                   "propose_ms": round(res.t_propose * 1000, 1),
                   "predicted_per_second": round((n - 1) / res.t_decode, 2) if n > 1 and res.t_decode > 0 else 0.0,
                   "rounds": len(res.rounds), "tokens_per_round": round(res.tok_per_round, 3),
                   "draft_by_src": dict(src_draft), "accepted_by_src": dict(src_acc),
                   "verify_ms": round(sum(r.t for r in res.rounds) * 1000, 1),
                   "propose_ms_per_round": round(res.t_propose * 1000 / max(1, len(res.rounds)), 4),
                   "rounds_by_src": dict(Counter(r.src or "-" for r in res.rounds)),
                   "draft_len_hist": dict(sorted(Counter(min(r.n_draft, 32) for r in res.rounds).items())),
                   **self._think_stats(req, toks, res),
                   **(tb.info() if tb is not None else {}),
                   **_phase_stats(res, toks, req.thinking, {"tc": self.tc_id, "te": (self.think_ids or (0, {0}))[0] and
                                                             self.tok.t.token_to_id("</think>"),
                                                             "tce": self.tok.t.token_to_id("</tool_call>")}),
                   "acc_len_hist": dict(sorted(Counter(min(r.n_acc, 32) for r in res.rounds).items())),
                   "catch_ms": round(getattr(v, "t_catch", 0) * 1000, 1),
                   "tool_grammar": fmt is not None, "grammar_fixed": fmt.n_fixed if fmt is not None else 0,
                   "mask_accepted": getattr(v, "n_mask_acc", 0),
                   "grammar_fixes": [[self.tok.decode([a]), self.tok.decode([b])] for a, b in fmt.fixes][:8]
                   if fmt is not None else [],
                   "repetition": bool(flag.get("repetition")),
                   **({"sample_forced": v.n_forced, "sample_drawn": v.n_drawn, "sample_masked": v.s.n_masked,
                         "q_drafted": v.n_q, "q_accepted": v.n_q_acc}
                      if req.sampling and hasattr(v, "n_forced") else {}),   # flashnext: sampling in the engine
                   # B: accepted draft tokens per source (flashnext STATS; mtp/history/corpus/store + rounds)
                   **({"drafts": _dd} if (_dd := (base_v.draft_diff() if hasattr(base_v, "draft_diff") else None))
                      else {}),
                   **{f"prep_{k}": x for k, x in info.items()}}
        tobu = {"engine": getattr(self.eng, "name", self.model), "arm": arm, "prompt_hash": _h(ids),
                "out_hash": _h(toks), "cached_tokens": cached}
        tobu["sampling"] = req.sampling or "greedy"
        tobu["spec"] = (f"Arm {arm} (sampling, exact, engine)" if native_spec else
                        "off (sampling)" if req.sampling else
                        "off" if arm == "A" else f"Arm {arm}")
        tobu["accelerated"] = (native_spec or not req.sampling) and arm != "A"
        tobu["finish"] = finish
        if flag.get("debug"):                    # evaluation (not public): output tokens
            tobu["out_tokens"] = toks
        if flag.get("repetition"):
            tobu["stop_reason"] = "repetition"
        if req.ignored:
            tobu["ignored"] = req.ignored
        if native and (base_v.config or {}).get("ban_scripts"):    # script ban active in the engine
            tobu["ban_scripts"] = base_v.config["ban_scripts"]
        if req.api == "anthropic":
            tobu["timings"] = timings
        if self.log:
            with open(self.log, "a") as f:
                f.write(json.dumps({"t": time.time(), "api": req.api, "finish": finish,
                                    "harness": flag.get("harness") or "", "sampling": req.sampling or None,
                                    "thinking": bool(req.thinking), "effort": req.effort,
                                    "think_budget": req.thinking_budget,
                                    "reasoning_n": self._reasoning_n(req, toks),
                                    "slots_busy": (self.eng.slot_info().get("slots", 1) - self.eng.slot_info().get("free", 0))
                                    if hasattr(self.eng, "slot_info") else None,
                                    "tool_calls": n_tool[0], "stream": req.stream, "prompt_h": tobu["prompt_hash"],
                                    "out_h": tobu["out_hash"], "spec": tobu["spec"], **timings}) + "\n")
        if hasattr(self.eng, "remember_output") and flag.get("harness") != PROBE and not flag.get("cancel"):
            try:                                 # answer memory (copy source for later requests)
                self.eng.remember_output([int(x) for x in toks])
            except Exception as e:               # noqa: BLE001 – drafts only
                print(f"answer memory: {type(e).__name__}: {e}", flush=True)
        rn = self._reasoning_n(req, toks)
        if tb is not None:               # log: budget reached yes/no
            tobu["thinking_budget"] = {"budget": tb.budget, "hit": tb.hit, "reasoning_tokens": rn}
            if tb.capped:                # TOB-33: budget capped because of the reserve for the answer
                tobu["thinking_budget"].update(capped=True, requested=tb.requested, reserve=tb.reserve)
                print(f"thinking budget capped: {tb.requested} -> {tb.budget} (max_tokens {req.max_tokens}, reserve "
                      f"{tb.reserve} for the answer)", flush=True)
            print(f"thinking budget {tb.budget}: reached {'yes' if tb.hit else 'no'} (thinking {rn} tokens, "
                  f"output {n}, finish {finish})", flush=True)
        try:
            tobu["usage_tobu"] = self._request_metrics(req, ids, n, cached, rn, timings, base_v, flag, finish)
        except Exception as e:  # noqa: BLE001 – metrics must never prevent an answer
            print(f"metrics not created: {type(e).__name__}: {e}", flush=True)
        sink(("done", api.Done(finish, sf.hit, len(ids), cached, n, timings, tobu, reasoning_n=rn)))

    def _request_metrics(self, req, ids, n, cached, rn, timings, v, flag, finish) -> dict:
        """Metrics per request (usage.tobu, /tobu/metrics) – only numbers/ids, no contents."""
        pre_s = (timings.get("prefill_ms") or 0) / 1000
        dr = timings.get("drafts")
        if dr is not None:                       # savings per copy source (estimate, see _saved_ms)
            dr = {**dr, "est_saved_ms": {k: _saved_ms(dr, k, timings) for k in ("history", "corpus", "store")}}
        st = getattr(v, "stats", None) or {}
        acc = (st.get("accepted", 0) / st["drafted"]) if st.get("drafted") else None
        busy_end = self._slots_busy()
        live = self.live.get(flag.get("rid"), {})
        m = {"request_id": flag.get("rid"), "t": round(time.time(), 3), "t_start": round(live.get("t_start", 0), 3),
             "client_ip": flag.get("client_ip") or "", "harness": flag.get("harness") or "", "api": req.api,
             "finish": finish, "prompt_tokens": len(ids), "completion_tokens": n,
             "ttft_ms": timings.get("ttft_ms"),
             "prefill_tps": round((len(ids) - cached) / pre_s, 1) if pre_s > 0 and len(ids) > cached else None,
             "decode_tps": timings.get("predicted_per_second"),
             "thinking_tokens": rn, "mtp_accept": round(acc, 3) if acc is not None else None,
             "concurrency": max(int(live.get("busy_start") or 1), busy_end, 1),
             "cache_hit_tokens": int(cached or 0), "sampling": bool(req.sampling),
             "drafts": dr}
        self.recent_metrics.append(m)
        return m

    def _native_config(self, req, flag) -> dict:
        """Settings per request for engines that enforce grammar/thinking budget/sampling themselves (flashnext).
        Same meaning as the Python wrappers: tool grammar only with format grammar on (--no-tool-grammar,
        X-Tobu-Tool-Grammar: off switch it off), tool_choice from req.force, thinking budget with TOB-33 reserve."""
        cfg: dict = {"thinking": bool(req.thinking), "parallel": bool(req.parallel), "think_budget": -1}
        if req.all_tools() and self.tool_grammar and not flag.get("nogram"):
            cfg["tools"] = req.all_tools()          # also deferred ones (tool search): no redirect to Bash
            cfg["tool_choice"] = ("auto" if req.force is None else "required" if not req.force
                                  else {"function": {"name": req.force[0]}})
        budget = req.thinking_budget if req.thinking_budget is not None else api.DEFAULT_THINKING_BUDGET
        gp = req.gen_prompt + req.prefill
        if budget and req.thinking and "<think>" in gp and "</think>" not in gp[gp.rfind("<think>"):]:
            eff, _capped, _reserve = think_budget.effective_budget(budget, req.max_tokens, api.THINKING_ANSWER_RESERVE)
            cfg["think_budget"] = int(eff)
        if req.penalties and not req.sampling:      # greedy with explicit penalty (presence_penalty …)
            for k, val in req.penalties.items():
                cfg["repeat_penalty" if k == "repetition_penalty" else k] = val
        if req.sampling:
            for k in sampling.KEYS:
                if req.sampling.get(k) is not None:
                    cfg["repeat_penalty" if k == "repetition_penalty" else k] = req.sampling[k]
        if req.structured and "schema" in req.structured and not flag.get("nogram"):
            from engine_flashnext import engine_schema           # Structured Output in the engine (JsonConstraint)
            cfg["json_schema"] = engine_schema(req.structured["schema"])
        bs = api.ban_scripts_for(req)          # script ban: only if effective (else CONFIG as before)
        if bs:
            cfg["ban_scripts"] = bs
        return cfg

    def _think_budget(self, req, v, fmt):
        """TOB-32: wrapper for the thinking budget (thinking_budget of the request, else --default-thinking-budget; 0 = off).
        Only with thinking on, if the prompt ends inside the thinking block and the engine can replace a token (resample)."""
        budget = req.thinking_budget if req.thinking_budget is not None else api.DEFAULT_THINKING_BUDGET
        if not budget or not req.thinking or self.think_ids is None:
            return None
        gp = req.gen_prompt + req.prefill
        if "<think>" not in gp or "</think>" in gp[gp.rfind("<think>"):]:
            return None                  # template opens no thinking block or it is already closed
        if not callable(getattr(v, "resample", None)):
            req.ignored.append("thinking_budget")       # test engine without replacing: no effect
            return None
        te = self.tok.t.token_to_id("</think>")
        close = self.tok.encode("\n</think>\n\n")      # like the template: reasoning + "\n</think>\n\n" + content
        close_nl = self.tok.encode("</think>\n\n")
        if te not in close or te not in close_nl:
            return None
        # TOB-33: reserve for the answer: effective min(budget, max_tokens - reserve), 0 = end thinking immediately
        eff, capped, reserve = think_budget.effective_budget(budget, req.max_tokens, api.THINKING_ANSWER_RESERVE)
        tb = think_budget.ThinkBudget(v, eff, self.think_ids[1], close, close_nl,
                                      lambda t: self.tok.decode([t]).endswith("\n"), fmt=fmt,
                                      sampler=getattr(v, "s", None) if req.sampling else None)
        tb.requested, tb.capped, tb.reserve = budget, capped, reserve
        return tb

    def _reasoning_n(self, req, toks) -> int:
        """TOB-32: tokens in the thinking block (up to before </think> or <tool_call>) for usage.reasoning_tokens."""
        if not req.thinking or self.think_ids is None:
            return 0
        ends = self.think_ids[1]
        return next((i for i, t in enumerate(toks) if t in ends), len(toks))

    def _think_stats(self, req, toks, res) -> dict:
        """Thinking on: tokens and rounds in the thinking block or after it, drafts per source in the thinking."""
        if not req.thinking or self.think_ids is None:
            return {}
        ends = self.think_ids[1]
        e = next((i for i, t in enumerate(toks) if t in ends), len(toks))     # end of thinking in the output
        pos, tr, acc = 1, 0, Counter()
        for r in res.rounds:                     # round starts at output position pos (first token from the prefill)
            if pos < e:
                tr += 1
                if r.n_acc:
                    acc[r.src or "-"] += r.n_acc
            pos += r.n_acc + 1
        return {"think_n": e, "think_rounds": tr, "answer_n": len(toks) - e, "answer_rounds": len(res.rounds) - tr,
                "think_accepted_by_src": dict(acc)}


class _Sampler:
    """Diagnosis (TOBU_DIAG=1): which other Python threads execute code during the first 4 s of a request
    (every 5 ms the top stack line; wait/sleep calls do not count). Explains e.g. slower prefill."""
    IDLE = ("wait", "sleep", "select", "accept", "recv", "recv_into", "readinto", "readline", "_worker", "get",
            "serve_forever", "_read", "poll")

    def __init__(self, main_id: int):
        self.main, self.counts, self.n, self.on = main_id, Counter(), 0, True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        t_end = time.perf_counter() + 4.0
        me = threading.get_ident()
        while self.on and time.perf_counter() < t_end:
            for tid, fr in sys._current_frames().items():
                if tid in (self.main, me) or fr.f_code.co_name in self.IDLE:
                    continue
                self.counts[f"{Path(fr.f_code.co_filename).name}:{fr.f_code.co_name}"] += 1
            self.n += 1
            time.sleep(0.005)

    def stop(self) -> dict:
        self.on = False
        return {"samples": self.n, "top": self.counts.most_common(8)}


def _chat_session(token: str) -> str:
    """Value of the chat session cookie: derived from the API token, so the cookie never carries the token itself."""
    return hmac.new(token.encode(), b"tobu-chat-session", hashlib.sha256).hexdigest()


def _rid() -> str:
    """request-id like Anthropic (req_…), random per request."""
    return "req_" + secrets.token_hex(12)


def _h(ids: list[int]) -> str:
    """Short hash of a token sequence (check bit-identity without logging contents)."""
    return hashlib.sha1(json.dumps(ids).encode()).hexdigest()[:16]



def make_handler(B: Backend):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *x):
            pass

        def end_headers(self):
            """TOB-12: every answer with request-id (like the Anthropic API), one per request."""
            self.send_header("request-id", getattr(self, "rid", None) or _rid())
            super().end_headers()

        def _send(self, code: int, obj, ctype="application/json", headers: dict | None = None):
            if isinstance(obj, dict) and obj.get("type") == "error" and getattr(self, "rid", None):
                obj = {**obj, "request_id": self.rid}       # Anthropic form: request_id also in the error JSON
            body = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _starting(self, anth: bool):
            """TOB-30: before loading/warm-up is finished, 503 with Retry-After instead of putting the request into the worker.
            The start (warm-up, clear_cache before the slots) runs over the same single worker thread; a request
            waiting there on B.warm held it -> deadlock, server dead until restart."""
            e = api.ApiError("tobu is still starting, try again", 503, "overloaded_error" if anth else "unavailable")
            body = api.AnthropicOut.error_body(e) if anth else api.OpenAIOut.error_body(e)
            return self._send(503, body, headers={"Retry-After": str(STARTING_RETRY_S)})

        def _gate(self, p: str) -> bool:
            """Secure localhost: Host header only 127.0.0.1/localhost (DNS rebinding from the browser), with --token
            only with session token (Authorization: Bearer or x-api-key) or the chat session cookie (same origin only);
            the chat page itself needs the cookie or ?token= once (-> redirect that sets the cookie).
            Errors in the format of the interface."""
            anth = p.startswith("/v1/messages")
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0] if not (self.headers.get("Host") or "").startswith("[") \
                else (self.headers.get("Host") or "").split("]")[0] + "]"
            if host.lower() not in B.allow_hosts:
                err = api.ApiError("forbidden host", 403, "permission_error")
            elif B.api_token and p != "/health":
                if p in ("", "/chat") and self.command == "GET":       # ?token= always leaves the address bar
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    given = (q.get("token") or [""])[0]
                    if given and hmac.compare_digest(given.encode(), B.api_token.encode()):
                        self._chat_login()
                        return False
                if hmac.compare_digest(self._given_token().encode(), B.api_token.encode()) or self._cookie_ok():
                    return True
                err = api.ApiError("invalid or missing token" + (" (open /chat?token=<token> once)"
                                                                 if p in ("", "/chat") else ""),
                                   401, "authentication_error")
            else:
                return True
            self._send(err.status, (api.AnthropicOut if anth else api.OpenAIOut).error_body(err))
            return False

        def _given_token(self) -> str:
            """Session token from Authorization: Bearer or x-api-key."""
            auth = self.headers.get("Authorization") or ""
            return auth[7:].strip() if auth.lower().startswith("bearer ") else (self.headers.get("x-api-key") or "")

        def _cookie_ok(self) -> bool:
            """Chat session cookie (HMAC of the token, never the token itself). Only for same-origin requests:
            SameSite=Strict keeps foreign sites out, Origin/Sec-Fetch-Site also other ports on the same host."""
            jar = http.cookies.SimpleCookie()
            try:
                jar.load(self.headers.get("Cookie") or "")
            except http.cookies.CookieError:
                return False
            c = jar.get(CHAT_COOKIE)
            if c is None or not hmac.compare_digest(c.value.encode(), _chat_session(B.api_token).encode()):
                return False
            origin = self.headers.get("Origin")
            if origin is not None and origin.lower() not in ("http://" + (self.headers.get("Host") or "").lower(),
                                                             "https://" + (self.headers.get("Host") or "").lower()):
                return False
            return (self.headers.get("Sec-Fetch-Site") or "same-origin").lower() in ("same-origin", "none")

        def _chat_login(self):
            """?token= was correct: set the session cookie and redirect to /chat (token out of the address bar)."""
            self.send_response(303)
            for k, v in (("Location", "/chat"), ("Content-Length", "0"), ("Cache-Control", "no-store"),
                         ("Referrer-Policy", "no-referrer"),
                         ("Set-Cookie", f"{CHAT_COOKIE}={_chat_session(B.api_token)}; HttpOnly; SameSite=Strict; Path=/")):
                self.send_header(k, v)
            self.end_headers()

        def _an_404(self, p: str):
            """TOB-12: unknown path under /v1/messages in Anthropic form; batches do not exist."""
            msg = ("Message Batches API is not supported by tobu" if p.startswith("/v1/messages/batches")
                   else f"unknown path {p}")
            return self._send(404, api.AnthropicOut.error_body(api.ApiError(msg, 404, "not_found_error")))

        def do_GET(self):
            self.rid = _rid()
            p = self.path.split("?")[0].rstrip("/")
            if not self._gate(p):
                return
            if p in ("", "/chat"):
                return self._chat_page()
            if p == "/chat/info":                 # context line (not under /tobu/, that is publicly blocked)
                return self._send(200, B.context_info())
            if p == "/chat/ready" or p.startswith("/chat/ready/"):     # page: /chat/ready/tobu
                name = {"": "tobu", "tobu": "tobu"}.get(
                    p[len("/chat/ready/"):] if p.startswith("/chat/ready/") else "")
                if name is None:
                    return self._send(404, {"error": {"message": f"unknown path {p}", "type": "not_found_error"}})
                if not B.ready(name):
                    return self._send(503, {"error": {"message": "tobu is still starting, try again",
                                                      "type": "unavailable"}})
                return self._send(200, {"ready": True})
            if p in ("/v1/models", "/models"):
                if not B.warm.is_set():           # TOB-30: 200 only after the warm-up (clients do not wait otherwise)
                    return self._starting(bool(self.headers.get("anthropic-version")))
                if self.headers.get("anthropic-version"):
                    m = {"type": "model", "id": B.model, "display_name": B.model, "created_at": "2026-10-19T00:00:00Z"}
                    return self._send(200, {"data": [m], "has_more": False, "first_id": B.model, "last_id": B.model})
                return self._send(200, {"object": "list", "data": [{"id": B.model, "object": "model", "owned_by": "tobu"}]})
            if p == "/tobu/metrics":              # metrics (no contents), same access as the API
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                try:
                    since = float(q["since"][0]) if "since" in q else None
                except ValueError:
                    since = None
                try:
                    limit = int(q["limit"][0]) if "limit" in q else 50
                except ValueError:
                    limit = 50
                one = lambda k: (q.get(k) or [""])[0].strip()[:64]          # noqa: E731
                return self._send(200, B.metrics_snapshot(since, one("client_ip"), one("harness"), limit,
                                                          one("probes") in ("1", "true")))
            if p == "/props":                     # like llama-server: context window + template for clients
                if not B.warm.is_set():
                    return self._starting(False)
                n_ctx = int(getattr(B.eng, "max_len", 0) or 0)
                tpl = B.template or getattr(B.eng, "template", None) or ""
                return self._send(200, {"model_alias": B.model, "n_ctx": n_ctx,
                                        "default_generation_settings": {"n_ctx": n_ctx},
                                        "total_slots": int(getattr(B, "slots", 0) or 1),
                                        "chat_template": str(tpl), "modalities": {"vision": False, "audio": False}})
            if p == "/health":                    # like llama-server: 503 while loading or warming up
                if not B.warm.is_set():
                    return self._send(503, {"status": "loading", "model": B.model})
                if not getattr(B.eng, "alive", lambda: True)():      # engine process gone (OOM or similar)
                    return self._send(503, {"status": "engine_dead", "model": B.model})
                return self._send(200, {"status": "ok", "model": B.model, "busy": B.busy,
                                        **({"mode": B.mode} if getattr(B, "mode", None) else {}),
                                        **({"mtp_k": B.mtp_k} if getattr(B, "mtp_k", None) else {}),
                                        **({"rewarm": B.rewarm_stats} if B.rewarm_stats else {}),
                                        **({"slots": B.eng.slot_info()} if getattr(B, "slots", 0) else {})})
            if p.startswith("/v1/messages"):
                return self._an_404(p)
            self._send(404, {"error": {"message": f"unknown path {p}", "type": "not_found_error"}})

        def _chat_page(self):
            """Browser chat: chat.html (only after host check and, with --token, session cookie; the token is never in
            the page, the page's requests authenticate with the cookie), do not cache, not embeddable in foreign
            pages, no foreign sources (CSP)."""
            t = CHAT_HTML if CHAT_HTML is not None else (HERE / "chat.html").read_text(encoding="utf-8")
            body = t.encode()
            self.send_response(200)
            for k, v in (("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body))),
                         ("Cache-Control", "no-store"), ("X-Frame-Options", "DENY"), ("Referrer-Policy", "no-referrer"),
                         ("X-Content-Type-Options", "nosniff"), ("Content-Security-Policy", CHAT_CSP)):
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rid = _rid()
            p = self.path.split("?")[0].rstrip("/")
            if not self._gate(p):
                return
            anth = p.startswith("/v1/messages")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if anth and not isinstance(body, dict):
                    raise json.JSONDecodeError("request body must be a JSON object", "", 0)
            except (ValueError, RecursionError) as e:   # TOB-54: also UTF-8 errors and too deep nesting -> 400
                if anth:                 # TOB-12: Anthropic form
                    return self._send(400, api.AnthropicOut.error_body(api.ApiError(f"invalid JSON: {e}")))
                return self._send(400, {"error": {"message": f"invalid JSON: {e}", "type": "invalid_request_error"}})
            xs = (self.headers.get("X-Tobu-Sampling") or "").strip().lower()
            if isinstance(body, dict) and body.get("temperature") is None and (
                    xs == "greedy" or (not xs and _harness(self.headers) in getattr(B, "greedy_harness", ()))):
                body["temperature"] = 0      # L35: Claude Code sends no temperature; --greedy-harness per harness
            if p.endswith("/chat/completions"):
                parse, Out = api.parse_openai, api.OpenAIOut
            elif p == "/v1/messages/count_tokens":
                try:
                    return self._send(200, {"input_tokens": len(B.encode(api.parse_anthropic(body, count=True)))})
                except api.ApiError as e:
                    return self._send(e.status, api.AnthropicOut.error_body(e))
            elif p == "/v1/messages":
                parse, Out = api.parse_anthropic, api.AnthropicOut
            elif anth:
                return self._an_404(p)
            else:
                return self._send(404, {"error": {"message": f"unknown path {p}", "type": "not_found_error"}})
            if not B.warm.is_set():              # TOB-30: never wait for the warm-up in the worker (deadlock at start)
                return self._starting(anth)
            try:
                req = parse(body)
                ids = B.prompt(req)
            except api.ApiError as e:
                return self._send(e.status, Out.error_body(e))
            except (AttributeError, TypeError, KeyError, ValueError, IndexError) as e:     # L49: broken request -> 400
                return self._send(400, Out.error_body(api.ApiError(f"invalid request: {type(e).__name__}: {e}"[:300])))
            arm = (self.headers.get("X-Tobu-Arm") or B.arm_default).upper()
            if arm not in ARMS:
                return self._send(400, Out.error_body(api.ApiError(f"X-Tobu-Arm {arm!r}: {'/'.join(ARMS)}")))
            out, q = Out(B.model, req), queue.Queue()
            out.request_id = self.rid
            flag = {"nocache": (self.headers.get("X-Tobu-Cache") or "").lower() == "off"
                               or (isinstance(body, dict) and body.get("cache_prompt") is False),   # like llama-server
                    "nogram": (self.headers.get("X-Tobu-Tool-Grammar") or "").lower() == "off",
                    "debug": bool(self.headers.get("X-Tobu-Debug")) and not os.environ.get("TOBU_PUBLIC"),
                    "harness": PROBE if _is_probe(body) else _harness(self.headers),
                    "rid": self.rid,
                    "client_ip": (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip(),
                    # measurement: off = sampling entirely without drafts; one = at most one draft per round
                    # (comparison value for the full width)
                    "nospecs": (self.headers.get("X-Tobu-Spec-Sampling") or "").strip().lower() in ("off", "one"),
                    "specone": (self.headers.get("X-Tobu-Spec-Sampling") or "").strip().lower() == "one"}

            # L39 --rewarm-after: idle clock, last prompts, priority over warm-up. Not for image prompts (image tokens
            # without image = wrong state) and not for probes (monitoring every second would never leave the server idle).
            noted = not req.images and flag["harness"] != PROBE
            if noted:
                B.note_request(ids)

            def job():
                try:
                    B.run(req, ids, arm, q.put, flag)
                except api.ApiError as e:
                    q.put(("error", e))
                except Exception as e:   # engine error: server keeps running
                    import traceback
                    print(f"engine error: {type(e).__name__}: {e}\n{traceback.format_exc()[-2500:]}", flush=True)
                    q.put(("error", api.ApiError(f"{type(e).__name__}: {e}", 500, "api_error")))
                finally:
                    if noted:                    # only counted requests (else pending goes negative)
                        B.request_done()

            B.live_entry(flag)                   # /tobu/metrics: visible already in the queue
            JOBS[0].submit(job)
            if req.stream:
                self._stream(out, q, flag)
            else:
                self._whole(out, q)

        def _whole(self, out, q):
            while True:
                kind, x = q.get()
                if kind == "error":
                    return self._send(x.status, out.error_body(x))
                if kind == "done":
                    return self._send(200, out.body(x))
                if kind not in ("start", "progress"):
                    out.event(kind, x)

        def _stream(self, out, q, flag):
            try:                                   # every packet immediately (else TCP bundles small chunks)
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            started = False
            while True:
                try:
                    kind, x = q.get(timeout=KEEPALIVE_S)
                except queue.Empty:
                    kind, x = "keepalive", None
                if kind == "keepalive":
                    data = out.keepalive if started else b": keepalive\n\n"
                elif kind == "progress":           # prefill progress as SSE comment
                    d, tot, eta = x
                    data = f": tobu prefill {d}/{tot}{f' eta {eta:.0f}s' if eta is not None else ''}\n\n".encode()
                elif kind == "start":
                    started = True
                    data = out.start(x)
                elif kind == "error":
                    data = out.error(x)
                elif kind == "done":
                    data = out.end(x)
                else:
                    data = out.event(kind, x)
                try:
                    self.wfile.write(data)
                    self.wfile.flush()
                except OSError:
                    flag["cancel"] = True          # client gone: generation ends after the current round
                    while kind not in ("done", "error"):
                        kind, x = q.get()
                    return
                if kind in ("done", "error"):
                    return

    return H


def build_parser() -> argparse.ArgumentParser:
    """All switches of the server (factored out of main so that tests without GPU can parse and check them)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="flashnext", help="flashnext (Qwen3.8-Flash-Next, tobu-fnx on gfx1151) | "
                    "test engines without GPU: toy | keyed:<json> | script:<json> (with --tokenizer)")
    ap.add_argument("--gguf", default="", help="--engine flashnext: first part of the Flash-Next GGUF")
    ap.add_argument("--fnx-exe", default="", help="--engine flashnext: command for tobu-fnx (default: TOBU_FNX or "
                    "scripts/run-fnx.sh engine/build/tobu-fnx)")
    ap.add_argument("--fnx-mtp", default="", help="--engine flashnext: MTP GGUF (draft head), empty = without MTP")
    ap.add_argument("--tokenizer", default="", help="tokenizer.json for the test engines (flashnext reads it from the GGUF)")
    ap.add_argument("--chat", default="auto", help="template of the test engines: auto (by engine) | bonsai | qwen35")
    ap.add_argument("--template", default="", help="chat template (jinja2 file) instead of the template from the GGUF")
    ap.add_argument("--arm", default="C", help="C (default) or M: with engine drafts (MTP, copies); A/B: without")
    ap.add_argument("--kmax", type=int, default=0, help="longest draft; 0 = default (8)")
    ap.add_argument("--allow-host", action="append", default=[], help="allow another Host header (repeatable), "
                    "e.g. host.docker.internal for agents in Docker; default only 127.0.0.1/localhost")
    ap.add_argument("--token", default="", help="session token for the API (Bearer or x-api-key; browser chat: open "
                    "/chat?token=<token> once, then an HttpOnly cookie); also via TOBU_TOKEN; without token open (only 127.0.0.1)")
    ap.add_argument("--token-file", default="", help="without --token: generate a token and write it to this file (0600)")
    ap.add_argument("--no-tool-grammar", action="store_true", help="tool calls without format grammar (plain greedy); "
                    "default: enforce the skeleton of the tool format")
    ap.add_argument("--greedy-harness", default="", help="harnesses (comma-separated, X-Tobu-Harness or User-Agent) "
                    "whose requests without temperature run greedy instead of with the model card values (mode G); per "
                    "request X-Tobu-Sampling: model overrides it")
    ap.add_argument("--no-spec-sampling", action="store_true", help="sampling (temperature > 0) without drafts "
                    "(measurement); default: drafts at full width, exact in the distribution (mode A)")
    ap.add_argument("--max-len", type=int, default=65536)
    ap.add_argument("--slots", type=int, default=0, help="parallel requests (one session in tobu-fnx per slot), "
                    "0 = 1")
    ap.add_argument("--default-thinking", choices=("auto", "on", "off"), default="auto",
                    help="thinking when the request does not say (auto = as the template)")
    ap.add_argument("--default-thinking-budget", type=int, default=0,
                    help="TOB-32: thinking budget in tokens when the request names no thinking_budget (e.g. 4096; 0 = off): "
                         "if thinking reaches the budget without </think>, the engine ends the thinking and the "
                         "model answers afterwards")
    ap.add_argument("--thinking-answer-reserve", type=int, default=1024,
                    help="TOB-33: tokens kept free for the answer after the thinking budget (at least 25 %% of "
                         "max_tokens): effective budget = min(thinking_budget, max_tokens - reserve); if max_tokens "
                         "is smaller than the reserve, thinking ends immediately")
    ap.add_argument("--ban-scripts", default="", help="script ban as server default (comma-separated, known: "
                    "cjk; empty = off): CJK tokens hard-banned when the request names no ban_scripts and its "
                    "messages (except assistant) contain no CJK themselves; ban_scripts: [] turns it off per request")
    ap.add_argument("--tool-choice-lax", action="store_true", help="tool_choice required lax (text before the "
                    "call allowed, only no end); default: strict (only whitespace before the call)")
    ap.add_argument("--rewarm-after", type=float, default=0, help="after this many seconds idle, prefill the last prompt "
                    "again (same prefill path, bit-identical; real requests take priority; only without "
                    "--slots), 0 = off")
    ap.add_argument("--rewarm-n", type=int, default=4, help="--rewarm-after: remember this many last prompts")
    ap.add_argument("--no-warmup", action="store_true", help="no warm-up after loading (only for measurements)")
    ap.add_argument("--model", default="", help="model name in the answers (default qwen38-flash-next)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--log", default="", help="JSONL log per request (counters and hashes, no contents)")
    return ap


def _saved_ms(dr: dict | None, src: str, timings: dict) -> float:
    """Estimated decode savings of a copy source (B): the tokens of its rounds (accepted + one per round) the
    MTP chain would have had to produce at its rate for this answer (tokens per MTP round); saved = these rounds minus
    the ones actually used, times ms per round. Estimate (copy rounds are slightly wider than MTP rounds)."""
    if not dr or not dr.get("cycles"):
        return 0.0
    cyc_src = dr.get("history_cycles" if src == "history" else f"{src}_cycles", 0)
    acc_src = dr.get(src, 0)
    mtp_cyc = dr["cycles"] - dr.get("copy_cycles", 0)
    r = (dr.get("mtp", 0) + mtp_cyc) / mtp_cyc if mtp_cyc > 0 else 1.0
    if r <= 0 or acc_src < 0 or cyc_src < 0:
        return 0.0
    ms_round = (timings.get("decode_ms") or 0) / dr["cycles"]
    return round(max(0.0, (acc_src + cyc_src) / r - cyc_src) * ms_round, 1)


class PromptHistory:
    """Diagnosis for the prefix memory (K12, 22.11.2026): the last prompts as token ids, only in RAM. For every new
    request: longest common prefix with one of the earlier prompts (lcp_prev) and its length (lcp_prev_n).
    Together with prompt_cached the log shows why it was recomputed: lcp_prev clearly above prompt_cached =
    the prefix was there before, but no snapshot anymore (evicted, no grid point); lcp_prev ≈ prompt_cached =
    the prompt really differs there (harness). Only numbers in the log, no contents."""

    def __init__(self, keep: int = 24):
        self.items: collections.deque = collections.deque(maxlen=keep)
        self.lock = threading.Lock()

    def lcp(self, ids) -> tuple[int, int]:
        import numpy as np
        a = np.asarray(ids, dtype=np.int32)
        with self.lock:
            prev = list(self.items)
        best = (0, 0)
        for b in prev:
            n = min(len(a), len(b))
            if n <= best[0]:
                continue
            ne = np.flatnonzero(a[:n] != b[:n])
            k = int(ne[0]) if len(ne) else n
            if k > best[0]:
                best = (k, len(b))
        with self.lock:
            self.items.append(a)
        return best


PROBE = "probe"
# Monitoring probes (e.g. a monitor that sends a tiny streamed request every second). They run
# normally, but do not count as work: not in the /tobu/metrics history (except ?probes=1), not in the
# answer memory, not in the idle clock. Detected by the header X-Tobu-Harness: probe or by exactly
# this request (one user message "Reply with exactly: ok", max_tokens ≤ 16, without tools).
PROBE_PROMPTS = {"Reply with exactly: ok"}


def _is_probe(body) -> bool:
    if not isinstance(body, dict) or body.get("tools"):
        return False
    m = body.get("messages")
    mt = body.get("max_tokens") or body.get("max_completion_tokens") or 0
    return (isinstance(m, list) and len(m) == 1 and isinstance(m[0], dict) and m[0].get("role") == "user"
            and isinstance(m[0].get("content"), str) and m[0]["content"] in PROBE_PROMPTS   # a list (images) is not a probe
            and isinstance(mt, int) and 0 < mt <= 16)


def _harness(headers) -> str:
    """Harness of the request for the log: X-Tobu-Harness, else the first product in the User-Agent (e.g. opencode)."""
    h = (headers.get("X-Tobu-Harness") or "").strip()
    if not h:
        ua = (headers.get("User-Agent") or "").strip()
        h = "opencode" if "opencode" in ua.lower() else ua.split("/")[0].split(" ")[0]
    return h[:40]


def main():
    a = build_parser().parse_args()
    if a.engine == "flashnext" and not a.gguf:
        sys.exit("tobu: --engine flashnext needs --gguf <model-00001-of-0000N.gguf>")
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))      # atexit: prefix memory to the SSD
    B = WORKER.submit(Backend, a).result()
    print(f"tobu server {B.model} ({B.style}) on {a.host}:{a.port}, default arm {a.arm}, kmax {B.kmax}", flush=True)
    httpd = AgentHTTPServer((a.host, a.port), make_handler(B))
    B.warm.clear()                               # /health 503 and /chat/ready wait until after the warm-up
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    if hasattr(B.eng, "persist_hot"):            # stop (SIGTERM -> sys.exit): hot prefix entries to the SSD
        import atexit

        def _persist():
            try:
                t0 = time.perf_counter()
                n = B.eng.persist_hot(float(os.environ.get("TOBU_FNX_DISK_STOP_S", "20")))
                # also report 0 (everything already on the SSD) – else a stop without new entries looks like a failure
                print(f"prefix memory: {n} entries written to the SSD on stop ({time.perf_counter() - t0:.1f} s)"
                      + (" – everything already on the SSD" if not n else ""), flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"prefix memory: saving on stop failed ({e})", flush=True)
        atexit.register(_persist)
    try:
        if not a.no_warmup:
            WORKER.submit(B.warmup).result()
            print(f"warmed up in {B.warmup_s} s", flush=True)
    except Exception as e:                       # noqa: BLE001  (warm-up is optional, the server keeps running)
        print(f"warm-up failed: {type(e).__name__}: {e}", flush=True)
    if a.tool_choice_lax:
        tool_format.STRICT_REQUIRED = False
    if a.slots == 0:
        a.slots = 1
    if a.slots > 1 and hasattr(B.eng, "enable_slots"):     # after the warm-up (that runs without slots)
        WORKER.submit(B.eng.clear_cache).result()
        B.eng.enable_slots(a.slots)
        JOBS[0] = ThreadPoolExecutor(max_workers=a.slots)
        B.slots = a.slots                                   # /props total_slots
        B.rewarm_after = 0
        print(f"slots: {a.slots}", flush=True)
    elif a.slots > 1:
        print(f"slots: engine {a.engine} does not support slots, 1 slot", flush=True)
    B.warm.set()
    if hasattr(B.eng, "alive"):                  # watchdog: engine dead (OOM kill or similar) -> server exits (exit 70),
        def _watch_engine():                     # so that start script/service see it, instead of silently with /health 200
            while True:                          # continuing to run and letting every request fail
                time.sleep(2.0)
                if not B.eng.alive():
                    print("ERROR: engine process exited (OOM?) – server exits (exit 70)", flush=True)
                    os._exit(70)
        threading.Thread(target=_watch_engine, daemon=True).start()
    if B.rewarm_after > 0 and hasattr(B.eng, "prefill"):
        threading.Thread(target=B._rewarm_loop, daemon=True).start()
    try:
        threading.Event().wait()
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()
