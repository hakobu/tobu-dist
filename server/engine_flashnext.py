"""tobu/qwen38-flash-next-gfx1151: compute core tobu-fnx (engine/) as the engine of the agent server (`--engine flashnext`).
Interface: make(ids) -> Verifier with start, verify(draft), self_draft, logits/resample (decoder.py).

tobu-fnx computes like Gufo: a decode round picks the next token itself, drafts with the MTP head (and copies from
history/answer memory), verifies and commits the accepted tokens immediately (state and logits are then
behind the last token). Mapping onto the decoder (decoder.py) in two modes per request:

* Stream (default; the engine itself enforces grammar, thinking budget and sampling, CONFIG): committed tokens
  go into a queue. self_draft() fills it with one round and offers everything except the last token as a
  draft; verify(d) compares d with the queue (n = common prefix) and returns its next token.
  The output is thus always the engine's sequence, independent of the draft (like OracleVerifier).
* Step (as soon as a Python wrapper needs logits(): GBNF grammar, rejected JSON schema): LOGITS -> the server
  picks the token (argmax or mask/sample/forced token via resample) -> EVAL(token). Only allowed from the
  initial state (empty queue); switching in the middle of the stream is an error.

Prefix memory at grid points (SAVE/LOAD/SYNCCP, RAM + optional SSD), slots (parallel sessions, batched
decode) and images (vision tower): see PrefixStore, DiskTier, _StepBatcher.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import struct
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

INFO, RESET, SYNC, STEP, EVAL, LOGITS, SAVE, LOAD, DROP, SYNCCP, CONFIG, PEEK, CORPUS_SET, CORPUS_DROP, SELECT, STEPB, \
    USAGE = range(17)
PERSIST, LOADFILE, PERSIST_WAIT = 22, 23, 24
IMAGE, VPROMPT, IMAGE_DROP = 19, 20, 21
STATS = 25
STATS_KEYS = ("cycles", "drafted", "accepted", "lookup", "lookup_accepted", "lookup_cycles", "corpus", "corpus_accepted",
              "corpus_cycles", "store", "store_accepted", "store_cycles")
XSTORE_BASE = 1 << 24                       # corpus ids of the answer memory (files count from 1)
IMG_SNAP_VERSION = 2                        # computation path of the image checkpoints (PrefixStore.key)
GRID = int(os.environ.get("TOBU_FNX_GRID", "4"))
# Fair prefills (a cold 105k prompt otherwise held the pipe for ~2 min, all slots stalled): prefill in
# chunks at grid points; between the chunks the pipe is free for decode rounds and other prefills.
# Splitting at multiples of GRID is bit-identical to one single chunk (gate test_parity/--splits). 0 = off.
CHUNK = int(os.environ.get("TOBU_FNX_PREFILL_CHUNK", "4096")) // GRID * GRID
# Alone in the server (no other slot busy) the small chunk helps nobody: then the large chunk
# (16,384 instead of 4,096 = +6–10 % measured cold, bit-identical). As soon as a second slot is busy,
# CHUNK applies again – a newly arriving request waits at most one large chunk. 0 = off (always CHUNK).
CHUNK_ALONE = int(os.environ.get("TOBU_FNX_PREFILL_CHUNK_ALONE", "16384")) // GRID * GRID
YIELD_MAX_S = float(os.environ.get("TOBU_FNX_PREFILL_YIELD_S", "2.0"))   # share for the others per chunk (at most)   # grid points of the prefix memory (multiples of the compression factor 4)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # tobu-dist: server/, engine/, scripts/


class FnxError(RuntimeError):
    pass


def _words(text: str) -> list[int]:
    """UTF-8 text as [length in bytes, int32 words...] (framed protocol, command CONFIG)."""
    b = text.encode("utf-8")
    pad = b + b"\0" * (-len(b) % 4)
    return [len(b), *struct.unpack(f"<{len(pad) // 4}i", pad)]


class FnxProc:
    """Child process tobu-fnx with framed protocol (engine/tobu/)."""

    def __init__(self, cmd: list[str], stderr=None):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, bufsize=0)
        self.lock = threading.RLock()             # prefix memory and corpus: one thread at a time (slots)
        self.waiting = 0                          # threads currently waiting for the pipe (fair prefills)
        self._wl = threading.Lock()

    @contextlib.contextmanager
    def hold(self):
        """Take the pipe while counting as a waiter (yield_pipe sees whether someone wants it)."""
        with self._wl:
            self.waiting += 1
        self.lock.acquire()
        with self._wl:
            self.waiting -= 1
        try:
            yield
        finally:
            self.lock.release()

    def yield_pipe(self, max_s: float, quiet_s: float = 0.02, busy=None):
        """Fair prefills: between two chunks leave the pipe to others (decode rounds, short prefills) until
        nobody has been waiting for quiet_s (and busy() does not report that other slots are still decoding) or max_s has
        passed."""
        t0 = q0 = time.perf_counter()
        while (now := time.perf_counter()) - t0 < max_s:
            if self.waiting or (busy is not None and busy()):
                q0 = now
            elif now - q0 >= quiet_s:
                return
            time.sleep(0.002)

    def _read(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            b = self.p.stdout.read(n - len(buf))
            if not b:
                raise FnxError("tobu-fnx exited")
            buf += b
        return bytes(buf)

    def _send(self, cmd: int, data) -> tuple[int, bytes]:
        self.p.stdin.write(struct.pack(f"<ii{len(data)}i", cmd, len(data), *data))
        self.p.stdin.flush()
        st, n = struct.unpack("<ii", self._read(8))
        return st, (self._read(4 * n) if n > 0 else b"")

    def call_blob(self, cmd: int, head: list[int], blob: bytes) -> list[int]:
        """Command with int32 head and raw bytes (padded to 4) – images without a detour through an int list."""
        pad = blob + b"\0" * (-len(blob) % 4)
        with self.hold():
            self.p.stdin.write(struct.pack(f"<ii{len(head)}i", cmd, len(head) + len(pad) // 4, *head) + pad)
            self.p.stdin.flush()
            st, n = struct.unpack("<ii", self._read(8))
            body = self._read(4 * n) if n > 0 else b""
        if st != 0:
            raise FnxError(f"tobu-fnx command {cmd}: status {st}")
        return list(struct.unpack(f"<{len(body) // 4}i", body))

    def call(self, cmd: int, *data: int, raw: bool = False, slot: int | None = None):
        """One command; with `slot`, SELECT first (atomic under the same lock, parallel requests per slot)."""
        with self.hold():
            if slot is not None and slot != getattr(self, "cur", 0):
                st, _ = self._send(SELECT, [slot])
                if st != 0:
                    raise FnxError(f"tobu-fnx SELECT {slot}: status {st}")
                self.cur = slot
            st, body = self._send(cmd, data)
        if st != 0:
            raise FnxError(f"tobu-fnx command {cmd}: status {st}")
        return body if raw else list(struct.unpack(f"<{len(body) // 4}i", body))

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=60)
        except Exception:  # noqa: BLE001
            self.p.kill()


def engine_schema(schema):
    """Prepare a JSON schema for Gufo's JsonConstraint (meaning unchanged): `enum`/`const` without `type` get the
    type of their values (Gufo requires a type; JSON Schema and llama.cpp do not) – some clients generate
    `{"enum": [...]}`. Recursive over all subschemas; the original stays unchanged."""
    def typ(v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "boolean"
        if isinstance(v, int):
            return "integer"
        if isinstance(v, float):
            return "number"
        if isinstance(v, str):
            return "string"
        return "array" if isinstance(v, list) else "object"

    def walk(x):
        if isinstance(x, list):
            return [walk(v) for v in x]
        if not isinstance(x, dict):
            return x
        out = {k: walk(v) for k, v in x.items()}
        if "type" not in out and ("enum" in out or "const" in out):
            vals = out["enum"] if "enum" in out and isinstance(out["enum"], list) else [out.get("const")]
            ts = sorted({typ(v) for v in vals})
            if "integer" in ts and "number" in ts:
                ts.remove("integer")
            if ts:
                out["type"] = ts[0] if len(ts) == 1 else ts
        return out
    return walk(schema)


class _StepBatcher:
    """S2: batched decode. Every slot registers its engine round; whoever gets the pipe (FnxProc.lock) waits
    briefly until all currently decoding slots are registered (at most `wait_s`, at most 8), and sends them as
    one STEPB. Per slot the result is bit-identical to STEP (Gufo DecodeBatch/EvaluateBatch, checked exactly); only the
    throughput changes (one round for all instead of one per slot)."""

    def __init__(self, proc: "FnxProc", wait_s: float = 0.004):
        self.proc, self.wait_s = proc, wait_s
        self.cv = threading.Condition()
        self.q: list[dict] = []
        self.decoding: set[int] = set()
        self.sizes: Counter = Counter()           # batch sizes (diagnostics, /props)

    def part(self, slot: int):
        with self.cv:
            self.decoding.discard(slot)
            self.cv.notify_all()

    def step(self, slot: int, max_tokens: int) -> list[int]:
        me = {"slot": slot, "n": max_tokens, "res": None, "err": None}
        with self.cv:
            self.decoding.add(slot)
            self.q.append(me)
            self.cv.notify_all()
        while me["res"] is None and me["err"] is None:
            with self.proc.hold():
                if me["res"] is not None or me["err"] is not None:
                    break
                with self.cv:
                    end = time.monotonic() + self.wait_s
                    while len(self.q) < min(8, len(self.decoding)):
                        left = end - time.monotonic()
                        if left <= 0:
                            break
                        self.cv.wait(left)
                    batch, self.q = self.q[:8], self.q[8:]
                if batch:
                    self._run(batch)
        if me["err"] is not None:
            raise me["err"]
        return me["res"]

    def _run(self, batch: list[dict]):
        self.sizes[len(batch)] += 1
        try:
            if len(batch) == 1:                    # alone: the same path as without batching
                b = batch[0]
                b["res"] = self.proc.call(STEP, b["n"], slot=b["slot"])
                return
            data = [len(batch)] + [v for b in batch for v in (b["slot"], b["n"])]
            st, body = self.proc._send(STEPB, data)
            if st != 0:
                raise FnxError(f"tobu-fnx STEPB: status {st}")
            r, i = list(struct.unpack(f"<{len(body) // 4}i", body)), 0
            for b in batch:
                status, n = r[i], r[i + 2]
                if status != 0:
                    b["err"] = FnxError(f"tobu-fnx STEPB slot {b['slot']}: status {status}")
                else:
                    b["res"] = r[i + 1:i + 3 + n + 3]     # like STEP: stop, n, tok[n], cycles, drafted, accepted
                i += 3 + n + 3
        except Exception as e:  # noqa: BLE001 – error to all waiters of the round
            for b in batch:
                if b["res"] is None:
                    b["err"] = e


class DiskTier:
    """Persistent prefix memory (otherwise cold 105k prompts after a restart/LRU overflow): snapshots in
    Gufo format (check header: magic, version, vocabulary, size) as files under <dir>/<fingerprint>/, index in
    index.json. Fingerprint = start command of the engine including size/time of all files in it (model, MTP head, PLE,
    binary) -> a new directory after every change, never foreign states. Written on eviction
    from RAM and on stop (only entries of at least min_n tokens), read on a hit (LOADFILE). Budget in MiB, LRU.
    Folder 0700: states come from user prompts and stay local."""

    def __init__(self, root: str, fingerprint: str, budget_mb: int, min_n: int):
        import shutil

        base = Path(root).expanduser()
        base.mkdir(parents=True, exist_ok=True)
        os.chmod(base, 0o700)
        self.dir = base / fingerprint
        self.dir.mkdir(exist_ok=True)
        os.chmod(self.dir, 0o700)
        self.budget, self.min_n = budget_mb, min_n
        for old in base.iterdir():               # old fingerprints (other engine/models) removed after 7 days
            if old.is_dir() and old != self.dir and time.time() - old.stat().st_mtime > 7 * 86400:
                shutil.rmtree(old, ignore_errors=True)
        for t in self.dir.glob("*.tmp"):         # aborted writes
            t.unlink(missing_ok=True)
        try:
            idx = json.loads((self.dir / "index.json").read_text())
        except (OSError, ValueError):
            idx = {}
        self.entries = {k: e for k, e in idx.items() if (self.dir / e["file"]).exists()}
        self.loads = self.writes = 0
        self._save()

    def path(self, k: str) -> str:
        return str(self.dir / self.entries[k]["file"]) if k in self.entries else str(self.dir / f"{k}.snap")

    def ready(self, k: str) -> bool:
        return k in self.entries and (self.dir / self.entries[k]["file"]).exists()   # otherwise the thread is still writing

    def add(self, k: str, n: int, mb: int, t: float):
        self.entries[k] = {"n": n, "mb": mb, "t": t, "file": f"{k}.snap"}
        self.writes += 1
        while sum(e["mb"] for e in self.entries.values()) > self.budget and len(self.entries) > 1:
            old = min(self.entries, key=lambda x: self.entries[x]["t"])
            (self.dir / self.entries.pop(old)["file"]).unlink(missing_ok=True)
        self._save()

    def touch(self, k: str):
        if k in self.entries:
            self.entries[k]["t"] = time.time()
            self._save()

    def remove(self, k: str):
        e = self.entries.pop(k, None)
        if e is not None:
            (self.dir / e["file"]).unlink(missing_ok=True)
            self._save()

    def _save(self):
        tmp = self.dir / "index.json.new"
        tmp.write_text(json.dumps(self.entries))
        os.chmod(tmp, 0o600)
        tmp.replace(self.dir / "index.json")

    def info(self) -> dict:
        return {"entries": len(self.entries), "mb": sum(e["mb"] for e in self.entries.values()), "loads": self.loads,
                "writes": self.writes, "dir": str(self.dir)}


def fingerprint(cmd: list[str], state_id: str = "") -> str:
    """Key of the SSD memory. With state_id (tobu-fnx INFO, hash of the state-determining sources): start command
    without executables (binary, wrapper) and without --mmproj (images are not stored) + size/time of the
    other named files (model parts, MTP head, PLE) + state_id – a new build with the same state keeps the
    memory. Without state_id (old binary): as before over everything, including the binary."""
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    if state_id:
        h.update(b"state:" + state_id.encode() + b"\0")
        keep, skip = [], False
        for a in cmd:
            if skip:
                skip = False
                continue
            if a == "--mmproj":
                skip = True
                continue
            if os.path.isfile(a) and os.access(a, os.X_OK):
                continue
            keep.append(a)
        cmd = keep
    for a in cmd:
        h.update(a.encode() + b"\0")
        for f in [a] + ([str(Path(a).parent / Path(a).name.replace("-00001-of-", f"-{i:05d}-of-"))
                         for i in range(2, 10)] if "-00001-of-" in a else []):
            if os.path.isfile(f):
                st = os.stat(f)
                h.update(f"{f}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


class PrefixStore:
    """Prefix memory: states only at grid points of the prompt prefill (multiples of GRID), captured in
    the same pass (tobu-fnx SYNCCP = Gufo SyncThrough); never states from decode (decode computes with different
    kernels than prefill – continuing the live state is not bit-identical to a cold start, measured). A
    hit computes from the grid point like a cold start -> output depends only on the prompt.

    GRID (TOBU_FNX_GRID, default 4 = compression factor of the attention): snapshots at multiples of 4 are bit-identical
    to a cold start (logits + tokens, ctx-8k, boundaries 4100…8700), arbitrary boundaries are not (8701, 8703, 4359). A fine
    grid = an agent turn computes only its new part (at most GRID-1 tokens more) instead of up to 1023.
    Key: hash of the whole prefix per entry; LRU by MiB budget."""

    def __init__(self, proc: "FnxProc", budget_mb: int, disk: DiskTier | None = None):
        self.p, self.budget, self.disk = proc, budget_mb, disk
        self.entries: dict[bytes, dict] = {}     # key -> {n, slot, mb, t}
        self.next_slot, self.clock = 1, 0
        self.hits = self.misses = 0

    @staticmethod
    def key(ids, n: int, vision: list | None = None) -> bytes:
        """Hash of the prefix ids[:n]. TOB-37 phase 5: all image tokens have the same id – images starting before n
        enter the key with content (sha), grid and position; without such images it is the same as for text."""
        import hashlib

        h = hashlib.blake2b(np.asarray(ids[:n], dtype=np.int32).tobytes(), digest_size=16)
        imgs = [(img, off) for img, off in vision or [] if off < n]
        if imgs:
            # version of the image snapshots: v2 = chunk boundaries on grid points + full layout after LOAD;
            # snapshots of older computation paths never match (otherwise warm != cold). Text keys stay unchanged.
            h.update(f"|imgv{IMG_SNAP_VERSION}".encode())
        for img, off in imgs:
            h.update(f"|img:{off}:{img.gh}x{img.gw}:{img.detail}:{img.sha}".encode())
        return h.digest()

    @staticmethod
    def inside_image(n: int, vision: list | None) -> bool:
        return any(off < n < off + img.tokens for img, off in vision or [])

    def lookup(self, ids: list[int], vision: list | None = None) -> tuple[int, int | str | None]:
        """Longest stored grid point g < len(ids) whose prefix matches -> (g, slot) or (0, None).
        All matching entries are kept fresh (LRU), not only the one used: the snapshot before the first
        user message (system + tools) is a prefix of every agent turn and is only needed after a compaction
        – without that it would drop out as the oldest entry in a long conversation."""
        best = None
        ns = {e["n"] for e in self.entries.values() if e["n"] < len(ids)}
        if self.disk is not None:
            ns |= {e["n"] for e in self.disk.entries.values() if e["n"] < len(ids)}
        for n in sorted(ns, reverse=True):
            if self.inside_image(n, vision):
                continue
            k = self.key(ids, n, vision)
            e = self.entries.get(k)
            if e is not None:
                self.clock += 1
                e["t"] = self.clock
                if best is None:
                    best = e
            elif best is None and self.disk is not None and self.disk.ready(k.hex()):
                best = {"n": n, "file": k.hex()}       # SSD: only if there is no longer RAM hit
        if best is not None and "file" in best:
            self.hits += 1
            self.disk.touch(best["file"])
            return best["n"], self.disk.path(best["file"])
        if best is not None:
            best["t"] = self.clock + 1        # the used entry last
            self.clock += 1
            self.hits += 1
            return best["n"], best["slot"]
        self.misses += 1
        return 0, None

    def drop_point(self, ids: list[int], n: int, vision: list | None = None):
        """Remove the entry (RAM and SSD) of a grid point, e.g. when the engine rejects it."""
        k = self.key(ids, n, vision)
        e = self.entries.pop(k, None)
        if e is not None and "slot" in e:
            self.p.call(DROP, e["slot"])
        if self.disk is not None:
            self.disk.remove(k.hex())

    def plan(self, ids: list[int], start: int, marks: list[int], vision: list | None = None) -> list[int]:
        """New grid points (> start, < len(ids)): per mark (e.g. first user message, prompt end) the last
        grid point before it that is not stored yet (never in the middle of an image)."""
        pts = set()
        for m in marks:
            g = (min(m, len(ids) - 1) // GRID) * GRID
            while g > start and self.inside_image(g, vision):
                g -= GRID
            if g > start and self.key(ids, g, vision) not in self.entries:
                pts.add(g)
        return sorted(pts)

    def new_slot(self) -> int:
        self.next_slot += 1
        return self.next_slot

    def add(self, ids: list[int], points: list[int], slots: list[int], mbs: list[int], vision: list | None = None):
        for g, sl, mb in zip(points, slots, mbs):
            if mb <= 0:
                continue
            if self.key(ids, g, vision) in self.entries:   # a parallel prefill has already stored the same point
                self.p.call(DROP, sl)
                continue
            self.clock += 1
            self.entries[self.key(ids, g, vision)] = {"n": g, "slot": sl, "mb": mb, "t": self.clock}
        self._evict()

    def _evict(self):
        while sum(e["mb"] for e in self.entries.values()) > self.budget and len(self.entries) > 1:
            k = min(self.entries, key=lambda k: self.entries[k]["t"])
            e = self.entries.pop(k)
            self.persist(k, e)                   # to the SSD before discarding (if enabled and large enough)
            self.p.call(DROP, e["slot"])

    def persist(self, k: bytes, e: dict) -> bool:
        """Write a RAM entry to the SSD (background thread in tobu-fnx) if the tier is on, the entry is large enough
        and not there yet."""
        d = self.disk
        if d is None or e["n"] < d.min_n or k.hex() in d.entries:
            return False
        try:
            mb = self.p.call(PERSIST, e["slot"], *_words(d.path(k.hex())))[0]
        except FnxError as err:
            print(f"prefix memory: SSD write failed ({err})", flush=True)
            return False
        d.add(k.hex(), e["n"], mb, time.time())
        return True

    def persist_hot(self, budget_s: float = 20.0) -> int:
        """On stop: newest RAM entries to the SSD (at most budget_s), then wait for the writers."""
        if self.disk is None:
            return 0
        t0, n = time.perf_counter(), 0
        for k, e in sorted(self.entries.items(), key=lambda kv: -kv[1]["t"]):
            if time.perf_counter() - t0 > budget_s:
                break
            n += self.persist(k, e)
        self.p.call(PERSIST_WAIT)
        return n

    def clear(self):
        for e in self.entries.values():
            self.p.call(DROP, e["slot"])
        self.entries.clear()

    def info(self) -> dict:
        return {"entries": len(self.entries), "mb": sum(e["mb"] for e in self.entries.values()),
                "hits": self.hits, "misses": self.misses, "grid": GRID,
                **({"disk": self.disk.info()} if self.disk is not None else {})}


class FlashNextEngine:
    def __init__(self, gguf: str, exe: str = "", mtp: str = "", max_len: int = 65536, kmax: int = 8, sessions: int = 1):
        import gguf_read as G

        h = G.read_header(gguf)
        self.hf_tok = G.tokenizer(h)
        self.template = h.meta.get("tokenizer.chat_template")
        exe = exe or os.environ.get("TOBU_FNX") or shlex.join(
            [str(ROOT / "scripts" / "run-fnx.sh"), os.environ.get("TOBU_FNX_BIN") or str(ROOT / "engine" / "build" / "tobu-fnx")])
        ple = os.environ.get("TOBU_FNX_PLE", "")      # TOB-40: n-gram table from its own GGUF (engram)
        mmproj = os.environ.get("TOBU_FNX_MMPROJ", "")  # TOB-37: vision tower (lazy: GPU only on the first image)
        cmd = shlex.split(exe) + ["--model", gguf, "--context", str(max_len), "--sessions", str(max(1, sessions))] + \
            (["--mtp", mtp] if mtp else []) + (["--ple", ple] if ple else []) + (["--mmproj", mmproj] if mmproj else [])
        self._attach(cmd, max_len, kmax, sessions=max(1, sessions))

    @classmethod
    def from_proc(cls, cmd: list[str], hf_tok=None, template: str | None = None, max_len: int = 65536,
                  kmax: int = 8, stderr=None, sessions: int = 1) -> "FlashNextEngine":
        """Seam for tests without GGUF/GPU (test_fnx_adapter.py): start child process `cmd` directly (e.g. the
        dummy fixtures/fake_fnx.py), pass tokenizer and chat template ready-made instead of reading them from the GGUF head.
        Otherwise like the normal constructor (INFO, max_len, MTP, name)."""
        fe = cls.__new__(cls)
        fe.hf_tok, fe.template = hf_tok, template
        fe._attach(cmd, max_len, kmax, stderr, sessions=max(1, sessions))
        return fe

    def _attach(self, cmd: list[str], max_len: int, kmax: int, stderr=None, sessions: int = 1):
        self.proc = FnxProc(cmd, stderr=stderr)
        info = self.proc.call(INFO)
        self.vocab, ctx, has_mtp, self.eos, _ = info[:5]
        # state id (newer tobu-fnx): 16 bytes -> hex; old binary: empty (fingerprint as before)
        self.state_id = struct.pack("<4i", *info[5:9]).hex() if len(info) >= 9 else ""
        self.max_len = min(max_len, ctx)
        self.kmax = kmax
        self.mtp = True if has_mtp else None      # the server only takes engine drafts if mtp is not None
        self.name = "tobu/qwen38-flash-next-gfx1151" + ("+mtp" if has_mtp else "")
        # tobu-fnx itself enforces tool grammar, thinking budget and sampling (CONFIG), also in MTP rounds
        self.native_constraints = True
        # tobu-fnx computes sampling with drafts itself (Session::DecodeStep: proposal from the MTP head's distribution,
        # acceptance min(1, p/q), otherwise residual distribution; copies as point mass) – exact in distribution, repeatable
        # with a seed. The server therefore lets the engine draft at full width even at temperature > 0
        # (--no-spec-sampling or X-Tobu-Spec-Sampling: off = without drafts, one = at most one draft per round).
        self.native_spec_sampling = bool(has_mtp)
        # Slots (parallel requests, --slots): one session in tobu-fnx per slot; a request holds its slot from
        # make() until leave() (the server calls leave() after every request, L49). Without enable_slots: only slot 0.
        import queue as _q
        self.sessions = sessions
        self.free: "_q.Queue[int]" = _q.Queue()
        self.free.put(0)
        self.tl = threading.local()
        self.lock = threading.Lock()
        disk = None
        if os.environ.get("TOBU_FNX_DISK_DIR"):   # persistent prefix memory on the SSD (survives restarts)
            try:
                disk = DiskTier(os.environ["TOBU_FNX_DISK_DIR"], fingerprint(cmd, self.state_id),
                                int(os.environ.get("TOBU_FNX_DISK_MB", "65536")),
                                int(os.environ.get("TOBU_FNX_DISK_MIN", "2048")))
                print(f"prefix memory SSD: {disk.dir} ({len(disk.entries)} entries, state "
                      f"{self.state_id[:12] or 'binary'})", flush=True)
            except OSError as e:
                print(f"prefix memory SSD off ({e})", flush=True)
        self.store = PrefixStore(self.proc, int(os.environ.get("TOBU_FNX_CACHE_MB", "8192")), disk)
        self.use_store = os.environ.get("TOBU_FNX_CACHE", "1") != "0"
        # S2: batch rounds of parallel slots (STEPB); TOBU_FNX_BATCH=0 disables it (slots then take STEP in turns)
        self.batcher = _StepBatcher(self.proc, float(os.environ.get("TOBU_FNX_BATCH_WAIT_MS", "4")) / 1000) \
            if sessions > 1 and os.environ.get("TOBU_FNX_BATCH", "1") != "0" else None
        self.vision = "--mmproj" in cmd           # TOB-37: images possible
        self._imgs: dict[str, int] = {}           # sha -> id in the engine's image store
        self.user_marker: list[int] = []
        if self.hf_tok is not None:              # start of the first user message (system + tools before it)
            um = self.hf_tok.encode("<|im_start|>user", add_special_tokens=False).ids
            self.user_marker = um if um and um[0] == self.hf_tok.token_to_id("<|im_start|>") else []

    def enable_slots(self, n: int, *_a, **_k):
        """L49 interface of the server: n parallel requests (at most as many as sessions in tobu-fnx)."""
        if n > self.sessions:
            raise ValueError(f"--slots {n} > sessions in tobu-fnx ({self.sessions})")
        for k in range(1, n):
            self.free.put(k)
        self.enabled = n

    def _acquire(self) -> int:
        k = getattr(self.tl, "slot", None)
        if k is None:
            k = self.free.get()                   # waits until a slot is free
            self.tl.slot = k
        return k

    def leave(self):
        k = getattr(self.tl, "slot", None)
        if k is not None:
            self.tl.slot = None
            if self.batcher is not None:
                self.batcher.part(k)
            self.free.put(k)

    def slot_info(self) -> dict:
        info = {"slots": self.sessions, "free": self.free.qsize()}
        if self.batcher is not None:
            info["batches"] = dict(sorted(self.batcher.sizes.items()))
        return info

    def round(self, slot: int, max_tokens: int) -> list[int]:
        """One engine round of the slot: stop, n, tok[n], cycles, drafted, accepted (batched if S2 is on)."""
        if self.batcher is not None:
            return self.batcher.step(slot, max_tokens)
        return self.proc.call(STEP, max_tokens, slot=slot)

    def make(self, prompt_ids: list[int]):
        return _Verifier(self, self._acquire())

    def clear_cache(self, disk: bool = True):
        self.store.clear()

    def draft_stats(self, slot: int) -> dict | None:
        """B: draft counters of the slot since start (tobu-fnx STATS); None with an old binary."""
        try:
            v = self.proc.call(STATS, slot=slot)
        except FnxError:
            return None
        return {k: (v[2 * i] & 0xffffffff) | ((v[2 * i + 1] & 0xffffffff) << 32) for i, k in enumerate(STATS_KEYS)}

    # Engine corpus (tobu/corpus.hpp, copy source of the drafts – speed only, every copy is verified).
    # Cost in the engine: Set/Index O(document); every Drop and every Set on an existing id rebuilds the whole index
    # (O(all tokens), ~0.7–0.9 s at 2 M tokens). Therefore:
    #  - Never sync in the request path: sync_corpus() only records the target state, a background thread applies it.
    #  - No Drop and no replace: removed/changed files stay as "dead" (changed ones get a
    #    new id); only when dead > live (and > TOBU_FNX_CORPUS_STALE_MIN) is it cleared once (CORPUS_DROP -1) and
    #    set anew. Same for the answer memory at its limit.
    #  - In slices: at most TOBU_FNX_CORPUS_SLICE tokens (100 k ≈ 50 ms engine) per turn; in between all slots get
    #    the pipe. Own lock (_clock), not self.lock: prefill/prefix memory never wait for the corpus.
    #  - New target state during a sync: the running one stops after the current slice and replans.
    def _corpus_init(self) -> None:
        if getattr(self, "_cthread", None) is not None:
            return
        import collections
        self._clock = threading.Lock()                     # corpus state (not the pipe, not self.lock)
        self._ccv = threading.Condition(self._clock)
        self._ctarget, self._cgen, self._cdone = None, 0, 0   # target state {rel: (h, ids)}, generation, applied up to
        self._corpus: dict = {}                            # rel -> (did, h, n) in the engine (live)
        self._corpus_next, self._clive, self._cstale = 0, 0, 0
        self._ccompact, self._cbusy = False, False
        self.__dict__.setdefault("_xstore", collections.deque())   # (did, array) – for re-setting
        self._xstore_n = getattr(self, "_xstore_n", 0)
        self._cstop = False
        self._cthread = threading.Thread(target=self._corpus_worker, name="tobu-corpus", daemon=True)
        self._cthread.start()

    def _corpus_set(self, did: int, ids) -> None:
        self.proc.call(CORPUS_SET, did, *ids)

    def _corpus_worker(self) -> None:
        import array
        slice_tok = max(1, int(os.environ.get("TOBU_FNX_CORPUS_SLICE", "100000")))
        stale_min = int(os.environ.get("TOBU_FNX_CORPUS_STALE_MIN", "200000"))
        while True:
            with self._ccv:
                while not self._cstop and self._cdone == self._cgen and not self._ccompact:
                    self._ccv.wait()
                if self._cstop:
                    return
                gen, target = self._cgen, dict(self._ctarget or {})
                have = dict(self._corpus)
                self._cbusy = True
            try:
                # plan: count dead ones, set new/changed ones; too much dead -> clear once and set everything anew
                with self._clock:
                    for rel, (did, h, n) in list(have.items()):
                        if rel not in target or target[rel][0] != h:
                            self._cstale += n
                            self._clive -= n
                            del have[rel]
                    self._corpus = have
                    compact = self._ccompact or (self._cstale > self._clive + self._xstore_n and self._cstale > stale_min)
                    self._ccompact = False
                    xs_ids, xs = [], []
                    if compact:                              # under _clock: remember_output does not set in between
                        self.proc.call(CORPUS_DROP, -1)      # clearing only costs the freeing
                        self._corpus, self._clive, self._cstale = {}, 0, 0
                        xs = list(self._xstore)
                        xs_ids = [d for d, _ in xs]
                todo = [(rel, h, ids) for rel, (h, ids) in target.items() if rel not in self._corpus]
                todo += [(None, None, a) for _, a in xs]
                # apply in slices; new target state -> replan after the slice
                budget, k, aborted = 0, 0, False
                for rel, h, ids in todo:
                    if rel is None:                          # re-insert the answer memory after clearing
                        self._corpus_set(xs_ids[k], ids)
                        k += 1
                    else:
                        arr = ids if isinstance(ids, array.array) else array.array("i", ids)
                        with self._clock:
                            self._corpus_next += 1
                            did = self._corpus_next
                        self._corpus_set(did, arr)
                        with self._clock:
                            self._corpus[rel] = (did, h, len(arr))
                            self._clive += len(arr)
                    budget += len(ids)
                    if budget >= slice_tok:
                        budget = 0
                        time.sleep(0)                        # pipe free for slots (FnxProc.hold per command)
                        with self._clock:
                            if self._cgen != gen and not compact:
                                aborted = True
                                break
                with self._ccv:
                    if not aborted:
                        self._cdone = gen
                    self._cbusy = False
                    self._ccv.notify_all()
            except Exception as e:                           # noqa: BLE001 – drafts only; next target state anew
                print(f"corpus sync: {type(e).__name__}: {e}"[:300], flush=True)
                with self._ccv:
                    self._cdone = gen
                    self._cbusy = False
                    self._ccompact = True                    # state unclear -> clean rebuild next time
                    self._ccv.notify_all()

    def corpus_idle(self, timeout: float = 10.0) -> bool:
        """For tests/metrics: waits until the background sync has applied the last target state."""
        if getattr(self, "_cthread", None) is None:
            return True
        with self._ccv:
            return self._ccv.wait_for(lambda: self._cdone == self._cgen and not self._ccompact and not self._cbusy, timeout)

    def remember_output(self, ids: list[int]) -> None:
        """Memory of earlier answers (like vLLM suffix / llama.cpp dynamic cache): finished answers as
        corpus documents, across all requests; the engine only queries the corpus if the history has no hit.
        At most TOBU_FNX_XSTORE_TOKENS tokens (0 = off); above the limit the oldest drop out of the list in a batch (down to 3/4)
        and count as dead – clearing happens in the background (see above). Speed only."""
        cap = int(os.environ.get("TOBU_FNX_XSTORE_TOKENS", "1000000"))
        if cap <= 0 or len(ids) < 16:
            return
        import array
        self._corpus_init()
        arr = array.array("i", ids[-cap:])
        with self._ccv:                                      # under _clock: no clearing by the background in between
            self._xstore_next = getattr(self, "_xstore_next", 0) + 1
            did = XSTORE_BASE + self._xstore_next % XSTORE_BASE
            try:
                self._corpus_set(did, arr)                   # one small document: index O(answer), no rebuild
            except FnxError:
                return
            self._xstore.append((did, arr))
            self._xstore_n += len(arr)
            if self._xstore_n > cap and len(self._xstore) > 1:
                while self._xstore_n > cap * 3 // 4 and len(self._xstore) > 1:
                    n = len(self._xstore.popleft()[1])
                    self._xstore_n -= n
                    self._cstale += n
                stale_min = int(os.environ.get("TOBU_FNX_CORPUS_STALE_MIN", "200000"))
                if self._cstale > self._clive + self._xstore_n and self._cstale > stale_min:
                    self._ccompact = True
                    self._ccv.notify_all()

    def sync_corpus(self, files: dict) -> dict:
        """Documents ({rel: (mtime, size, ids, ...)}, token ids at index 2) as
        the engine's copy source, at most TOBU_FNX_CORPUS_TOKENS tokens (smallest first). Returns immediately: the
        sync runs in the background (see above); the request uses what is already in the corpus. The output never depends
        on it (every copy is verified), only the speed."""
        self._corpus_init()
        cap = int(os.environ.get("TOBU_FNX_CORPUS_TOKENS", "2000000"))
        want, total, skipped = {}, 0, {"short": 0, "cap": 0}
        for rel, e in sorted(files.items(), key=lambda kv: len(kv[1][2])):
            ids = e[2]
            if len(ids) < 16 or total + len(ids) > cap:
                skipped["short" if len(ids) < 16 else "cap"] += 1
                continue
            want[rel] = (hash(tuple(ids)), ids)
            total += len(ids)
        with self._ccv:
            cur = self._ctarget or {}
            if want.keys() != cur.keys() or any(want[r][0] != cur[r][0] for r in want):
                self._ctarget, self._cgen = want, self._cgen + 1
                self._ccv.notify_all()
            return {"docs": len(self._corpus), "tokens": self._clive, "target_docs": len(want), "target_tokens": total,
                    "pending": self._cdone != self._cgen, "stale": self._cstale, "skipped": skipped, "cap": cap}

    def first_user(self, ids: list[int]) -> int:
        um = self.user_marker
        if not um:
            return 0
        for i in range(len(ids) - len(um) + 1):
            if ids[i:i + len(um)] == um:
                return i
        return 0

    def _image_id(self, img) -> int:
        """Image into the engine's image store (once per content); checks the grid against media.grid()."""
        with self.__dict__.setdefault("img_lock", threading.RLock()):
            iid = self._imgs.get(img.sha)
            if iid is not None:
                return iid
            iid = len(self._imgs) + 1 + getattr(self, "_img_base", 0)
            r = self.proc.call_blob(IMAGE, [iid, img.detail, img.width, img.height, img.max_pixels], img.rgb)
            if (r[0], r[1]) != (img.gh, img.gw):
                raise FnxError(f"image grid of the engine {r[0]}x{r[1]} != server {img.gh}x{img.gw}")
            self._imgs[img.sha] = iid
            return iid

    def _vprompt(self, sess: int, vision: list):
        for attempt in (0, 1):
            data = [len(vision)]
            for img, off in vision:
                data += [self._image_id(img), off]
            try:
                self.proc.call(VPROMPT, *data, slot=sess)
                self.__dict__.setdefault("vis_slots", set()).add(sess)   # layout on the slot (text prefill resolves it)
                return
            except FnxError as e:
                if attempt or "Status -4" not in str(e):
                    raise
                with self.__dict__.setdefault("img_lock", threading.RLock()):   # engine has evicted: upload again
                    self._img_base = getattr(self, "_img_base", 0) + len(self._imgs) + 1
                    self._imgs.clear()

    def prefill(self, ids: list[int], use_cache: bool = True, slot: int | None = None, progress=None,
                vision: list | None = None) -> int:
        """Bring the prompt into the state: longest grid point from the memory (LOAD), otherwise cold (RESET); compute the rest
        with SYNC/SYNCCP and capture at most two new grid points along the way (before the first user message
        and before the prompt end). The rest runs in chunks of CHUNK tokens (grid points); only memory accesses
        are under self.lock, each chunk takes the pipe individually -> other slots get in between.
        progress(done, total, eta_s | None) after each chunk. -> reused tokens."""
        st = self.store
        sess = self._acquire() if slot is None else slot      # rewarm/warm-up without a verifier: slot of the thread
        vision = vision or []
        if vision:
            return self._prefill_vision(ids, use_cache, sess, progress, vision)
        if sess in getattr(self, "vis_slots", set()):
            # phase 5: after a LOAD of an image snapshot the layout stays on the slot (tobu-fnx considers it resolved) –
            # resolve it explicitly before text, otherwise a later RESET/SYNC would compute with old image positions
            self.proc.call(VPROMPT, 0, slot=sess)
            self.vis_slots.discard(sess)
        with self.lock:
            g, snap = st.lookup(ids) if (use_cache and self.use_store) else (0, None)
            if isinstance(snap, str):            # hit on the SSD
                try:
                    self.proc.call(LOADFILE, *_words(snap), slot=sess)
                    st.disk.loads += 1
                except FnxError as e:            # does not match (any more) -> remove file, cold
                    print(f"prefix memory: SSD snapshot discarded ({e})", flush=True)
                    st.disk.remove(Path(snap).stem)
                    g, snap = 0, None
            elif snap is not None:
                self.proc.call(LOAD, snap, slot=sess)
            # origin of the hit for the log (cache_src): ram | ssd | miss
            self.tl.cache_src = "miss" if snap is None else "ssd" if isinstance(snap, str) else "ram"
            if snap is None:
                self.proc.call(RESET, slot=sess)
            pts = st.plan(ids, g, [self.first_user(ids), len(ids) - 1]) if self.use_store else []
            cps = [st.new_slot() for _ in pts]
        total, done, t0 = len(ids), g, time.perf_counter()
        while done < total:
            e = self._next_end(done, total, pts, self._chunk_now())
            if progress is not None:
                el = time.perf_counter() - t0
                progress(done, total, round((total - done) * el / (done - g), 1) if done > g and el > 0 else None)
            tc = time.perf_counter()
            mine = [(p, c) for p, c in zip(pts, cps) if done < p < e]
            if mine:
                r = self.proc.call(SYNCCP, len(mine), *[p for p, _ in mine], *[c for _, c in mine], *ids[:e], slot=sess)
                with self.lock:
                    st.add(ids, [p for p, _ in mine], [c for _, c in mine], r[2:2 + len(mine)])
            else:
                self.proc.call(SYNC, *ids[:e], slot=sess)
            dt = time.perf_counter() - tc
            done = e
            if done < total:
                self._yield(sess, dt)
        if progress is not None:
            progress(total, total, 0.0)
        return g

    def _yield(self, sess: int, dt: float):
        if getattr(self, "enabled", 1) - self.free.qsize() > 1:
            # other slots active: leave them the pipe at most as long as this chunk (≤ YIELD_MAX_S)
            bt = self.batcher
            self.proc.yield_pipe(min(dt, YIELD_MAX_S),
                                 busy=(lambda: len(bt.decoding) > (sess in bt.decoding)) if bt is not None else None)

    def _prefill_vision(self, ids: list[int], use_cache: bool, sess: int, progress, vision: list) -> int:
        """TOB-37 phase 5: prefix memory also behind images. Key with image identities (PrefixStore.key);
        Gufo captures in-pass checkpoints (SYNCCP) only without an image layout -> SYNC exactly up to the point and SAVE. Hit behind an image: first the layout (VPROMPT), then LOAD
        (Gufo checks the snapshot's image identity against the layout); hit before it: LOAD, then VPROMPT (a
        text snapshot resolves the layout). Chunk boundaries at fixed positions (multiples of CHUNK from the prompt start, never in
        an image, plus the image start and the checkpoint points): cold and with a hit, the chunks computed behind the hit are
        the same."""
        st = self.store
        cut = vision[0][1]
        with self.lock:
            g, snap = st.lookup(ids, vision) if (use_cache and self.use_store) else (0, None)
            img_hit = snap is not None and g > cut
            if img_hit:
                self._vprompt(sess, vision)
            if isinstance(snap, str):
                try:
                    self.proc.call(LOADFILE, *_words(snap), slot=sess)
                    st.disk.loads += 1
                except FnxError as e:
                    print(f"prefix memory: SSD snapshot discarded ({e})", flush=True)
                    st.disk.remove(Path(snap).stem)
                    g, snap, img_hit = 0, None, False
            elif snap is not None:
                try:
                    self.proc.call(LOAD, snap, slot=sess)
                except FnxError as e:            # image snapshot does not match the layout: remove entry, text hit before it
                    print(f"prefix memory: image snapshot at {g} discarded ({e}); images "
                          f"{[(off, img.gh, img.gw) for img, off in vision]}", flush=True)
                    st.drop_point(ids, g, vision)
                    g, snap = st.lookup(ids[:cut + 1]) if (use_cache and self.use_store) else (0, None)
                    img_hit = False
                    if isinstance(snap, str):
                        try:
                            self.proc.call(LOADFILE, *_words(snap), slot=sess)
                        except FnxError:
                            g, snap = 0, None
                    elif snap is not None:
                        self.proc.call(LOAD, snap, slot=sess)
            if snap is None:
                self.proc.call(RESET, slot=sess)
            elif img_hit:
                # when restoring, the engine sets the snapshot's layout (only images before the point) – later
                # images of this request would then be missing (wrong positions); append the full layout again
                self._vprompt(sess, vision)
            pts = st.plan(ids, g, [self.first_user(ids), len(ids) - 1], vision) if self.use_store else []
            cps = [st.new_slot() for _ in pts]
        total, done, t0 = len(ids), g, time.perf_counter()
        ends = self._vision_ends(g, total, vision, pts)
        if not img_hit:                          # layout before the first chunk (after a text LOAD, which resolves it)
            self._vprompt(sess, vision)
        for i, e in enumerate(ends):
            if progress is not None:
                el = time.perf_counter() - t0
                progress(done, total, round((total - done) * el / (done - g), 1) if done > g and el > 0 else None)
            tc = time.perf_counter()
            self.proc.call(SYNC, *ids[:e], slot=sess)
            for p_, c_ in [(p, c) for p, c in zip(pts, cps) if p == e]:   # save the state exactly at the point
                r = self.proc.call(SAVE, c_, slot=sess)
                b = (r[0] & 0x7fffffff) | (r[1] << 31)
                with self.lock:
                    st.add(ids, [p_], [c_], [(b + (1 << 20) - 1) >> 20], vision)
            dt = time.perf_counter() - tc
            done = e
            if done < total:
                self._yield(sess, dt)
        if progress is not None:
            progress(total, total, 0.0)
        return g

    def _chunk_now(self) -> int:
        """Size of the next prefill chunk: alone in the server CHUNK_ALONE (if larger), otherwise CHUNK. Decided anew before
        each chunk – if someone takes a second slot in the meantime, it continues with small chunks."""
        if CHUNK <= 0 or CHUNK_ALONE <= CHUNK:
            return CHUNK
        alone = getattr(self, "enabled", 1) - self.free.qsize() <= 1
        return CHUNK_ALONE if alone else CHUNK

    @staticmethod
    def _next_end(done: int, total: int, points: list[int], chunk: int) -> int:
        """End of the next chunk from `done` (like _chunk_ends, but chunk by chunk with selectable size): multiple
        of GRID from a grid point, never exactly on a checkpoint point, the last remainder in one chunk."""
        if chunk <= 0 or total - done <= chunk:
            return total
        e = done + chunk
        while e in points:
            e += GRID
        return total if e >= total else e

    @staticmethod
    def _vision_ends(start: int, total: int, vision: list, points: list[int]) -> list[int]:
        """Fixed chunk boundaries for prompts with images, all (except the prompt end) on multiples of GRID: Gufo's
        chunk boundaries are only bit-identical to a single-chunk pass there – so cold and with a hit compute the same, even
        when a new image follows the hit. Multiples of CHUNK from the prompt start; if a boundary falls into an
        image, it moves to the grid point before it (or, if that is not behind start, to the one after it)."""
        spans = [(off, off + img.tokens) for img, off in vision]
        cuts = set(points)
        if CHUNK > 0:
            for e in range((start // CHUNK + 1) * CHUNK, total, CHUNK):
                for a, b in spans:
                    if a < e < b:
                        lo = (a // GRID) * GRID
                        e = lo if lo > start else -(-b // GRID) * GRID
                cuts.add(e)
        return sorted(x for x in cuts if start < x < total) + [total]

    @staticmethod
    def _image_ends(start: int, total: int, vision: list) -> list[int]:
        """Chunk boundaries from the first image on: never in the middle of an image (embeddings per image in one piece)."""
        spans = [(off, off + img.tokens) for img, off in vision]
        if CHUNK <= 0:
            return [total]
        ends, e = [], start + CHUNK
        while e < total:
            for a, b in spans:
                if a < e < b:
                    e = b
            if e >= total:
                break
            ends.append(e)
            e += CHUNK
        return ends + [total]

    @staticmethod
    def _chunk_ends(start: int, total: int, points: list[int]) -> list[int]:
        """Chunk boundaries (multiples of GRID from a grid point) up to total; never exactly on a checkpoint point
        (every point lies strictly inside a chunk)."""
        if CHUNK <= 0 or total - start <= CHUNK:
            return [total]
        ends, e = [], start + CHUNK
        while e < total:
            while e in points:
                e += GRID
            if e >= total:
                break
            ends.append(e)
            e += CHUNK
        return ends + [total]

    def persist_hot(self, budget_s: float = 20.0) -> int:
        """Stop: hot prefix entries to the SSD (if TOBU_FNX_DISK_DIR)."""
        with self.lock:
            return self.store.persist_hot(budget_s)

    def alive(self) -> bool:
        """Is tobu-fnx still running? (e.g. after an OOM kill of the container: process ended, every request would otherwise fail
        with BrokenPipe while /health reported 200)"""
        return self.proc.p.poll() is None

    def mem_bytes(self) -> int:
        return 0

    def close(self):
        if getattr(self, "_cthread", None) is not None:
            with self._ccv:
                self._cstop = True
                self._ccv.notify_all()
        self.proc.close()


class _Call:
    """FnxProc with a fixed slot (all session commands of a verifier go to its session)."""

    def __init__(self, proc: FnxProc, slot: int):
        self.proc, self.slot = proc, slot

    def call(self, cmd: int, *data: int, raw: bool = False):
        return self.proc.call(cmd, *data, raw=raw, slot=self.slot)


class _Verifier:
    def __init__(self, fe: FlashNextEngine, slot: int = 0):
        self.fe, self.slot = fe, slot
        self.p = _Call(fe.proc, slot)
        self.queue: list[int] = []        # committed tokens not yet reported (stream)
        self.stopped = False              # the engine stopped at the stop token
        self.mode = "fresh"               # fresh (before the first step) | stream | step
        self.forced = False               # force_step(): whole request in step mode
        self.pending: int | None = None   # step: chosen token, not yet fed in
        self.reuse = 0
        self.config: dict | None = None   # settings per request (server: _native_config), None = greedy without anything
        self.use_cache = True             # the server sets False for cold starts per request (cache_prompt: false)
        self.use_mtp = True
        self.plain = False                # sampling without drafts: every engine round returns exactly one token (no MTP)
        self.progress = None              # server: prefill progress (done, total, eta_s) for metrics/stream
        self.vision: list | None = None   # TOB-37: [(media.Image, offset)] of the request
        self.stats = {"cycles": 0, "drafted": 0, "accepted": 0}
        # Remaining budget per engine round like Gufo's runner (remaining max_tokens of the request): only a large budget
        # allows the engine's wide copy check (up to 64 lines, code/JSON); the server sets budget = max_tokens.
        self.budget = 4096
        self.given = 0                    # tokens delivered to the decoder

    def draft_diff(self) -> dict | None:
        """B: draft counters of this answer per source (difference since start()). Accepted tokens: mtp = chain of the
        MTP head, history = copy from its own history, corpus = copy from corpus documents, store = copy from the
        memory of earlier answers; cycles per source for the savings estimate."""
        d0 = getattr(self, "ds0", None)
        d1 = self.fe.draft_stats(self.slot) if d0 is not None else None
        if d1 is None:
            return None
        d = {k: d1[k] - d0[k] for k in d1}
        if any(x < 0 for x in d.values()):      # session reset in between: no reliable difference
            return None
        ext_a, ext_c = d["corpus_accepted"] + d["store_accepted"], d["corpus_cycles"] + d["store_cycles"]
        return {"cycles": d["cycles"], "accepted": d["accepted"],
                "mtp": d["accepted"] - d["lookup_accepted"], "history": d["lookup_accepted"] - ext_a,
                "corpus": d["corpus_accepted"], "store": d["store_accepted"],
                "copy_cycles": d["lookup_cycles"], "corpus_cycles": d["corpus_cycles"], "store_cycles": d["store_cycles"],
                "history_cycles": d["lookup_cycles"] - ext_c}

    # State: `pending` is the last delivered token, NOT yet fed in; LOGITS is the row it was
    # picked from (tobu meaning of logits()/resample(), like engine_tobu). In the stream an engine round returns the
    # chosen token itself again as the first one – it must equal pending (both greedy from the same row).

    def start(self, prompt_ids: list[int]) -> int:
        if len(prompt_ids) + 64 > self.fe.max_len:
            raise ValueError(f"prompt {len(prompt_ids)} tokens > max_len {self.fe.max_len}")
        self.p.call(CONFIG, *_words(json.dumps(self.config or {}, ensure_ascii=False)))
        self.fe.tl.cache_src = None
        self.reuse = self.fe.prefill(prompt_ids, self.use_cache, slot=self.slot, progress=self.progress,
                                     vision=self.vision)
        self.cache_src = getattr(self.fe.tl, "cache_src", None)      # None (image path): the server derives it
        # B: draft counters only after the prefill (RESET/LOAD/LOADFILE reset the session's counters; the prefill
        # itself drafts nothing)
        self.ds0 = self.fe.draft_stats(self.slot)
        self.queue, self.stopped = [], False
        if self.forced:                   # step mode (wrappers with logits): first token = argmax of the row
            self.mode = "step"
            self.pending = self._argmax()
            self.given = 1
            return self.pending
        # Stream: the engine picks the first token itself (grammar, thinking budget, sampling already apply). PEEK shows it
        # right after the prefill (short TTFT); the first round then returns it again as the first token (verified).
        self.mode = "stream"
        self.pending = self.p.call(PEEK)[0]
        self.given = 1
        return self.pending

    def probe_config(self, config: dict) -> bool:
        """Does the engine accept these settings (e.g. a JSON schema that Gufo's JsonConstraint does not compile)?
        Otherwise the server falls back to the Python wrapper for this request (instead of 500)."""
        try:
            self.p.call(CONFIG, *_words(json.dumps(config or {}, ensure_ascii=False)))
            return True
        except FnxError:
            return False

    def _argmax(self) -> int:
        return int(np.argmax(np.frombuffer(self.p.call(LOGITS, raw=True), dtype=np.float32)))

    # ---- stream (greedy, engine rounds with MTP)
    def _round(self, max_tokens: int) -> int:
        r = self.fe.round(self.slot, max(1, max_tokens))
        stop, n = r[0], r[1]
        toks = r[2:2 + n]
        c, d, a = r[2 + n:5 + n]
        if self.pending is not None:       # first round after start: the token delivered via PEEK comes first
            if not toks or toks[0] != self.pending:
                raise FnxError(f"engine chose {toks[:1]} instead of {self.pending} (PEEK)")
            toks, self.pending = toks[1:], None
        self.stats["cycles"] += c
        self.stats["drafted"] += d
        self.stats["accepted"] += a
        self.queue += toks
        if stop:
            self.stopped = True
            if self.fe.batcher is not None:
                self.fe.batcher.part(self.slot)
        return n

    def _fill(self, want: int):
        while len(self.queue) < want and not self.stopped:
            left = max(1, self.budget - self.given - len(self.queue) + 1)   # +1: the open token counts too
            if self._round(1 if self.plain else left if self.use_mtp else want - len(self.queue) + 1) == 0:
                break

    def self_draft(self, field: str = "free", kind: str = "text") -> list[int]:
        if self.mode == "step":
            return []
        self._fill(2)
        return list(self.queue[:-1]) if len(self.queue) >= 2 else []

    def verify(self, draft: list[int]) -> tuple[int, int]:
        if self.mode == "step":
            return self._verify_step()
        self._fill(len(draft) + 1)
        n = 0
        while n < len(draft) and n < len(self.queue) and draft[n] == self.queue[n]:
            n += 1
        if n >= len(self.queue):           # engine has stopped (stop/context)
            del self.queue[:n]
            return n, self.fe.eos
        nxt = self.queue[n]
        del self.queue[:n + 1]
        self.given += n + 1
        return n, nxt

    # ---- step (sampling, format grammar, thinking budget)
    def force_step(self):
        """The server wraps the verifier (thinking budget, format grammar, sampling) with layers that need logits per position
        or replace tokens: then compute in step mode from the start (without MTP)."""
        self._to_step()
        self.forced = True

    def _to_step(self):
        if self.mode == "stream":
            raise FnxError("logits after the start of the token stream (engine binding)")
        self.mode = "step"

    def logits(self, row: int | None = None):
        self._to_step()
        return np.frombuffer(self.p.call(LOGITS, raw=True), dtype=np.float32).copy()

    def resample(self, tok: int):
        self._to_step()
        self.pending = tok

    def _verify_step(self) -> tuple[int, int]:
        self.p.call(EVAL, self.pending)
        self.pending = self._argmax()
        self.given += 1
        return 0, self.pending
