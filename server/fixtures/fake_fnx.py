"""Stand-in for tobu-fnx for tests without GPU (test_fnx_adapter.py): speaks the framed protocol from
engine/tobu/tobu_fnx.cpp with a deterministic toy model.

    python fake_fnx.py [--vocab 1000] [--context 4096]          (stdin/stdout: frames, little endian int32)

Model: next token = toy_next(second to last, last) (missing tokens count as 0); EOS = vocab - 1.
STEP like the real engine greedy and self-drafting (MTP): one round commits 1 + (position % 4) tokens
(at most max_tokens), stops at EOS (stop = 1, EOS is in the tokens). LOGITS: row with the maximum exactly
at the greedy token. Additionally, for tests only: 99 HIST -> all tokens in the state (prompt + committed/fed).
SAVE is missing (status -1); SYNCCP/LOAD/DROP: snapshot = copy of the history up to the boundary. Slots: --sessions N, 14 SELECT k, 15 STEPB (one round per slot like STEP, batched);
for tests only 98 BATCHES -> [largest batch, number of STEPB]. --delay-ms d: every round takes d ms (GPU time), so that
parallel slots overlap.
"""

from __future__ import annotations

import json
import struct
import sys
import time

import numpy as np

INFO, RESET, SYNC, STEP, EVAL, LOGITS, SAVE, LOAD, DROP = range(9)
HIST = 99
BATCHES = 98
SELECT, STEPB = 14, 15


def toy_next(a: int, b: int, vocab: int) -> int:
    """Greedy successor of (a, b); the test recomputes the same rule independently."""
    if (3 * a + b) % 41 == 0:
        return vocab - 1                       # EOS, roughly every 41st position
    return (7 * a + 13 * b + 5) % (vocab - 1)  # ordinary token (never EOS)


def main():
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    vocab = int(args.get("--vocab", 1000))
    ctx = int(args.get("--context", 4096))
    eos = vocab - 1
    delay = float(args.get("--delay-ms", 0)) / 1000
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    hists: list[list[int]] = [[] for _ in range(int(args.get("--sessions", 1)))]
    hist = hists[0]
    big, nb = 0, 0
    snaps: dict[int, list[int]] = {}
    corpus: dict[int, int] = {}
    stats = {"cycles": 0}

    def nxt(h=None) -> int:
        a, b = ([0, 0] + (hist if h is None else h))[-2:]
        return toy_next(a, b, vocab)

    def round_(h: list[int], max_tokens: int) -> list[int]:
        k = min(max(1, max_tokens), 1 + len(h) % 4)
        toks, stop = [], 0
        for _ in range(k):
            t = nxt(h)
            h.append(t)
            toks.append(t)
            if t == eos:
                stop = 1
                break
        return [stop, len(toks), *toks, 1, k - 1, len(toks) - 1]

    def read(n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            b = inp.read(n - len(buf))
            if not b:
                return None
            buf += b
        return buf

    def reply(status: int, data=(), raw: bytes | None = None):
        if raw is not None:
            out.write(struct.pack("<ii", status, len(raw) // 4) + raw)
        else:
            out.write(struct.pack(f"<ii{len(data)}i", status, len(data), *data))
        out.flush()

    while True:
        h = read(8)
        if h is None:
            return
        cmd, n = struct.unpack("<ii", h)
        data = list(struct.unpack(f"<{n}i", read(4 * n))) if n > 0 else []
        if cmd == INFO:                        # + state id (16 bytes, fixed)
            reply(0, [vocab, ctx, 1, eos, len(hist), 0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10])
        elif cmd == RESET:
            hist.clear()
            reply(0)
        elif cmd == 10:                        # CONFIG (grammar/budget/sampling): the stand-in computes greedy only
            raw = struct.pack(f"<{len(data) - 1}i", *data[1:])[:data[0]] if data else b""
            # test case: a schema with "x-reject" is rejected by the stand-in like the native JsonConstraint (status -3)
            reply(-3 if b'"x-reject"' in raw else 0)
        elif cmd == 11:                        # PEEK: next token without feeding
            reply(0, [nxt()])
        elif cmd == SYNC:
            hist[:] = data
            reply(0, [len(hist), 0])
        elif cmd == STEP:
            stats["cycles"] += 1
            time.sleep(delay)
            reply(0, round_(hist, data[0]) + [0])
        elif cmd == 9:                         # SYNCCP nb, b[nb], slot[nb], ids...
            ncp = data[0]
            bounds, sids, ids = data[1:1 + ncp], data[1 + ncp:1 + 2 * ncp], data[1 + 2 * ncp:]
            hist[:] = ids
            for b_, k_ in zip(bounds, sids):
                snaps[k_] = list(ids[:b_])
            reply(0, [len(hist), 0] + [1] * ncp)
        elif cmd == LOAD:
            if data[0] not in snaps:
                reply(-1)
                continue
            hist[:] = snaps[data[0]]
            reply(0, [len(hist)])
        elif cmd == DROP:
            snaps.pop(data[0], None)
            reply(0)
        elif cmd in (22, 23):                  # PERSIST slot, len, path / LOADFILE len, path (stand-in: JSON of the ids)
            w = data[1:] if cmd == 22 else data
            path = struct.pack(f"<{len(w) - 1}i", *w[1:])[:w[0]].decode()
            if cmd == 22:
                if data[0] not in snaps:
                    reply(-1)
                    continue
                with open(path, "w") as f:
                    json.dump(snaps[data[0]], f)
                reply(0, [1])
            else:
                try:
                    with open(path) as f:
                        hist[:] = json.load(f)
                except (OSError, ValueError):
                    reply(-2)
                    continue
                reply(0, [len(hist)])
        elif cmd == 24:                        # PERSIST_WAIT
            reply(0, [0])
        elif cmd == SELECT:
            if not 0 <= data[0] < len(hists):
                reply(-1)
                continue
            hist = hists[data[0]]
            reply(0, [len(hist)])
        elif cmd == STEPB:
            k = data[0]
            slots = data[1::2]
            if not 1 <= k <= 8 or len(data) != 1 + 2 * k or len(set(slots)) != k or \
                    any(not 0 <= x < len(hists) for x in slots):
                reply(-1)
                continue
            time.sleep(delay)              # one round for all (like the GPU)
            big, nb = max(big, k), nb + 1
            r = []
            for i in range(k):
                r += [0] + round_(hists[data[1 + 2 * i]], data[2 + 2 * i])
            reply(0, r + [0])
        elif cmd == 16:                        # USAGE reset: synthetic counters (2 layers x 4 experts)
            reply(0, [2, 4] + list(range(24)))
        elif cmd == BATCHES:
            reply(0, [big, nb])
        elif cmd == EVAL:
            hist.append(data[0])
            reply(0, [len(hist)])
        elif cmd == LOGITS:
            row = ((np.arange(vocab) * 37 + len(hist)) % 101).astype(np.float32) / 100.0  # noise < 1
            row[nxt()] = 5.0
            reply(0, raw=row.astype("<f4").tobytes())
        elif cmd == HIST:
            reply(0, hist)
        elif cmd == 12:                        # CORPUS_SET id, tokens... (stand-in: count only)
            corpus[data[0]] = len(data) - 1
            reply(0, [len(corpus), sum(corpus.values())])
        elif cmd == 13:                        # CORPUS_DROP id (-1 all)
            corpus.clear() if data[0] < 0 else corpus.pop(data[0], None)
            reply(0, [len(corpus), sum(corpus.values())])
        elif cmd == 25:                        # STATS: 13 counters (lo, hi); stand-in: rounds = number of STEP/STEPB
            v = [stats["cycles"], stats["cycles"] * 2, stats["cycles"], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            reply(0, [x for t in v for x in (t & 0xffffffff, t >> 32)])
        else:
            reply(-1)


if __name__ == "__main__":
    main()
