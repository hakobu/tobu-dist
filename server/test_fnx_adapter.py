"""E2: engine wiring engine_flashnext.py against the mock fixtures/fake_fnx.py – without GPU, without model, without network.

    python test_fnx_adapter.py          (Python with numpy; tokenizers not needed)

The mock speaks the tobu-fnx protocol with a toy model (next token from the last two,
EOS roughly every 41st position) and emulates MTP: a STEP round commits 1..4 tokens. Checked:

  a. Pure greedy: generate(v, ids, N) == greedy sequence of the toy model (recomputed independently here),
     several prompts and N, with stop at EOS; also single step (use_self=False); MTP rounds really occur.
  b. Direct verify calls with arbitrary (partly wrong) drafts: output still bit-identical to the greedy sequence.
  c. Step (force_step before start): logits() right after start() works, argmax == first token; resample(t) + verify([]) feeds t in
     (token history of the mock) and greedy continues from there.
  d. logits()/resample() after the token stream has started -> FnxError.
  e. S2 (batched decode): 4 slots, 8 requests in threads; each delivers the greedy sequence, and the rounds
     really came batched (STEPB with >= 2 slots); with TOBU_FNX_BATCH=0 the same outputs without batching.
  g. Prefix store (grid 4): hits at grid points; in a long conversation the prefix entry before the first
     user message is kept although only the end snapshots are used (LRU refreshes all matching ones).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine_flashnext import FlashNextEngine, FnxError  # noqa: E402
from decoder import generate  # noqa: E402

FAKE = HERE / "fixtures" / "fake_fnx.py"
VOCAB = 1000
EOS = VOCAB - 1
HIST = 99                 # test command of the mock: token history
FAILS: list[str] = []


def check(ok: bool, msg: str, info=""):
    print(("ok   " if ok else "FAIL ") + msg + ((" – " + str(info)[:600]) if info and not ok else ""), flush=True)
    if not ok:
        FAILS.append(msg)


def toy(a: int, b: int) -> int:
    """Independent recomputation of the mock's rule."""
    return EOS if (3 * a + b) % 41 == 0 else (7 * a + 13 * b + 5) % (VOCAB - 1)


def ref(prompt: list[int], n: int, stop_eos: bool = True) -> list[int]:
    """Greedy sequence after the prompt, at most n tokens, with EOS as the last token (stop_eos)."""
    ctx, out = [0, 0] + list(prompt), []
    while len(out) < n:
        t = toy(ctx[-2], ctx[-1])
        out.append(t)
        ctx.append(t)
        if stop_eos and t == EOS:
            break
    return out


def prompts() -> list[list[int]]:
    r = random.Random(7)
    ps = [[5], [1, 2], [EOS - 1, 3, 4]] + [[r.randrange(VOCAB - 1) for _ in range(r.randrange(2, 40))] for _ in range(30)]
    return ps


def hist(fe) -> list[int]:
    return fe.proc.call(HIST)


# ---- a. pure greedy
def test_greedy(fe):
    eos_hit = mismatch = 0
    bad = None
    for ids in prompts():
        for n in (1, 2, 5, 17, 80):
            want = ref(ids, n)
            for use_self in (True, False):
                got = generate(fe.make(ids), ids, n, stop={fe.eos}, use_self=use_self).tokens
                if got != want:
                    mismatch += 1
                    bad = bad or (ids, n, use_self, got, want)
            eos_hit += want[-1] == EOS and len(want) < n
    check(mismatch == 0, "a greedy: Ausgabe == Greedy-Folge (33 Prompts × 5 N × MTP/Einzelschritt)", bad)
    check(eos_hit >= 5, f"a greedy: Stopp am EOS vor N kommt vor ({eos_hit} Fälle)", eos_hit)

    ids = next(p for p in prompts() if EOS not in ref(p, 60))
    v = fe.make(ids)
    r = generate(v, ids, 60, stop={fe.eos})
    multi = sum(1 for x in r.rounds if x.src == "m" and x.n_acc > 0)
    check(len(r.tokens) == 60 and v.stats["drafted"] > 0 and multi >= 5 and r.tok_per_round > 1.5,
          f"a greedy: MTP-Runden mit mehreren Tokens ({r.tok_per_round:.2f} Tokens/Runde, {multi} Engine-Entwürfe)",
          (v.stats, r.tok_per_round))
    check(hist(fe)[: len(ids)] == ids and hist(fe)[len(ids):len(ids) + len(r.tokens)] == r.tokens,
          "a greedy: Token-Verlauf der Engine = Prompt + Ausgabe")


# ---- b. foreign drafts
def test_drafts(fe):
    # directly: arbitrary drafts to verify, without engine draft
    mismatch, bad, acc = 0, None, 0
    for j, ids in enumerate(prompts()):
        r = random.Random(100 + j)
        want = ref(ids, 70)
        v = fe.make(ids)
        out = [v.start(ids)]
        while len(out) < 70 and out[-1] != EOS:
            k = len(out)
            c = r.random()
            if c < 0.3:
                d = want[k:k + r.randrange(1, 8)]
            elif c < 0.6:
                d = want[k:k + r.randrange(0, 4)] + [r.randrange(VOCAB)] + want[k + 5:k + 7]
            else:
                d = [r.randrange(VOCAB) for _ in range(r.randrange(0, 5))]
            m, nxt = v.verify(d)
            acc += m
            out += d[:m] + [nxt]
        cut = next((i + 1 for i, t in enumerate(out) if t == EOS), len(out))
        out = out[:min(cut, 70)]
        if out != want:
            mismatch += 1
            bad = bad or (ids, out, want)
    check(mismatch == 0 and acc > 0, f"b direkte verify-Aufrufe mit beliebigen Entwürfen: bitgleich ({acc} angenommen)",
          bad)


# ---- c. step
def test_step(fe):
    ids = next(p for p in prompts() if EOS not in ref(p, 30) and len(p) >= 3)
    v = fe.make(ids)
    v.force_step()                         # like the server for wrappers with logits (stream picks the 1st token itself)
    first = v.start(ids)
    lg = v.logits()
    check(lg.shape == (VOCAB,) and lg.dtype == np.float32 and int(np.argmax(lg)) == first == ref(ids, 1)[0],
          "c logits() direkt nach start(): Zeile vocab float32, argmax == erstes Token")
    check(hist(fe) == ids, "c nach start() + logits(): nichts eingespeist")

    t = (first + 123) % (VOCAB - 1)
    v.resample(t)
    m, nxt = v.verify([])
    check(m == 0 and hist(fe) == ids + [t], "c resample(t) + verify([]): t eingespeist (EVAL)", hist(fe)[len(ids):])
    want = ref(ids + [t], 12, stop_eos=False)
    out = [nxt]
    for _ in range(11):
        out.append(v.verify([])[1])
    check(out == want, "c danach greedy ab t weiter", (out, want))
    check(hist(fe) == ids + [t] + want[:-1], "c Token-Verlauf = Prompt + t + Ausgabe ohne das offene letzte Token")
    lg = v.logits()
    check(int(np.argmax(lg)) == want[-1], "c logits() im Schritt-Modus: argmax == zuletzt geliefertes Token")

    # step without resample (logits only): like greedy
    v = fe.make(ids)
    v.force_step()
    out = [v.start(ids)]
    v.logits()
    for _ in range(15):
        out.append(v.verify([7, 8])[1])     # drafts in step mode: are not accepted
    check(out == ref(ids, 16, stop_eos=False), "c Schritt ohne resample == Greedy-Folge")


# ---- d. switch in the middle of the stream
def test_switch(fe):
    ids = prompts()[3]
    for name, go in (("verify([])", lambda v: v.verify([])), ("self_draft()", lambda v: v.self_draft()),
                     ("verify(Entwurf)", lambda v: v.verify(ref(ids, 3)[1:]))):
        for call, f in (("logits()", lambda v: v.logits()), ("resample()", lambda v: v.resample(1))):
            v = fe.make(ids)
            v.start(ids)
            go(v)
            try:
                f(v)
                ok = False
            except FnxError:
                ok = True
            check(ok, f"d {call} nach {name} -> FnxError")
    # afterwards a new request works normally (RESET + SYNC)
    check(generate(fe.make(ids), ids, 20, stop={fe.eos}).tokens == ref(ids, 20),
          "d neue Anfrage nach dem Fehler: wieder greedy")


# ---- e. S2: parallel slots with batched decode
def test_batch():
    import os
    import threading

    for batch in ("1", "0"):
        os.environ["TOBU_FNX_BATCH"] = batch
        fe = FlashNextEngine.from_proc([sys.executable, str(FAKE), "--vocab", str(VOCAB), "--sessions", "4",
                                        "--delay-ms", "3"], max_len=4096, sessions=4)
        try:
            fe.enable_slots(4)
            ps = prompts()[3:11]
            got: list = [None] * len(ps)

            def run(i):
                try:
                    got[i] = generate(fe.make(ps[i]), ps[i], 120, stop={fe.eos}).tokens
                except Exception as e:  # noqa: BLE001
                    got[i] = e
                finally:
                    fe.leave()

            th = [threading.Thread(target=run, args=(i,)) for i in range(len(ps))]
            for t in th:
                t.start()
            for t in th:
                t.join()
            bad = [i for i in range(len(ps)) if got[i] != ref(ps[i], 120)]
            check(not bad, f"e Slots (Bündel {batch}): 8 parallele Anfragen = Greedy-Folge", [got[i] for i in bad])
            big, nb = fe.proc.call(98)
            if batch == "1":
                check(big >= 2 and fe.slot_info().get("batches"), "e STEPB gebündelt (>= 2 Slots je Runde)",
                      (big, nb, fe.slot_info()))
            else:
                check(nb == 0 and fe.batcher is None, "e TOBU_FNX_BATCH=0: kein STEPB", (big, nb))
        finally:
            fe.close()
    os.environ.pop("TOBU_FNX_BATCH", None)


# ---- i. fair prefills: chunks at grid points, progress, output unchanged
def test_chunks():
    import engine_flashnext as E

    old, old_alone = E.CHUNK, E.CHUNK_ALONE
    try:
        E.CHUNK, E.CHUNK_ALONE = 8, 0          # fixed chunks (the big chunk alone in the server is checked by the part below)
        ends = E.FlashNextEngine._chunk_ends(0, 30, [8, 17])
        check(ends == [12, 20, 28, 30] and all(e % E.GRID == 0 for e in ends[:-1]),
              "i Stückgrenzen: Raster, nie auf einem Checkpoint-Punkt", ends)
        check(E.FlashNextEngine._chunk_ends(4, 10, []) == [10], "i kurzer Rest: ein Stück")
        # chunk by chunk (adaptive size): the same boundaries as _chunk_ends with fixed size
        step, d = [], 0
        while d < 30:
            d = E.FlashNextEngine._next_end(d, 30, [8, 17], 8)
            step.append(d)
        check(step == ends, "i _next_end: Stück für Stück = _chunk_ends", (step, ends))
        check(E.FlashNextEngine._next_end(0, 30, [16], 16) == 20 and E.FlashNextEngine._next_end(20, 30, [], 16) == 30
              and E.FlashNextEngine._next_end(0, 30, [], 0) == 30, "i _next_end: Checkpoint-Punkt, Rest, aus")
        fe = FlashNextEngine.from_proc([sys.executable, str(FAKE), "--vocab", str(VOCAB)], max_len=4096)
        try:
            ids = prompts()[5] * 4
            seen = []
            v = fe.make(ids)
            v.progress = lambda d, t, eta: seen.append((d, t))
            got = generate(v, ids, 40, stop={fe.eos}).tokens
            fe.leave()
            check(got == ref(ids, 40), "i gestückelter Prefill: Ausgabe = Greedy-Folge")
            check(len(seen) >= 3 and seen[-1] == (len(ids), len(ids)) and
                  all(a[0] <= b[0] for a, b in zip(seen, seen[1:])), "i Fortschritt je Stück, monoton, endet bei total",
                  seen)
            v2 = fe.make(ids)
            got2 = generate(v2, ids, 40, stop={fe.eos}).tokens
            fe.leave()
            check(got2 == got and v2.reuse > 0, "i zweiter Lauf: Treffer im Prefix-Speicher, gleiche Ausgabe",
                  (v2.reuse, fe.store.info()))
            check(v.cache_src == "miss" and v2.cache_src == "ram", "i cache_src: kalt = miss, Treffer = ram",
                  (v.cache_src, v2.cache_src))
        finally:
            fe.close()
        # K4: alone in the server the big chunk (CHUNK_ALONE), with a second occupied slot CHUNK again
        fe = FlashNextEngine.from_proc([sys.executable, str(FAKE), "--vocab", str(VOCAB), "--sessions", "2"],
                                       max_len=4096, sessions=2)
        try:
            fe.enable_slots(2)
            E.CHUNK_ALONE = 32
            rnd = random.Random(11)
            ids = [rnd.randrange(VOCAB - 1) for _ in range(150)]
            runs = {}
            for name, busy in (("allein", False), ("belegt", True)):
                fe.store.clear()
                other = fe.free.get() if busy else None       # a second slot is occupied
                seen = []
                v = fe.make(ids)
                v.progress = lambda d, t, eta, seen=seen: seen.append(d)
                toks = generate(v, ids, 40, stop={fe.eos}).tokens
                fe.leave()
                if other is not None:
                    fe.free.put(other)
                runs[name] = (toks, [b - a for a, b in zip(seen, seen[1:]) if b > a])
            check(runs["allein"][0] == runs["belegt"][0] == ref(ids, 40), "i großes Stück: Ausgabe = Greedy-Folge")
            check(runs["allein"][1] and min(runs["allein"][1][:-1] or [32]) >= 32 and
                  max(runs["belegt"][1]) <= 8 + E.GRID and len(runs["allein"][1]) < len(runs["belegt"][1]),
                  "i allein große Stücke (CHUNK_ALONE), mit zweitem belegtem Slot kleine (CHUNK)", runs)
            E.CHUNK_ALONE = 0
            check(fe._chunk_now() == 8, "i CHUNK_ALONE = 0: immer CHUNK")
        finally:
            fe.close()
    finally:
        E.CHUNK, E.CHUNK_ALONE = old, old_alone


# ---- m. prefix store diagnostics (K12): longest common prefix with earlier prompts
def test_prompt_history():
    try:
        import server
    except Exception as e:                      # noqa: BLE001 – without the server dependencies (tokenizers) no test
        print(f"übersprungen m PromptHistory ({type(e).__name__})")
        return
    h = server.PromptHistory(keep=3)
    a = list(range(100))
    check(h.lcp(a) == (0, 0), "m erster Prompt: nichts gemeinsam")
    check(h.lcp(a + [7, 8]) == (100, 100), "m Fortsetzung: ganzer früherer Prompt gemeinsam")
    check(h.lcp(a[:40] + [999] + a[41:])[0] == 40, "m Abweichung an Stelle 40: Präfix 40")
    check(h.lcp([5, 5, 5]) == (0, 0), "m ganz anderer Prompt: nichts gemeinsam")
    for k in range(3):
        h.lcp([1000 + k] * 10)
    check(h.lcp(a)[0] == 0, "m nur die letzten keep Prompts bleiben (RAM begrenzt)")


# ---- l. sampling in the engine (mode A): the server leaves the engine drafts on -> rounds at full width
def test_sampled_width(fe):
    ids = prompts()[7]
    check(getattr(fe, "native_spec_sampling", False) is True, "l Engine meldet native_spec_sampling (MTP da)")
    out = {}
    for name, mtp in (("ohne", False), ("ein Entwurf", False), ("volle Breite", True)):
        fe.store.clear()
        v = fe.make(ids)
        v.use_mtp = mtp                       # this is how the server sets it: use_self = native_spec when sampling
        v.plain = name == "ohne"              # X-Tobu-Spec-Sampling: off / --no-spec-sampling
        v.config = {"thinking": False, "think_budget": -1, "temperature": 1.0, "top_p": 0.95, "top_k": 20}
        toks = generate(v, ids, 60, stop={fe.eos}, use_self=mtp).tokens
        d = v.draft_diff()
        fe.leave()
        out[name] = (toks, d["cycles"] if d else None)
    check(out["ohne"][0] == out["ein Entwurf"][0] == out["volle Breite"][0] == ref(ids, 60),
          "l gleiche Tokens ohne Entwürfe, mit einem Entwurf und in voller Breite", out)
    check(out["ohne"][1] > out["ein Entwurf"][1], "l ohne Entwürfe: eine Runde je Token", (out["ohne"][1], out["ein Entwurf"][1]))
    check(out["volle Breite"][1] is not None and out["volle Breite"][1] < out["ein Entwurf"][1],
          "l volle Breite: weniger Engine-Runden als mit höchstens einem Entwurf", (out["volle Breite"][1], out["ein Entwurf"][1]))


# ---- j. persistent prefix store (SSD): eviction -> file, restart -> hit from the SSD, broken file -> cold
def test_disk():
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        env = {"TOBU_FNX_DISK_DIR": d, "TOBU_FNX_DISK_MIN": "4", "TOBU_FNX_CACHE_MB": "1"}
        os.environ.update(env)
        try:
            cmd = [sys.executable, str(FAKE), "--vocab", str(VOCAB)]
            a, b = prompts()[6] * 3, prompts()[7] * 3
            fe = FlashNextEngine.from_proc(cmd, max_len=4096)
            try:
                fe.prefill(a, slot=0)
                fe.prefill(b, slot=0)            # budget 1 MiB: a's entries are evicted -> SSD
                n_hot = fe.persist_hot(5)
                info = fe.store.info()["disk"]
            finally:
                fe.close()
            mode = oct(os.stat(info["dir"]).st_mode & 0o777)
            check(info["entries"] >= 2 and mode == "0o700", "j Verdrängen + Stopp schreiben auf die SSD (Ordner 0700)",
                  (info, n_hot, mode))
            fe = FlashNextEngine.from_proc(cmd, max_len=4096)
            try:
                v = fe.make(a)
                got = generate(v, a, 30, stop={fe.eos}).tokens
                fe.leave()
                di = fe.store.info()["disk"]
                check(v.reuse > 0 and di["loads"] == 1 and got == ref(a, 30),
                      "j Neustart: Treffer von der SSD (LOADFILE), Ausgabe = Greedy-Folge", (v.reuse, di))
                for k, e in list(fe.store.disk.entries.items()):   # broken files: engine rejects -> cold, file gone
                    Path(fe.store.disk.path(k)).write_text("kaputt")
                fe.store.entries.clear()
                v = fe.make(b)
                got = generate(v, b, 30, stop={fe.eos}).tokens
                fe.leave()
                check(got == ref(b, 30) and len(fe.store.disk.entries) < di["entries"],
                      "j kaputte SSD-Datei: verworfen, kalt gerechnet, Ausgabe richtig", fe.store.info())
            finally:
                fe.close()
            os.environ["TOBU_FNX_CACHE_MB"] = "1"
            fe = FlashNextEngine.from_proc(cmd + ["--x", "anders"], max_len=4096)
            try:
                check(len(fe.store.disk.entries) == 0, "j anderer Startbefehl -> anderer Fingerabdruck, leer")
            finally:
                fe.close()
        finally:
            for k in env:
                os.environ.pop(k, None)


# ---- g. prefix store: grid, LRU
def test_store(fe):
    from engine_flashnext import GRID, PrefixStore

    st = PrefixStore(fe.proc, budget_mb=3)          # mock reports 1 MiB per snapshot -> at most 3 entries
    sysp = list(range(1, 41))                        # "system + tools"
    conv = list(sysp)
    for turn in range(8):                            # agent turns: prompt grows, two grid points per turn
        conv = conv + [100 + turn * 7 + i for i in range(7)]
        g, _slot = st.lookup(conv)
        pts = st.plan(conv, g, [len(sysp), len(conv) - 1])
        slots = [st.new_slot() for _ in pts]
        st.add(conv, pts, slots, [1] * len(pts))
    after = sysp + [999, 998, 997, 996]              # after compaction: only system + tools equal
    g, _ = st.lookup(after)
    check(GRID == 4 and g == (len(sysp) // GRID) * GRID,
          "g Präfix-Eintrag (System + Werkzeuge) überlebt 8 Züge bei 3 Einträgen Budget", (GRID, g, st.info()))


def test_state_b():
    """Fingerprint by state, draft counters (B), answer memory."""
    import os
    import tempfile

    from engine_flashnext import XSTORE_BASE, fingerprint

    with tempfile.TemporaryDirectory() as d:
        b1, b2 = os.path.join(d, "tobu-fnx"), os.path.join(d, "tobu-fnx-neu")
        model, mm = os.path.join(d, "m.gguf"), os.path.join(d, "mmproj.gguf")
        for f, x in ((b1, True), (b2, True), (model, False), (mm, False)):
            open(f, "w").write(f)
            os.chmod(f, 0o755 if x else 0o644)
        c1 = [b1, "--model", model, "--mmproj", mm, "--sessions", "4"]
        c2 = [b2, "--model", model, "--sessions", "4"]
        check(fingerprint(c1, "aa") == fingerprint(c2, "aa"),
              "j2 Fingerabdruck: andere Binärdatei und --mmproj ändern ihn bei gleichem Zustand nicht")
        check(fingerprint(c1, "aa") != fingerprint(c1, "bb"), "j2 Fingerabdruck: anderer Zustand -> anderer Schlüssel")
        check(fingerprint(c1) != fingerprint(c2), "j2 ohne Zustands-Kennung (alte Binärdatei): wie bisher über alles")
        f1 = fingerprint(c2, "aa")
        open(model, "a").write("xx")
        check(fingerprint(c2, "aa") != f1, "j2 Fingerabdruck: geändertes Modell -> anderer Schlüssel")
    cmd = [sys.executable, str(FAKE), "--vocab", str(VOCAB)]
    # corpus small and strict: slices of 30 tokens, clearing already from the first dead token (background sync)
    os.environ.update(TOBU_FNX_XSTORE_TOKENS="40", TOBU_FNX_CORPUS_SLICE="30", TOBU_FNX_CORPUS_STALE_MIN="0")
    fe = FlashNextEngine.from_proc(cmd, max_len=4096)
    try:
        check(fe.state_id == "04030201080706050c0b0a09100f0e0d", "j2 INFO: Zustands-Kennung gelesen", fe.state_id)
        v = fe.make([1, 2, 3])
        v.start(prompts()[0])
        for _ in range(3):
            fe.round(v.slot, 8)
        dd = v.draft_diff()
        check(dd is not None and dd["cycles"] == 3 and {"mtp", "history", "corpus", "store"} <= set(dd),
              "j2 draft_diff: Zähler je Quelle seit start()", dd)
        fe.leave()
        from engine_flashnext import CORPUS_DROP, CORPUS_SET

        def engine_docs():                       # mock answers SET with (documents, tokens)
            check(fe.corpus_idle(10), "j2 Korpus-Hintergrund fertig")
            n = fe.proc.call(CORPUS_SET, 999_999, *range(16))
            fe.proc.call(CORPUS_DROP, 999_999)
            return n[0] - 1

        def ws(names, size=20, salt=0):          # WorkspaceCache shape {rel: (mtime, size, ids_raw, ids_esc, text)}
            return {nm: (0, 0, [salt + k * 1000 + i for i in range(size)], [], "") for k, nm in enumerate(names)}

        for n in (20, 20, 20, 5):
            fe.remember_output(list(range(n)))
        q = list(getattr(fe, "_xstore", []))
        # 20+20+20 > 40: in the batch up to 3/4 (30) from the list -> only the youngest lives; dead > live -> clearing in the background
        check(len(q) == 1 and all(i >= XSTORE_BASE for i, _ in q) and fe._xstore_n == 20,
              "j2 Antwort-Speicher: Obergrenze, älteste im Schub raus, zu kurze nicht", (q, getattr(fe, "_xstore_n", None)))
        check(engine_docs() == 1 and fe._cstale == 0, "j2 Antwort-Speicher: Leeren im Hintergrund, die jüngste bleibt",
              (engine_docs(), fe._cstale))
        r = fe.sync_corpus({})
        check(engine_docs() == 1, "j2 Datei-Abgleich lässt den Antwort-Speicher stehen", r)
        a = fe.sync_corpus(ws(["a", "b", "c"]))
        check(a["pending"] and a["target_docs"] == 3, "j2 Korpus: sync_corpus kehrt sofort zurück (Hintergrund)", a)
        check(engine_docs() == 4 and len(fe._corpus) == 3 and fe._clive == 60, "j2 Korpus: neue Dateien in Scheiben",
              (engine_docs(), fe._corpus))
        one = ws(["a", "b", "c"]); one["b"] = (0, 0, [7] * 20, [], "")
        fe.sync_corpus(one)
        check(engine_docs() == 5 and fe._cstale == 20 and fe._clive == 60,
              "j2 Korpus: geänderte Datei ohne Drop (neue Id, alte zählt tot)", (engine_docs(), fe._cstale))
        fe.sync_corpus(ws(["x", "y", "z", "w"], salt=500))
        check(engine_docs() == 5 and len(fe._corpus) == 4 and fe._cstale == 0,
              "j2 Korpus: Sitzungswechsel -> tot > lebend -> einmal leeren, neu setzen, Antwort-Speicher bleibt",
              (engine_docs(), len(fe._corpus), fe._cstale))
        fe.sync_corpus(ws(["x", "y", "z", "w"], salt=500))
        check(fe.corpus_idle(5) and fe._cgen == 3, "j2 Korpus: gleicher Sollstand -> keine neue Runde", fe._cgen)
        big = ws([f"f{i}" for i in range(40)])
        fe.sync_corpus(big)
        fe.sync_corpus(ws(["x", "y", "z", "w"], salt=500))   # new target state in the middle of the sync
        check(fe.corpus_idle(10) and set(fe._corpus) == {"x", "y", "z", "w"},
              "j2 Korpus: neuer Sollstand während des Abgleichs gewinnt", sorted(fe._corpus)[:6])
    finally:
        for k in ("TOBU_FNX_XSTORE_TOKENS", "TOBU_FNX_CORPUS_SLICE", "TOBU_FNX_CORPUS_STALE_MIN"):
            os.environ.pop(k, None)
        fe.close()


if __name__ == "__main__":
    fe = FlashNextEngine.from_proc([sys.executable, str(FAKE), "--vocab", str(VOCAB)], max_len=4096)
    try:
        check(fe.vocab == VOCAB and fe.eos == EOS and fe.mtp is True and fe.max_len == 4096
              and fe.hf_tok is None and fe.name.endswith("+mtp"), "INFO: vocab, eos, MTP, max_len, Name")
        test_greedy(fe)
        test_drafts(fe)
        test_step(fe)
        test_switch(fe)
        test_store(fe)
        test_sampled_width(fe)
        v = fe.make([1, 2, 3])
        check(v.probe_config({"json_schema": {"type": "object"}}) is True and
              v.probe_config({"json_schema": {"type": "object", "x-reject": 1}}) is False,
              "h probe_config: Engine nimmt Schema an / lehnt ab (Status -3) -> True/False")
        fe.leave()
        from engine_flashnext import engine_schema
        sc = {"type": "object", "properties": {"a": {"anyOf": [{"enum": ["x", "y"]}, {"type": "null"}]},
                                               "b": {"enum": [1, 2]}, "c": {"enum": ["x", None]}}}
        es = engine_schema(sc)["properties"]
        check(es["a"]["anyOf"][0]["type"] == "string" and es["b"]["type"] == "integer" and
              es["c"]["type"] == ["null", "string"] and "type" not in sc["properties"]["b"],
              "h engine_schema: enum ohne type bekommt den Typ seiner Werte, Original unverändert", es)
    finally:
        fe.close()
    test_batch()
    test_chunks()
    test_disk()
    test_state_b()
    test_prompt_history()
    print("FEHLER:", ", ".join(FAILS) if FAILS else "keine")
    sys.exit(1 if FAILS else 0)
