"""Sampling with drafts (Python sampler, sampling.py), exact in distribution.

No GPU: toy model with 6 tokens, logits = fixed function of the last two tokens. Compared is the
distribution of whole sequences (3 tokens after the prompt, 216 cells) against the exactly computed distribution, once without
drafts, once with drafts of varying quality (correct, wrong, mixed, with stop token), via a chi-squared test
(p > 0.01), with penalties (repetition/presence/frequency), top_k/top_p/min_p and a format grammar (mask like the
fork, point mass without drawing). Plus: the same candidates via numpy and torch (bit-identical).

    python test_l38.py
    python test_l38.py --n 40000     # more draws
"""

from __future__ import annotations

import argparse
import os
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sampling  # noqa: E402

V, STOP = 6, 5
FAILS: list[str] = []


def check(name: str, ok: bool, info: str = ""):
    print(("ok   " if ok else "FAIL ") + name + (f"  ({info})" if info else ""))
    if not ok:
        FAILS.append(name)


def toy_logits(ctx: list[int]) -> np.ndarray:
    """Fixed, irregular logits per context (last two tokens), fp32 like the engine."""
    a, b = (ctx[-2] if len(ctx) > 1 else 7), (ctx[-1] if ctx else 3)
    r = np.random.default_rng(1000 + 31 * a + b)
    x = r.normal(0, 1.6, V).astype(np.float32)
    x[STOP] -= 1.5                                    # stop token less likely
    return x


class ToyLM:
    """Engine contract like engine_bonsai._Verifier: start/logits/resample/verify (without draft) and
    verify_rows/commit (with draft)."""

    def __init__(self):
        self.ctx: list[int] = []
        self.rows_calls = 0

    def start(self, prompt):
        self.ctx = list(prompt)
        return int(np.argmax(toy_logits(self.ctx)))

    def logits(self, row=None, dev=False):
        return toy_logits(self.ctx)

    def resample(self, tok):
        self.ctx.append(tok)

    def verify(self, draft):
        assert not draft

    def verify_rows(self, draft):
        self.rows_calls += 1
        self._d = list(draft)
        return np.stack([toy_logits(self.ctx + self._d[:i]) for i in range(len(draft) + 1)])

    def commit(self, m, tok):
        self.ctx += self._d[:m] + [tok]


class QLM(ToyLM):
    """6c: engine with a sampled head draft (q known per position)."""
    q_sampler = None
    last_self = None
    used_self = False


def toy_q(ctx: list[int]) -> np.ndarray:
    """Distribution of a "draft head": similar to, but not equal to the model (different randomness, flatter)."""
    x = 0.7 * toy_logits(ctx) + np.random.default_rng(77 + 13 * (ctx[-1] if ctx else 0)).normal(0, 0.8, V)
    e = np.exp(x - x.max())
    return e / e.sum()


class NoRows(ToyLM):
    verify_rows = None


class ToyFormat:
    """Format grammar for testing: token 0 never allowed; after token 2 exactly token 3 (point mass); stop allowed."""
    eog = {STOP}

    def __init__(self):
        self.last = None

    def allowed(self, t, state=None):
        if t == 0:
            return False
        return t == 3 if self.last == 2 else True

    def accept(self, t):
        assert self.allowed(t), (self.last, t)
        self.last = t

    def prefix(self, toks):
        last = self.last
        for i, t in enumerate(toks):
            if t == 0 or (last == 2 and t != 3):
                return i
            last = t
        return len(toks)

    def single(self):
        return 3 if self.last == 2 else None

    n_fixed, fixes = 0, []

    def best(self, logits):
        x = np.asarray(logits, dtype=np.float64)
        ok = [t for t in range(V) if self.allowed(t)]
        return max(ok, key=lambda t: (x[t], -t))


class GreedyLM(ToyLM):
    """Greedy engine for --draft-mask: verify_argmax/commit/logits(row) like engine_bonsai._Verifier."""

    def verify_argmax(self, draft):
        self._d = list(draft)
        self._rows = [toy_logits(self.ctx + self._d[:i]) for i in range(len(draft) + 1)]
        return [int(np.argmax(r)) for r in self._rows]

    def logits(self, row=None, dev=False):    # row of the last verification (after start: the prompt row)
        return self._rows[row or 0]

    def start(self, prompt):                  # greedy: first token is in the history, resample replaces it
        t = int(np.argmax(toy_logits(list(prompt))))
        self._rows = [toy_logits(list(prompt))]
        self.ctx = list(prompt) + [t]
        return t

    def resample(self, tok):
        self.ctx[-1] = tok


def test_draft_mask(N: int = 300):
    """Item 5: mask per draft position - output equals greedy + grammar without draft, more accepted."""
    import tool_format as tf

    prompt = [1, 2, 3, 4]
    same, acc_mask, acc_plain, extra = True, 0, 0, 0
    for k in range(N):
        rng = np.random.default_rng(k)
        p0 = prompt + [int(x) for x in rng.integers(0, V, 2)]
        ref, f = [], ToyFormat()          # reference: step by step, forbidden argmax -> allowed argmax
        c = list(p0)
        for _ in range(8):
            t = int(np.argmax(toy_logits(c)))
            if not f.allowed(t):
                t = f.best(toy_logits(c))
            f.accept(t)
            ref.append(t)
            c.append(t)
        for mask in (True, False):
            eng = GreedyLM()
            fv = tf.FormatVerifier(eng, ToyFormat(), mask=mask)
            if not mask:                  # without mask FormatVerifier needs verify() with commit
                def verify(d, eng=eng):
                    rows = eng.verify_argmax(d)
                    m = 0
                    while m < len(d) and d[m] == rows[m]:
                        m += 1
                    eng.ctx += d[:m] + [rows[m]]
                    return m, rows[m]
                eng.verify = verify
            out = [fv.start(p0)]
            while len(out) < 8:
                d = ref[len(out): len(out) + 4]           # perfect draft (greedy + grammar)
                m, x = fv.verify(d)
                out += d[:m] + [x]
                if mask:
                    acc_mask += m
                else:
                    acc_plain += m
            same &= out[:8] == ref
            if mask:
                extra += fv.n_mask_acc
    check("Maske je Entwurfsposition: Ausgabe == greedy + Grammatik (mit und ohne Maske)", same)
    check("Maske nimmt mehr Entwurfs-Tokens an", acc_mask > acc_plain, f"{acc_mask} gegen {acc_plain}, {extra} nur dank Maske")


def exact(params, prompt, n, grammar=False) -> dict:
    """Exact probability of every sequence of length n (stop ends the sequence early)."""
    res = {}

    def step_dist(prefix):
        s = sampling.Sampler(params, prompt)
        s.out = Counter(prefix)
        x = toy_logits(prompt + prefix)
        idx, pr = s.dist(*s.cands(x)[0])
        pu = dict(zip(idx.tolist(), pr.tolist()))
        if not grammar:
            return pu
        last = prefix[-1] if prefix else None
        ok = (lambda t: t == 3) if last == 2 else (lambda t: t != 0)
        bad = sum(p for t, p in pu.items() if not ok(t))
        out = {t: p for t, p in pu.items() if ok(t)}
        if bad > 0:                                    # like the fork: redraw over the allowed tokens
            xf = s.full(x)
            k = int(params.get("top_k") or 0)
            sel = [int(t) for t in np.argsort(-xf, kind="stable") if ok(int(t))][: (k if k > 0 else 1000)]
            sel = np.asarray(sel, dtype=np.int64)
            i2, p2 = s.dist(sel, xf[sel])
            for t, p in zip(i2.tolist(), p2.tolist()):
                out[t] = out.get(t, 0.0) + bad * p
        return out

    def rec(prefix, prob):
        if len(prefix) == n or (prefix and prefix[-1] == STOP):
            res[tuple(prefix)] = res.get(tuple(prefix), 0.0) + prob
            return
        for t, p in step_dist(prefix).items():
            if p > 0:
                rec(prefix + [t], prob * p)

    rec([], 1.0)
    return res


def run(params, prompt, n, N, drafter, grammar=False, spec=True, seed0=0) -> tuple[Counter, dict]:
    cnt, st = Counter(), Counter()
    seqs = st.setdefault("seqs", [])
    for k in range(N):
        eng = (QLM() if drafter == "q" else ToyLM()) if spec else NoRows()
        fmt = ToyFormat() if grammar else None
        sv = sampling.SampledVerifier(eng, sampling.Sampler(dict(params, seed=seed0 + k), prompt), fmt, spec=spec,
                                      stop={STOP}, q_mtp=drafter == "q")
        out = [sv.start(prompt)]
        rng = np.random.default_rng(10**6 + seed0 + k)
        while len(out) < n and out[-1] != STOP:
            if drafter == "q":                       # 6c: draw the draft from q, pass q per position along
                d, qd, c = [], [], prompt + out
                for _ in range(min(3, n - len(out) - 1)):
                    q = toy_q(c)
                    t = int(rng.choice(V, p=q))
                    d.append(t)
                    qd.append({i: float(q[i]) for i in range(V)})
                    c = c + [t]
                eng.last_self, eng.used_self = (d, qd), True
            else:
                d = drafter(prompt + out, rng, n - len(out) - 1) if spec else []
            m, x = sv.verify(d)
            new = d[:m] + [x]
            cut = next((i + 1 for i, t in enumerate(new) if t == STOP), None)
            out += new[:cut] if cut else new
            st["rounds"] += 1
            st["acc"] += m
            st["draft"] += len(d)
        cnt[tuple(out[:n])] += 1
        seqs.append(tuple(out[:n]))
        st["forced"] += sv.n_forced
    return cnt, dict(st)


def chi2_sf(x: float, df: int) -> float:
    """P(chi2_df > x) = Q(df/2, x/2), regularised incomplete gamma function (series or continued fraction)."""
    import math

    a, z = df / 2.0, x / 2.0
    if z <= 0:
        return 1.0
    lg = math.lgamma(a)
    if z < a + 1:                                  # series for P, then 1 - P
        s = t = 1.0 / a
        n = a
        for _ in range(10000):
            n += 1
            t *= z / n
            s += t
            if abs(t) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-z + a * math.log(z) - lg))
    b, c, d = z + 1 - a, 1e300, 1 / (z + 1 - a)    # continued fraction (Lentz) for Q
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return math.exp(-z + a * math.log(z) - lg) * h


def chi2(cnt: Counter, probs: dict, N: int) -> tuple[float, int, float]:
    """Chi-squared with small cells merged (expectation < 5) -> (chi2, degrees of freedom, p)."""
    obs, exp = [], []
    rest_o = rest_e = 0.0
    for k, p in probs.items():
        e = p * N
        if e < 5:
            rest_o += cnt.get(k, 0)
            rest_e += e
        else:
            obs.append(cnt.get(k, 0))
            exp.append(e)
    extra = sum(v for k, v in cnt.items() if k not in probs)
    if rest_e > 0:
        obs.append(rest_o + extra)
        exp.append(rest_e)
    elif extra:
        return float("inf"), 0, 0.0
    x2 = sum((o - e) ** 2 / e for o, e in zip(obs, exp))
    df = len(obs) - 1
    return x2, df, chi2_sf(x2, df)


# ------------------------------------------------------------------------------------------------ drafts
def d_greedy(ctx, rng, kmax):          # argmax chain (like a good MTP draft)
    out, c = [], list(ctx)
    for _ in range(min(4, kmax)):
        t = int(np.argmax(toy_logits(c)))
        out.append(t)
        c.append(t)
    return out


def d_wrong(ctx, rng, kmax):           # always the least likely token
    out, c = [], list(ctx)
    for _ in range(min(3, kmax)):
        t = int(np.argmin(toy_logits(c)))
        out.append(t)
        c.append(t)
    return out


def d_mixed(ctx, rng, kmax):           # length 0..4, random tokens (fixed before verification)
    return [int(t) for t in rng.integers(0, V, int(rng.integers(0, min(4, kmax) + 1)))]


def d_stop(ctx, rng, kmax):            # stop token in the draft
    return [int(np.argmax(toy_logits(ctx))), STOP, 1][: max(0, kmax)]


DRAFTERS = {"greedy": d_greedy, "wrong": d_wrong, "mixed": d_mixed, "stop": d_stop, "q": "q"}


def test_dist(N: int):
    prompt = [1, 2, 3, 4]
    cases = [
        ("Modellkarte Denken (T 1,0, top_p 0,95, top_k 20, min_p 0,05)",
         {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05}, False),
        ("Modellkarte ohne Denken (T 0,7, top_p 0,8, presence 1,5)",
         {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}, False),
        ("Strafen + top_k 4 (repetition 1,3, frequency 0,4, T 1,2)",
         {"temperature": 1.2, "top_p": 1.0, "top_k": 4, "min_p": 0.0, "repetition_penalty": 1.3,
          "frequency_penalty": 0.4, "presence_penalty": 0.2}, False),
        ("Formatgrammatik (Maske wie der Fork, Punktmaß)",
         {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}, True),
    ]
    for name, params, gram in cases:
        probs = exact(params, prompt, 3, gram)
        check(f"exakt: Summe 1 [{name}]", abs(sum(probs.values()) - 1) < 1e-9)
        c0, st0 = run(params, prompt, 3, N, None, gram, spec=False)
        x2, df, p = chi2(c0, probs, N)
        check(f"ohne Entwurf ~ exakt [{name}]", p > 0.01, f"χ²={x2:.1f}, df={df}, p={p:.3f}")
        for j, (dn, dr) in enumerate(DRAFTERS.items()):
            c1, st = run(params, prompt, 3, N, dr, gram, spec=True, seed0=10**7 * (j + 1))
            x2, df, p = chi2(c1, probs, N)
            acc = st["acc"] / max(1, st["draft"])
            check(f"mit Entwurf '{dn}' ~ exakt [{name}]", p > 0.01,
                  f"χ²={x2:.1f}, df={df}, p={p:.3f}, angenommen {acc:.0%}, ohne Ziehen {st['forced']}")
            if dr == "q":                          # 6c draws differently (acceptance + residual): only the distribution counts
                continue
            # same seed: exactly one draw per position from the same distribution -> same sequence as without draft
            _, st2 = run(params, prompt, 3, min(N, 2000), dr, gram, spec=True, seed0=0)
            check(f"gleicher Seed = gleiche Folge wie ohne Entwurf '{dn}' [{name}]",
                  st2["seqs"] == st0["seqs"][: len(st2["seqs"])])


def test_cands():
    """numpy and torch return the same candidates (ids and z bit-identical), with penalties and ties."""
    try:
        import torch
    except ImportError:
        print("skip torch")
        return
    rng = np.random.default_rng(5)
    Vb = 5000
    for params in ({"temperature": 0.7, "top_k": 20, "presence_penalty": 1.5, "repetition_penalty": 1.1,
                    "frequency_penalty": 0.3},
                   {"temperature": 1.0, "top_k": 20}, {"temperature": 0.7, "top_k": 20, "presence_penalty": 0.0},
                   {"temperature": 1.0, "top_k": 0}):
        rows = rng.normal(0, 3, (5, Vb)).astype(np.float32)
        rows[:, 100:140] = rows[0, 7]                  # tie beyond the fetched candidates
        rows[2, :50] = 9.0
        s = sampling.Sampler(dict(params, seed=1), list(range(0, 300, 7)))
        s.out = Counter({3: 2, 11: 1, 100: 1})
        extra = [5, 3, 100, 42]
        a = s.cands(rows, extra)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        b = s.cands(torch.from_numpy(rows).to(dev), extra)
        same = all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))
        check(f"Kandidaten numpy == torch ({dev}, {params})", same)
        one = [s.cands(rows[i], [])[0] for i in range(1)]
        check("eine Zeile == Zeile 0 mehrerer", np.array_equal(one[0][0], a[0][0]) and np.array_equal(one[0][1], a[0][1]))


def test_single():
    """ToolFormat.single: point mass only when exactly one token is allowed (real format grammar, Qwen vocabulary)."""
    tj = None
    for c in ([Path(os.environ["TOBU_TEST_TOKENIZER"])] if os.environ.get("TOBU_TEST_TOKENIZER") else []):
        if c.exists():
            tj = c
            break
    if tj is None:
        print("skip single (kein Tokenizer)")
        return
    from tokenizers import Tokenizer

    import tool_format as tf

    t = Tokenizer.from_file(str(tj))
    tb = tf.token_bytes(t)
    g = 'root ::= "{\\"relevant\\": " ("true" | "false") ", \\"x\\": \\"" [a-z]* "\\"}"'
    fmt = tf.ToolFormat.eager(g, tb, {t.token_to_id("<|im_end|>")})
    s0 = fmt.single()
    seen, forced = [], 0
    text = '{"relevant": true, "x": "ab"}'
    ids = t.encode(text, add_special_tokens=False).ids
    for i in ids:
        s = fmt.single()
        if s is not None:
            forced += 1
            check(f"single {tb.b[s]!r} ist das einzige erlaubte", s == i and
                  sum(1 for k in range(tb.n) if fmt.allowed(k)) == 1)
        seen.append(s)
        fmt.accept(i)
    check("single am Anfang: mehrere Zerlegungen von '{\"relevant' -> None", s0 is None)
    # ranked (mask while drawing) == loop over the whole sorted row
    fmt = tf.ToolFormat.eager(g, tb, {t.token_to_id("<|im_end|>")})
    rng = np.random.default_rng(3)
    same = True
    for i in ids[:6]:
        x = rng.normal(0, 3, tb.n + 5)
        want = 20
        brute = []
        for k in np.lexsort((np.arange(x.size), -x)):
            if fmt.allowed(int(k)):
                brute.append(int(k))
                if len(brute) >= want:
                    break
        same &= fmt.ranked(x, want) == brute
        fmt.accept(i)
    check("ranked == Schleife über die sortierte Zeile (6 Zustände)", same)
    print(f"     Punktmaße in {len(ids)} Tokens: {forced}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    test_cands()
    test_draft_mask()
    test_single()
    test_dist(a.n)
    print("FEHLER: " + ", ".join(FAILS) if FAILS else "alle Tests ok")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
