"""Sampling for the tobu server.

Recommended values from the model card (Qwen3.8 / Bonsai 2) as default per mode, overridable by the client:
  Thinking on:  temperature 1.0, top_p 0.95, top_k 20, min_p 0.05, presence_penalty 0.0, repetition_penalty 1.0
  Thinking off: temperature 0.7, top_p 0.80, top_k 20, min_p 0.0,  presence_penalty 1.5, repetition_penalty 1.0
temperature = 0 -> greedy (measurements, bit-identity). Up to L38 the server turned speculation off at temperature > 0.
L38 (item 6a/b): drafts that are fixed before verification (lookup, grammar jump, MTP argmax)
stay on: per position a token is drawn from the full distribution p (after all transformations below); if the
drawn token matches the draft, it is accepted and we continue with the next position, otherwise the
drawn token is the output and the rest of the draft is dropped. This is speculative sampling with q = point mass on the
draft (accept with p(d), otherwise draw from p without d, renormalised) and therefore exact in distribution (same seed gives
not the same output as without drafts; what is required is the same distribution). If the format grammar allows exactly one
token, p is a point mass: then nothing is drawn at all (6a).

Order as in vLLM: repetition_penalty (prompt + output) -> presence/frequency_penalty (output only) ->
temperature -> top_k -> top_p -> min_p -> draw (numpy, seed optional). Logits arrive as fp32 rows from the engine
(numpy, or torch on the GPU). Penalties and temperature in fp64 (GPU and CPU compute the same IEEE operations),
top_k with a fixed order on ties (smaller token ID first), then softmax/top_p/min_p/draw on the CPU.
Without and with drafts the same computation runs, just over one or several rows.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

DEFAULTS = {
    True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05, "presence_penalty": 0.0,
           "repetition_penalty": 1.0},
    False: {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
            "repetition_penalty": 1.0},
}
KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty", "seed")


def resolve(body: dict, thinking: bool) -> dict | None:
    """Lay the request's parameters over the mode's defaults -> dict, or None for greedy."""
    p = dict(DEFAULTS[bool(thinking)])
    for k in KEYS:
        if body.get(k) is not None:
            p[k] = body[k]
    if float(p["temperature"]) <= 0.0:
        return None
    return p


PENALTY_KEYS = ("presence_penalty", "frequency_penalty", "repetition_penalty")


def greedy_penalties(body: dict) -> dict | None:
    """Penalties under greedy (temperature 0): only explicitly given values other than neutral, otherwise None.
    Default stays exact = argmax without penalty (code/agents: 0); Qwen recommends presence_penalty 1.5 against
    loops with thinking off. Meaning in flashnext (Gufo sampler): presence = once per token that already appeared in THIS
    answer (whole answer, no window); llama.cpp, in contrast, penalises only within the window of the last
    repeat_last_n (64) tokens. Without sampling support in the engine the value ends up in tobu.ignored."""
    if resolve(body, False) is not None and resolve(body, True) is not None:
        return None                       # sampling is used: penalties run there as well
    out = {}
    for k in PENALTY_KEYS:
        v = body.get(k)
        if v is not None and float(v) != (1.0 if k == "repetition_penalty" else 0.0):
            out[k] = float(v)
    return out or None


class Sampler:
    def __init__(self, p: dict, prompt_ids: list[int]):
        self.p = p
        self.rng = np.random.default_rng(p.get("seed"))
        self.prompt = np.unique(np.asarray(prompt_ids, dtype=np.int64)) if prompt_ids else np.zeros(0, np.int64)
        self.out: Counter = Counter()
        self._pm = None                 # prompt mask on the logits' device (repetition penalty)
        self.n_masked = 0               # draws with mask (drawn token did not fit the format grammar)

    # ------------------------------------------------------------------ one computation for all paths
    def _penal(self) -> bool:
        p = self.p
        return float(p.get("repetition_penalty") or 1.0) != 1.0 or bool(p.get("presence_penalty")) or \
            bool(p.get("frequency_penalty"))

    def _x_np(self, row, extra=()) -> np.ndarray:
        """One row (numpy) -> fp64 after penalties and temperature; extra = tokens that additionally count as output."""
        p = self.p
        x = np.asarray(row, dtype=np.float64).reshape(-1).copy()
        out = self.out + Counter(extra) if extra else self.out
        rp = float(p.get("repetition_penalty") or 1.0)
        if rp != 1.0:
            ids = np.union1d(self.prompt, np.fromiter(out, np.int64)) if out else self.prompt
            v = x[ids]
            x[ids] = np.where(v > 0, v / rp, v * rp)
        pres, freq = float(p.get("presence_penalty") or 0.0), float(p.get("frequency_penalty") or 0.0)
        if out and (pres or freq):
            ids = np.fromiter(out.keys(), np.int64)
            cnt = np.fromiter(out.values(), np.float64)
            x[ids] -= pres + freq * cnt
        x /= float(p["temperature"])
        return x

    def _x_torch(self, rows, extra=()):
        """Rows [R, V] (torch) -> fp64 [R, V] after penalties and temperature; row i counts extra[:i] as output."""
        import torch

        p = self.p
        x = rows.to(torch.float64)
        R, V = x.shape
        if self._penal():
            dev = x.device
            cnt = torch.zeros(R, V, dtype=torch.float64, device=dev)
            if self.out:
                ids = torch.tensor(list(self.out.keys()), dtype=torch.long, device=dev)
                cnt[:, ids] = torch.tensor(list(self.out.values()), dtype=torch.float64, device=dev)
            add = [(i, t) for i in range(1, R) for t in list(extra)[:i]]
            if add:
                ri = torch.tensor([a for a, _ in add], dtype=torch.long, device=dev)
                ti = torch.tensor([b for _, b in add], dtype=torch.long, device=dev)
                cnt.index_put_((ri, ti), torch.ones(len(add), dtype=torch.float64, device=dev), accumulate=True)
            rp = float(p.get("repetition_penalty") or 1.0)
            if rp != 1.0:
                if self._pm is None or self._pm.device != dev or self._pm.numel() != V:
                    self._pm = torch.zeros(V, dtype=torch.bool, device=dev)
                    if self.prompt.size:
                        self._pm[torch.from_numpy(self.prompt).to(dev)] = True
                inset = self._pm.unsqueeze(0) | (cnt > 0)
                x = torch.where(inset, torch.where(x > 0, x / _dt(rp, dev), x * rp), x)
            pres, freq = float(p.get("presence_penalty") or 0.0), float(p.get("frequency_penalty") or 0.0)
            if pres or freq:
                x = x - torch.where(cnt > 0, pres + freq * cnt, torch.zeros((), dtype=torch.float64, device=dev))
        return x / _dt(float(p["temperature"]), x.device)

    def cands(self, rows, extra=()) -> list:
        """Candidates per row: [(ids, z)] descending (ties: smaller ID first), at most top_k, z = logit after
        penalties/temperature (fp64). rows: [R, V] (torch or numpy) or [V]."""
        k = int(self.p.get("top_k") or 0)
        if not hasattr(rows, "is_cuda"):                       # numpy (test engines, MLX)
            rows = np.asarray(rows).reshape(-1, np.asarray(rows).shape[-1])
            return [_canon(self._x_np(rows[i], list(extra)[:i]), k) for i in range(rows.shape[0])]
        import torch

        rows = rows if rows.dim() == 2 else rows.unsqueeze(0)
        R, V = rows.shape
        fast = not self._penal() and 0 < k < V
        # without penalties: top_k directly on the fp32 logits (x / T is strictly monotonic and keeps distinct fp32 values
        # distinct), then the division in fp64 on the host – the same values as the general path
        x = rows if fast else self._x_torch(rows, extra)
        if not 0 < k < V:                                      # without top_k: whole rows (rare, slow)
            xs = x.cpu().numpy()
            return [_canon(xs[i], 0) for i in range(R)]
        kk = min(V, k + 32)
        val, idx = torch.topk(x, kk, dim=1)
        both = torch.cat([val.to(torch.float64), idx.to(torch.float64)], 1).cpu().numpy()   # one transfer
        res = []
        for i in range(R):
            v, ix = both[i, :kk], both[i, kk:].astype(np.int64)
            if fast:
                v = v / float(self.p["temperature"])
            o = np.lexsort((ix, -v))
            v, ix = v[o], ix[o]
            if kk < V and v[k - 1] == v[-1]:                  # tie extends beyond the fetched ones
                res.append(_canon(self._x_np(rows[i].cpu().numpy(), list(extra)[:i]) if fast
                                  else x[i].cpu().numpy(), k))
            else:
                res.append((ix[:k], v[:k]))
        return res

    def full(self, row):
        """Whole row after penalties/temperature (numpy fp64), for the format grammar's mask."""
        if hasattr(row, "is_cuda"):
            return self._x_torch(row.reshape(1, -1))[0].cpu().numpy()
        return self._x_np(row)

    def dist(self, idx, z):
        """Candidates (descending) -> (ids, p) after top_p and min_p."""
        p = self.p
        pr = np.exp(z - z[0])
        pr /= pr.sum()
        tp = float(p.get("top_p") or 1.0)
        if tp < 1.0:
            keep = int(np.searchsorted(np.cumsum(pr), tp) + 1)
            idx, pr = idx[:keep], pr[:keep]
        mp = float(p.get("min_p") or 0.0)
        if mp > 0.0:
            m = pr >= mp * pr[0]
            idx, pr = idx[m], pr[m]
        return idx, pr / pr.sum()

    def draw(self, idx, pr) -> int:
        return int(idx[self.rng.choice(len(idx), p=pr)])

    def _masked(self, row, allowed):
        """Chain only over allowed tokens (format grammar) -> (ids, p)."""
        self.n_masked += 1
        x = self.full(row)
        k = int(self.p.get("top_k") or 0)
        want = k if k > 0 else 1000
        rk = getattr(getattr(allowed, "__self__", None), "ranked", None)
        if rk is not None:
            sel = rk(x, want)
        else:
            sel = []
            for tk in np.lexsort((np.arange(x.size), -x)):
                if allowed(int(tk)):
                    sel.append(int(tk))
                    if len(sel) >= want:
                        break
        idx = np.asarray(sel, dtype=np.int64)
        return self.dist(idx, x[idx])

    def distribution(self, cand, row, allowed=None) -> dict:
        """The distribution pick() draws from, written out: {token: p} (6c needs p(x) for min(1, p/q) and the
        residual distribution). With format grammar as in the fork: p = Pu·[allowed] + Pu(not allowed)·P_masked."""
        idx, pr = self.dist(*cand)
        out = {int(t): float(x) for t, x in zip(idx, pr)}
        if allowed is None:
            return out
        bad = sum(x for t, x in out.items() if not allowed(t))
        out = {t: x for t, x in out.items() if allowed(t)}
        if bad > 0:
            i2, p2 = self._masked(row, allowed)
            for t, x in zip(i2.tolist(), p2.tolist()):
                out[int(t)] = out.get(int(t), 0.0) + bad * float(x)
        return out

    def pick(self, cand, row, allowed=None) -> int:
        """One token from the distribution (does not count it). allowed(tok) -> bool (format grammar): as in the fork, first
        draw without mask; if the token does not fit, run the chain (top_k/top_p/min_p) over allowed tokens only and draw again."""
        t = self.draw(*self.dist(*cand))
        if allowed is None or allowed(t):
            return t
        self.n_masked += 1
        x = self.full(row)
        k = int(self.p.get("top_k") or 0)
        want = k if k > 0 else 1000                 # allowed tokens in descending order (at most top_k or 1000)
        rk = getattr(getattr(allowed, "__self__", None), "ranked", None)
        if rk is not None:                          # format grammar: preselection by first byte, same order
            sel = rk(x, want)
        else:
            sel = []
            for tk in np.lexsort((np.arange(x.size), -x)):
                if allowed(int(tk)):
                    sel.append(int(tk))
                    if len(sel) >= want:
                        break
        idx = np.asarray(sel, dtype=np.int64)
        return self.draw(*self.dist(idx, x[idx]))

    def __call__(self, logits, allowed=None) -> int:
        """Without draft: draw one token from one row and count it as output."""
        t = self.pick(self.cands(logits)[0], logits, allowed)
        self.out[t] += 1
        return t


def _dt(v: float, dev):
    """Divisor as a tensor on the device: torch on CUDA divides by a Python scalar via reciprocal multiplication
    (a * (1/b)), numpy rounds a / b directly – with a tensor both compute the same IEEE division."""
    import torch

    return torch.tensor(v, dtype=torch.float64, device=dev)


def _canon(x: np.ndarray, k: int):
    """top_k of a row with fixed order (value descending, on ties smaller ID first)."""
    V = x.size
    if 0 < k < V:
        kk = min(V, k + 32)
        part = np.argpartition(-x, kk - 1)[:kk]
        o = np.lexsort((part, -x[part]))
        ix = part[o]
        if kk < V and x[ix[k - 1]] == x[ix[-1]]:
            ix = np.lexsort((np.arange(V), -x))
        ix = ix[:k]
    else:
        ix = np.lexsort((np.arange(V), -x))
    return ix.astype(np.int64), x[ix]


class SampledVerifier:
    """Wrapper around an engine's verifier: drawing instead of argmax.
    spec=False: without draft, one position per round (previous behaviour). The engine needs logits() and resample(tok).
    spec=True (L38, item 6a/b): drafts as with greedy; the engine needs verify_rows(draft) -> logits of all
    verification rows [T, V] and commit(m, tok) (commit m draft tokens + the drawn token). Per row a token is drawn from the full
    distribution as long as the drawn token equals the draft (exact, see module head)."""

    def __init__(self, v, sampler: Sampler, fmt=None, spec: bool = False, stop=frozenset(), q_mtp: bool = False):
        self.v, self.s, self.fmt = v, sampler, fmt
        self.n_q, self.n_q_acc = 0, 0     # 6c: draft tokens with distribution q, of which accepted
        rows = callable(getattr(v, "verify_rows", None))
        self.spec = bool(spec) and rows
        if self.spec and q_mtp and hasattr(v, "q_sampler"):   # 6c: draw the MTP draft from the head's distribution
            v.q_sampler = Sampler({k: x for k, x in sampler.p.items() if "penalty" not in k}, [])
            v.q_sampler.rng = sampler.rng
        self.dev = rows                 # verify via verify_rows/commit, logits as GPU rows (only top_k to the host)
        self.stop = set(stop) | (set(fmt.eog) if fmt is not None else set())
        self.n_forced = 0               # positions without drawing (grammar allows exactly one token, 6a)
        self.n_drawn = 0

    def __getattr__(self, k):           # self_draft, peek_self, last_head, t_catch ... of the engine
        if k in ("v", "s", "fmt", "spec", "stop", "dev", "n_q", "n_q_acc"):
            raise AttributeError(k)
        return getattr(self.v, k)

    @property
    def reuse(self):
        return getattr(self.v, "reuse", 0)

    @property
    def t_prefill(self):
        return getattr(self.v, "t_prefill", 0.0)

    def _one(self, get_cand, row) -> int:
        """Token for one position: grammar point mass without drawing, otherwise draw (with grammar like the fork)."""
        f = self.fmt
        if f is not None:
            t = f.single()
            if t is not None:
                self.n_forced += 1
                return t
        self.n_drawn += 1
        return self.s.pick(get_cand(), row, f.allowed if f is not None else None)

    def _take(self, t: int):
        self.s.out[t] += 1
        if self.fmt is not None:
            self.fmt.accept(t)

    def start(self, prompt_ids: list[int]) -> int:
        self.v.start(prompt_ids)
        row = self.v.logits(dev=True) if self.dev else self.v.logits()
        t = self._one(lambda: self.s.cands(row)[0], row)
        self._take(t)
        self.v.resample(t)
        return t

    def verify(self, draft: list[int]):
        if not self.spec and not self.dev:
            assert not draft, "sampling without draft"
            self.v.verify([])
            row = self.v.logits()
            t = self._one(lambda: self.s.cands(row)[0], row)
            self._take(t)
            self.v.resample(t)
            return 0, t
        if not self.spec:
            assert not draft, "sampling without draft"
        d = list(draft) if self.spec else []
        qs = None
        if d and getattr(self.v, "q_sampler", None) is not None and getattr(self.v, "used_self", False):
            ls = getattr(self.v, "last_self", None)              # draft comes from the head: q per position known
            if ls and list(ls[0][: len(d)]) == d:
                qs = ls[1][: len(d)]
        if hasattr(self.v, "used_self"):
            self.v.used_self = False
        if d and self.fmt is not None:
            k = self.fmt.prefix(d)
            # 6c: the first forbidden token stays in and is verified (p = 0 -> rejected, residual distribution); drawing
            # from p there without verification would shift the distribution
            d = d[: k + 1] if qs is not None else d[:k]
        cut = next((i + 1 for i, t in enumerate(d) if t in self.stop), None)
        if cut is not None:             # after a stop token verify nothing more
            d = d[:cut]
        rows = self.v.verify_rows(d)
        d = d[: rows.shape[0] - 1]
        cache, base = [], self.s.out.copy()

        def cand(i):
            if not cache:               # all rows at once: output up to before the round, row i additionally counts d[:i]
                cur, self.s.out = self.s.out, base
                try:
                    cache.extend(self.s.cands(rows, d))
                finally:
                    self.s.out = cur
            return cache[i]

        if qs is not None:
            return self._verify_q(rows, d, qs[: len(d)], cand)
        m = 0
        while True:
            t = self._one(lambda: cand(m), rows[m])
            self._take(t)
            if m < len(d) and t == d[m]:
                m += 1
                if t in self.stop:      # end inside the draft: the following token does not count (is cut off)
                    t = int(cand(m)[0][0])
                    break
                continue
            break
        self.v.commit(m, t)
        return m, t

    def _row_dist(self, i, rows, cand) -> dict:
        f = self.fmt
        if f is not None:
            t = f.single()
            if t is not None:
                self.n_forced += 1
                return {t: 1.0}
        self.n_drawn += 1
        return self.s.distribution(cand(i), rows[i], f.allowed if f is not None else None)

    def _verify_q(self, rows, d, qs, cand):
        """6c (Leviathan/Chen): draft d_i drawn from q_i; accept with min(1, p(d_i)/q_i(d_i)), otherwise draw from
        norm(max(0, p - q_i)) and stop; everything accepted -> bonus token from p. Exact in distribution."""
        rng = self.s.rng
        m = 0
        while True:
            p = self._row_dist(m, rows, cand)
            if m < len(d):
                x, q = d[m], qs[m]
                self.n_q += 1
                if rng.random() < min(1.0, p.get(x, 0.0) / q[x]):
                    self.n_q_acc += 1
                    self._take(x)
                    m += 1
                    if x in self.stop:
                        t = int(cand(m)[0][0])
                        break
                    continue
                res = {t: px - q.get(t, 0.0) for t, px in p.items() if px > q.get(t, 0.0)}
                tot = sum(res.values())
                src = res if tot > 0 else p
                tot = tot if tot > 0 else sum(p.values())
            else:
                src, tot = p, sum(p.values())
            toks = np.fromiter(src.keys(), np.int64)
            pr = np.fromiter(src.values(), np.float64) / tot
            o = np.argsort(toks)                                 # fixed order for drawing
            t = int(toks[o][rng.choice(len(toks), p=pr[o])])
            self._take(t)
            break
        self.v.commit(m, t)
        return m, t
