"""Shared decode loop for every engine: engine draft -> verify(draft) -> accepted tokens.

Engine interface (`Verifier`):
    start(prompt_ids) -> first token (prefill + first step)
    verify(draft)     -> (n_accepted, next token). The engine checks [last token, draft...] in one pass,
                         takes over the state up to the last accepted token and returns its greedy successor.
    self_draft()      -> optional: the engine's own draft (e.g. MTP head, copies from the history).
The output is always exactly the engine's sequence (every draft is verified), independent of the drafts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Round:
    n_draft: int
    n_acc: int
    src: str          # "m" (engine draft/MTP), "" (single step)
    t: float = 0.0    # seconds for verify (incl. synchronisation)
    head: str = ""    # which engine head supplied the draft (for src "m")


@dataclass
class Result:
    tokens: list = field(default_factory=list)
    rounds: list = field(default_factory=list)
    t_first: float = 0.0
    t_decode: float = 0.0
    t_propose: float = 0.0

    @property
    def tok_per_round(self):
        return (len(self.tokens) - 1) / max(1, len(self.rounds))


def generate(v, prompt_ids: list[int], n: int, stop: set[int] = frozenset(), use_self: bool = True,
             on_tokens=None) -> Result:
    """use_self: take the engine's drafts (v.self_draft), otherwise single steps.
    on_tokens(new tokens) after every round (streaming); if it returns True, generation ends there
    (stop string, client gone). The tokens up to that point are the same as without the callback."""
    r = Result()
    t0 = time.perf_counter()
    first = v.start(prompt_ids)
    r.t_first = time.perf_counter() - t0
    out = [first]
    t1 = time.perf_counter()
    done = first in stop
    if on_tokens is not None and on_tokens([first]):
        done = True
    while len(out) < n and not done:
        tp = time.perf_counter()
        ids, src = [], ""
        if use_self and hasattr(v, "self_draft"):
            ids = list(v.self_draft(field="free", kind="text"))[: max(0, n - len(out) - 1)]
            src = "m" if ids else ""
        r.t_propose += time.perf_counter() - tp
        tv = time.perf_counter()
        n_acc, nxt = v.verify(ids)
        dt = time.perf_counter() - tv
        new = ids[:n_acc] + [nxt]
        cut = next((i + 1 for i, t in enumerate(new) if t in stop), None)  # stop in the middle of the accepted draft
        if cut is not None:
            new, done = new[:cut], True
        r.rounds.append(Round(len(ids), n_acc, src, dt, getattr(v, "last_head", "") if src == "m" else ""))
        out += new
        if on_tokens is not None and on_tokens(new):
            done = True
    r.t_decode = time.perf_counter() - t1
    # truncate to n or up to the stop token
    cut = next((i + 1 for i, t in enumerate(out) if t in stop), len(out))
    r.tokens = out[: min(cut, n)]
    return r
