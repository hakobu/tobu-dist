"""Thinking budget (TOB-32): end thinking after at most `budget` tokens.

Thinking models (Bonsai-2-27B / Qwen3.8-27B) without a limit often think until max_tokens and give no answer. If
thinking reaches the budget without writing </think> (or <tool_call>) itself, the server inserts the closing chunk per
the model's template itself ("\\n</think>\\n\\n"; if thinking already ends with a line break, without the first one)
and the model then generates the answer (max_tokens still applies to the whole output).

Implemented as the outermost wrapper around the chain's verifier (engine, FormatVerifier, SampledVerifier), interface as
decoder.generate needs it (start/verify, the rest via __getattr__):
  - Drafts are shortened so that thinking never runs past the budget (generate takes d.ids[:n_acc], the wrapper
    may only shorten the draft, not replace it); within the closing chunk only the part of the draft that matches it stays.
  - If the following token differs from the forced one, it is replaced: engine via resample(tok) (as with sampling or the
    grammar), the format grammar is reset to the state before the round and re-accepts the accepted draft and the
    forced token (json_schema: the grammar allows free text in the thinking block and then </think> + JSON;
    lazy tool grammar: idle during thinking), the sampler counts the forced token instead of the drawn one.
Without a budget (0/None) the wrapper is not built: output bit-identical to before.

Reserve for the answer (TOB-33): if the budget is so large that no room for the answer is left after thinking
(thinking_budget >= max_tokens), the output still ended without an answer. Effective is therefore
min(thinking_budget, max_tokens - reserve) with reserve = max(--thinking-answer-reserve (default 1024),
25 % of max_tokens). If max_tokens is smaller than the reserve, the effective budget is 0: no error, thinking is
ended immediately (output starts with "\n</think>\n\n", the way the template writes an empty thinking block).
"""

from __future__ import annotations

DEFAULT_ANSWER_RESERVE = 1024   # TOB-33: --thinking-answer-reserve
MIN_RESERVE_FRAC = 0.25         # TOB-33: reserve at least 25 % of max_tokens


def answer_reserve(max_tokens: int, reserve: int = DEFAULT_ANSWER_RESERVE) -> int:
    """TOB-33: tokens that stay free for the answer after thinking: max(reserve, ceil(25 % of max_tokens))."""
    mt = max(0, int(max_tokens))
    return max(max(0, int(reserve)), -(-mt * 25 // 100))


def effective_budget(budget: int, max_tokens: int, reserve: int = DEFAULT_ANSWER_RESERVE) -> tuple[int, bool, int]:
    """TOB-33: (effective budget, capped?, reserve) = (min(budget, max_tokens - reserve) >= 0, ...)."""
    r = answer_reserve(max_tokens, reserve)
    eff = max(0, min(int(budget), int(max_tokens) - r))
    return eff, eff < int(budget), r


class ThinkBudget:
    def __init__(self, v, budget: int, end_ids: set[int], close: list[int], close_nl: list[int], ends_nl,
                 fmt=None, sampler=None):
        """v: verifier (outermost wrapper so far); budget: at most this many thinking tokens (>= 0; 0 = end thinking
        immediately, TOB-33); end_ids: tokens that
        end thinking (</think>, <tool_call>); close: closing chunk ("\\n</think>\\n\\n"), close_nl: the same without
        leading line break; ends_nl(tok) -> True if the token ends with a line break; fmt: the chain's format
        grammar (or None); sampler: the chain's sampler (counts out[tok], or None)."""
        assert budget >= 0
        self.v, self.budget, self.end_ids = v, int(budget), set(end_ids)
        self.close, self.close_nl, self.ends_nl = list(close), list(close_nl), ends_nl
        self.fmt, self.sampler = fmt, sampler
        self.n = 0                      # thinking tokens so far (without the ending token)
        self.thinking = True            # prompt ends inside the thinking block
        self.pending: list[int] = []    # closing-chunk tokens still to be forced
        self.last = None                # last emitted token
        self.hit = False                # budget reached, closing chunk inserted
        self.n_replaced = 0               # tokens actually replaced (the rest came from the model itself or as draft)

    def __getattr__(self, k):
        if k in ("v", "fmt", "sampler", "pending"):
            raise AttributeError(k)
        return getattr(self.v, k)

    # --- grammar state (immutable except for acc)
    def _snap(self):
        f = self.fmt
        return None if f is None else (f.state(), f.acc)

    def _restore(self, snap, toks: list[int]):
        f = self.fmt
        (f.active, f.buf, f.stacks, f.partial, f.reasoning), f.acc = snap
        for t in toks:
            f.accept(t)

    def _note(self, toks: list[int]):
        """Count emitted tokens; budget reached -> schedule the closing chunk."""
        for t in toks:
            self.last = t
            if self.pending:
                if t == self.pending[0]:
                    self.pending.pop(0)
                continue
            if not self.thinking:
                continue
            if t in self.end_ids:
                self.thinking = False
                continue
            self.n += 1
            if self.n >= self.budget:
                self.thinking, self.hit = False, True
                self.pending = list(self.close_nl if self.ends_nl(t) else self.close)

    def _replace(self, old: int, new: int, snap, acc: list[int]):
        self.v.resample(new)
        if self.fmt is not None:
            self._restore(snap, acc + [new])
        if self.sampler is not None:
            self.sampler.out[old] -= 1
            self.sampler.out[new] += 1
        self.n_replaced += 1

    def start(self, prompt_ids):
        if self.pending:
            raise AssertionError("pending before start")
        if self.budget > 0:             # before the first token nothing is scheduled
            first = self.v.start(prompt_ids)
            self._note([first])
            return first
        # TOB-33: budget 0 (max_tokens smaller than the reserve): end thinking immediately, already the first token belongs
        # to the closing chunk (in full with "\n": the prompt ends with "<think>\n", the template writes empty thinking that way)
        self.thinking, self.hit = False, True
        want = list(self.close)
        snap = self._snap()
        first = self.v.start(prompt_ids)
        if first != want[0]:
            self._replace(first, want[0], snap, [])
            first = want[0]
        self.pending = want
        self._note([first])
        return first

    def verify(self, draft):
        draft = list(draft)
        if self.pending:                # closing chunk: only the matching start of the draft
            k = 0
            while k < len(draft) and k < len(self.pending) and draft[k] == self.pending[k]:
                k += 1
            draft = draft[:k]
        elif self.thinking:             # never think past the budget (the following token counts too)
            draft = draft[: max(0, self.budget - self.n - 1)]
        want = list(self.pending)
        snap = self._snap() if want else None
        m, nxt = self.v.verify(draft)
        if want:
            acc = draft[:m]             # = want[:m]
            if m < len(want) and nxt != want[m]:
                self._replace(nxt, want[m], snap, acc)
                nxt = want[m]
        self._note(draft[:m] + [nxt])
        return m, nxt

    def info(self) -> dict:
        return {"thinking_budget": self.budget, "thinking_budget_hit": self.hit, "thinking_forced": self.n_replaced}
