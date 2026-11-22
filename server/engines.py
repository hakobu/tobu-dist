"""Test engines without GPU behind the Verifier (start/verify), see decoder.py.

OracleVerifier: "model" = fixed token sequence (greedy sequence of a real output). verify(draft) accepts the longest
prefix that matches the sequence and returns the next token of the sequence. This makes it possible to check without GPU:
output == sequence (bit-identical, independent of the drafts), tokens per round.
ScriptEngine: per request the next answer from a list (text), as an oracle – end-to-end test of the server.
KeyedScriptEngine, ToyEngine: see below.
"""

from __future__ import annotations

import json
from pathlib import Path


class OracleVerifier:
    def __init__(self, seq: list[int], eos: int):
        self.seq, self.eos = list(seq), eos
        self.pos = 0
        self.calls = 0          # checks (= rounds)
        self.checked = 0        # checked positions in total (1 + draft length per round)

    def _at(self, i: int) -> int:
        return self.seq[i] if i < len(self.seq) else self.eos

    def start(self, prompt_ids: list[int]) -> int:
        self.pos = 0
        return self._at(0)

    def verify(self, draft: list[int]) -> tuple[int, int]:
        self.calls += 1
        self.checked += 1 + len(draft)
        n = 0
        while n < len(draft) and draft[n] == self._at(self.pos + 1 + n):
            n += 1
        self.pos += n + 1
        return n, self._at(self.pos)


class ScriptEngine:
    """For the server: `make(prompt_ids)` -> Verifier with the next script answer (text, from "<think>\\n")."""

    def __init__(self, tok, script: str | Path, eos: int):
        self.tok, self.eos = tok, eos
        self.answers = json.loads(Path(script).read_text(encoding="utf-8"))
        self.i = 0
        self.name = "script"

    def make(self, prompt_ids: list[int]) -> OracleVerifier:
        text = self.answers[min(self.i, len(self.answers) - 1)]
        self.i += 1
        return OracleVerifier(self.tok.encode(text), self.eos)


class ToyEngine:
    """Deterministic toy "model" for dry runs without GPU: next token = the token
    after the last earlier occurrence of the last two tokens in the context (else a fixed token from the last two). verify computes every position
    like the single step -> output independent of the draft (like a bit-identical engine)."""

    def __init__(self, tokenizer_json: str, kmax: int = 8):
        from tokenizers import Tokenizer

        self.hf_tok = Tokenizer.from_file(tokenizer_json)
        self.eos = self.hf_tok.token_to_id("<|im_end|>")
        self.name, self.max_len = "toy", 1 << 20
        self.cur, self.seq = None, []

    def make(self, prompt_ids: list[int]):
        return _ToyVerifier(self)


class _ToyVerifier:
    def __init__(self, te: ToyEngine):
        self.te, self.ctx, self.last = te, [], {}

    def _next(self) -> int:
        key = tuple(self.ctx[-2:])
        i = self.last.get(key)
        if i is not None and i < len(self.ctx):
            return self.ctx[i]
        return 1000 + sum(self.ctx[-2:]) % 5000   # no earlier occurrence: fixed ordinary token (not EOS)

    def _push(self, t: int):
        if len(self.ctx) >= 2:
            self.last[tuple(self.ctx[-2:])] = len(self.ctx)
        self.ctx.append(t)

    def start(self, prompt_ids: list[int]) -> int:
        for t in prompt_ids:
            self._push(t)
        self.reuse = 0
        return self._next()

    def logits(self):
        import numpy as np

        x = np.zeros(self.te.hf_tok.get_vocab_size(), np.float32)
        x[self._next()] = 1.0
        return x

    def resample(self, tok: int):
        self.pend = tok                 # sampled token instead of the predicted one (sampling test)

    def verify(self, draft: list[int]):
        n = 0
        nxt = self._next() if getattr(self, "pend", None) is None else self.pend
        self.pend = None
        self._push(nxt)
        for d in draft:
            if d != self._next():
                break
            self._push(d)
            n += 1
        return n, self._next()


class KeyedScriptEngine:
    """Test engine of the server (L28): answer = text for the key whose marker "[[key]]" appears last in the prompt
    (else "default"). Pure function of the prompt -> the same request via OpenAI and Anthropic, streamed or
    not, must yield the same tokens. Checked as an oracle (verify), so bit-identical with drafts too."""

    def __init__(self, tok, answers: dict[str, str], eos: int, max_len: int = 1 << 20):
        self.tok, self.answers, self.eos, self.max_len = tok, answers, eos, max_len
        self.name = "keyed-script"

    def make(self, prompt_ids: list[int]) -> OracleVerifier:
        text = self.tok.decode(prompt_ids)
        best, at = "default", -1
        for k in self.answers:
            i = text.rfind(f"[[{k}]]")
            if i > at:
                best, at = k, i
        v = OracleVerifier(self.tok.encode(self.answers[best]), self.eos)
        v.reuse = 0
        return v
