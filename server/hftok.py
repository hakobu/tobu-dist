"""Tokenizer adapter: encode/decode without special handling on decode (special tokens stay as text,
so the grammar sees "<tool_call>" etc.)."""

from __future__ import annotations


class HFTok:
    def __init__(self, tok):
        """tok: tokenizers.Tokenizer (tokenizer.json)."""
        self.t = tok

    @classmethod
    def from_file(cls, path):
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)))

    def encode(self, text: str) -> list[int]:
        return self.t.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.t.decode(list(ids), skip_special_tokens=False)
