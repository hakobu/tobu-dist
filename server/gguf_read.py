"""Read GGUF without third-party code – head (metadata, tensor table) and tokenizer from the metadata.

Nothing from the file is executed: GGUF is a pure data format (head + tensors), the chat template is text.
Layout: "GGUF", version, number of tensors and keys; key/value pairs; tensor infos (name, shape ne[0..],
type, offset); data from general.alignment (default 32).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

QK_K = 256
# type ID -> (name, values per block, bytes per block); the IDs are those of ggml (ggml.h / gguf-py constants.py)
GGML = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
}
BY_NAME = {v[0]: k for k, v in GGML.items()}
_KV = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class GGUFError(ValueError):
    """File is not a (complete) GGUF."""


class GGUFTypeError(ValueError):
    """Tensor format without an unpack formula (and without its own kernel): clear error instead of wrong numbers."""


@dataclass
class TensorInfo:
    name: str
    ne: tuple[int, ...]           # GGUF order: ne[0] = columns (fastest axis)
    type_id: int
    offset: int                   # relative to the start of the data

    @property
    def type_name(self) -> str:
        return GGML.get(self.type_id, (f"type{self.type_id}",))[0]

    @property
    def shape(self) -> tuple[int, ...]:
        """torch order: [N, K] for matrices."""
        return tuple(reversed(self.ne))

    @property
    def n(self) -> int:
        out = 1
        for x in self.ne:
            out *= int(x)
        return out

    @property
    def nbytes(self) -> int:
        if self.type_id not in GGML:
            raise GGUFTypeError(f"{self.name}: unknown ggml type id {self.type_id}")
        _, blk, nb = GGML[self.type_id]
        if self.ne[0] % blk:
            raise GGUFError(f"{self.name}: row length {self.ne[0]} not a multiple of the block size {blk}")
        return self.n // blk * nb

    @property
    def row_bytes(self) -> int:
        _, blk, nb = GGML[self.type_id]
        return self.ne[0] // blk * nb


@dataclass
class Header:
    path: Path | None
    version: int
    meta: dict
    tensors: dict[str, TensorInfo]
    data_start: int
    header_bytes: int
    lazy: dict = field(default_factory=dict)     # large lists (tokens/merges) only on demand


def _rd(f, fmt):
    b = f.read(struct.calcsize(fmt))
    if len(b) != struct.calcsize(fmt):
        raise GGUFError("file ends inside the GGUF header (truncated download?)")
    return struct.unpack(fmt, b)[0]


def _str(f) -> str:
    n = _rd(f, "<Q")
    b = f.read(n)
    if len(b) != n:
        raise GGUFError("file ends inside the GGUF header (truncated download?)")
    return b.decode("utf-8", "replace")


def _val(f, t: int, lazy_from: int | None = None):
    """Read a value. Lists with more than lazy_from entries are skipped and recorded as ("lazy", pos, typ, n)."""
    if t == 8:
        return _str(f)
    if t == 9:
        et, n = _rd(f, "<I"), _rd(f, "<Q")
        if lazy_from is not None and n > lazy_from:
            pos = f.tell()
            if et in _KV:
                f.seek(struct.calcsize(_KV[et]) * n, 1)
            elif et == 8:
                for _ in range(n):
                    f.seek(_rd(f, "<Q"), 1)
            else:
                for _ in range(n):
                    _val(f, et)
            return ("lazy", pos, et, n)
        if et in _KV:
            sz = struct.calcsize(_KV[et])
            b = f.read(sz * n)
            if len(b) != sz * n:
                raise GGUFError("file ends inside the GGUF header (truncated download?)")
            return np.frombuffer(b, dtype=np.dtype(_KV[et])).tolist()
        return [_val(f, et) for _ in range(n)]
    if t not in _KV:
        raise GGUFError(f"unknown GGUF value type {t}")
    return _rd(f, _KV[t])


def read_header(path, lazy_lists: int | None = 4096) -> Header:
    """Head of a GGUF file (also of a truncated beginning: head only, no data needed).
    lazy_lists: read lists longer than this (tokens, merges) only on load_list(); None = everything at once."""
    p = Path(path)
    with open(p, "rb") as f:
        if f.read(4) != b"GGUF":
            raise GGUFError(f"{p.name}: not a GGUF file")
        ver = _rd(f, "<I")
        if ver not in (2, 3):
            raise GGUFError(f"{p.name}: GGUF version {ver} not supported (2, 3)")
        n_t, n_kv = _rd(f, "<Q"), _rd(f, "<Q")
        meta, lazy = {}, {}
        for _ in range(n_kv):
            k = _str(f)
            v = _val(f, _rd(f, "<I"), lazy_lists)
            if isinstance(v, tuple) and v and v[0] == "lazy":
                lazy[k] = v
                v = None
            meta[k] = v
        tensors = {}
        for _ in range(n_t):
            name = _str(f)
            nd = _rd(f, "<I")
            ne = tuple(int(_rd(f, "<Q")) for _ in range(nd))
            tid, off = _rd(f, "<I"), _rd(f, "<Q")
            if name in tensors:
                raise GGUFError(f"{p.name}: tensor {name} twice")
            tensors[name] = TensorInfo(name, ne, tid, off)
        hb = f.tell()
    align = int(meta.get("general.alignment") or 32)
    return Header(p, ver, meta, tensors, (hb + align - 1) // align * align, hb, lazy)


def load_list(h: Header, key: str) -> list | None:
    """Load a deferred list (e.g. tokenizer.ggml.tokens)."""
    if key not in h.lazy:
        return h.meta.get(key)
    _, pos, et, n = h.lazy[key]
    with open(h.path, "rb") as f:
        f.seek(pos)
        if et in _KV:
            sz = struct.calcsize(_KV[et])
            out = np.frombuffer(f.read(sz * n), dtype=np.dtype(_KV[et])).tolist()
        else:
            out = [_val(f, et) for _ in range(n)]
    h.meta[key] = out
    del h.lazy[key]
    return out


# --------------------------------------------------------------------------------------------- tokenizer from the GGUF

# tokenizer.ggml.pre -> pre-tokenization. NOT stored in the GGUF (only the name): table as in llama.cpp (llama-vocab.cpp)
# or tokenizer.json of the base model. Unknown name = clear error (otherwise silently different tokens).
PRE = {
    "qwen35": {"regex": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
                        r"|\s*[\r\n]+|\s+(?!\S)|\s+", "nfc": True},
    "qwen2": {"regex": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+"
                       r"|\s+(?!\S)|\s+", "nfc": True},
}
TOKEN_NORMAL, TOKEN_CONTROL, TOKEN_USER = 1, 3, 4


def tokenizer_json(h: Header) -> dict:
    """tokenizer.json (HF schema) from the GGUF lists: BPE (byte-level, gpt2) + special tokens with their IDs."""
    model = h.meta.get("tokenizer.ggml.model")
    pre = h.meta.get("tokenizer.ggml.pre")
    if model != "gpt2":
        raise GGUFError(f"tokenizer model {model!r}: only byte-level BPE (gpt2) is supported")
    if pre not in PRE:
        raise GGUFError(f"tokenizer pre-type {pre!r}: unknown (no splitting rule) - add it to gguf_read.PRE")
    tokens = load_list(h, "tokenizer.ggml.tokens")
    types = load_list(h, "tokenizer.ggml.token_type")
    merges = load_list(h, "tokenizer.ggml.merges")
    if not tokens or not merges or not types or len(types) != len(tokens):
        raise GGUFError("tokenizer lists missing or inconsistent in the GGUF")
    vocab, added = {}, []
    for i, (t, ty) in enumerate(zip(tokens, types)):
        if ty == TOKEN_NORMAL:
            vocab[t] = i
        elif ty in (TOKEN_CONTROL, TOKEN_USER):
            added.append({"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": ty == TOKEN_CONTROL})
    if len(vocab) != max(vocab.values()) + 1:
        raise GGUFError("tokenizer: normal tokens are not a contiguous id range")
    p = PRE[pre]
    bl = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
    return {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
            "normalizer": {"type": "NFC"} if p["nfc"] else None,
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": p["regex"]}, "behavior": "Isolated", "invert": False}, bl]},
            "post_processor": bl, "decoder": bl,
            "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                      "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                      "vocab": vocab, "merges": [m.split(" ", 1) for m in merges]}}


def tokenizer(h: Header):
    from tokenizers import Tokenizer

    return Tokenizer.from_str(json.dumps(tokenizer_json(h), ensure_ascii=False))
