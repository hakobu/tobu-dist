# Third-party code and data in tobu-dist

tobu's own code is under the Apache License 2.0 (`LICENSE`, `NOTICE`). The parts below come from other projects and
keep their own licenses. All of them are MIT, so the combination can be distributed under Apache-2.0 as long as
their notices travel with the source and with the binary. The full upstream notice files are in `third_party/`.

## Original work in tobu (hakobu GmbH)

Everything listed here was written by hakobu GmbH and is under Apache-2.0. Everything else comes from the upstream
projects below, under their MIT licences. Each row names its origin: "ours" means designed and written by tobu;
"ours, known technique" means our own implementation of a published idea, which the credits column names.

### Whole files and directories that are ours

`server/` (except `json_schema_to_grammar.py`); `engine/tobu/`; `engine/tests/`; `engine/cmake/`;
`engine/CMakeLists.txt`; `engine/build.sh`; `engine/qwen_kernels.txt`; `bench/` (except the llama.cpp text in
`bench/prompts/ctx-*.txt`); `scripts/`; `docker/`; `install.sh`; `tobu`; `results/`; the documentation.
Our changes inside upstream files carry a `tobu` comment at the changed place.

### Kernels and engine (tobu-fnx)

| technique | where | origin | credit |
|---|---|---|---|
| Paired IQ3_S/IQ4_XS expert gate+up prefill kernel. One launch with a SwiGLU epilogue, instead of two passes with the gate in global memory. Bit-identical to the old route. | `engine/src/kernels/rocm/kernels.hip.cpp` `RoutedPairIqF16Kernel`; `executor.cpp` `PairedGateUp()`; `TOBU_FNX_IQ_PAIR` | ours; extends Gufo's paired Q4_K/Q5_K kernel to the IQ formats | idea from observing halogen (see below); layout after Gufo `RoutedPairF16Kernel` |
| Register-resident IQ4_NL expert down kernel. Weights are decoded in registers instead of the generic LDS path. Bit-identical. | `kernels.hip.cpp` IQ4_NL branch of `RoutedDownF16Kernel`, `RoutedDownWideF16Gemm`; `executor.cpp` `WideDownIq4Nl()`; `TOBU_FNX_IQ_DOWN` | ours; extends Gufo's register-resident Q5_1/Q8_0 down kernel | idea from observing halogen (see below) |
| Row-by-row bit-identity tests and benchmarks for both kernels, at every tile height | `engine/tests/*`, `engine/cmake/fnx_tests.cmake` | ours | – |
| Kernel route switchable at run time through a file, for interleaved A/B tests inside one engine session | `TOBU_FNX_IQ_PAIR_FILE`, `TOBU_FNX_IQ_DOWN_FILE` (`executor.cpp`) | ours | – |
| tobu-fnx: the engine process and its binary command protocol (SYNC, STEP, EVAL, LOGITS, SAVE/LOAD, CONFIG, PEEK, STEPB, IMAGE, PERSIST, SCORE, STATS …) | `engine/tobu/tobu_fnx.cpp` | ours | – |
| Per-request CONFIG: grammar, thinking budget, sampling and script ban. MTP stays on with grammar and with sampling. | `tobu_fnx.cpp` CONFIG; `server/engine_flashnext.py` `probe_config` | ours (wiring) | `JsonConstraint`, `ReasoningBudget`: Gufo |
| PEEK: the first token comes straight from prefill, which lowers time to first token | `tobu_fnx.cpp` PEEK; `engine_flashnext.py` | ours | – |
| Several sessions in one process. A server batcher bundles concurrent decode rounds (4 ms window) into one STEPB. | `tobu-fnx --sessions`; `engine_flashnext.py` `_StepBatcher`; `TOBU_FNX_BATCH_WAIT_MS` | ours | `DecodeBatch`/`EvaluateBatch`: Gufo; comparable to continuous batching (Orca, Yu et al., OSDI 2022) |
| Prefill chunk size as a setting; 16,384 measured +6–10 %, bit-identical | `engine/src/engine.cpp` `TOBU_FNX_PREFILL_CHUNK` | ours | – |
| External corpus as a second prompt-lookup source: copy drafts from documents the client supplies; every draft is verified | `engine/tobu/corpus.hpp`; `engine.hpp` `SetCorpus`; `TOBU_FNX_CORPUS_MIN` | ours, known technique, built on Gufo's prompt lookup | prompt lookup decoding (A. Saxena, 2023); REST (He et al., 2024) |
| Corpus index without rehash spikes: flat index, lazy deletion, compaction in slices, background sync | `corpus.hpp`; `engine_flashnext.py` `_corpus_worker` | ours | – |
| Draft counters per source (MTP, history, corpus, answer memory), reported per request in `usage.tobu` | `tobu_fnx.cpp` STATS; `engine.hpp` | ours | – |
| Script ban (e.g. no-CJK): a banned token is handled exactly like a grammar-forbidden token, drafts included | `engine/tobu/scripts.hpp`; `engine/gufo/src/core/sampling.cpp/.hpp`; server `--ban-scripts` | ours | – |
| Escape path for tool names that are in no tool list (no silent coercion) | `engine/gufo/src/core/json_constraint.cpp/.hpp` | ours (patch) | – |
| N-gram table (`per_layer_token_embd`) loadable from a separate GGUF | `engine.cpp`; `tobu-fnx --ple`; `TOBU_FNX_PLE` | ours (loader) | follows llama.cpp's `--ple` option |
| MTP draft gate and draft-vocabulary window as settings (our measurements confirmed Gufo's defaults) | `TOBU_FNX_MTP_GATE`, `TOBU_FNX_DRAFT_VOCAB_WINDOW` | ours (settings) | values: Gufo |
| Optional expert-usage counters per phase, layer and expert; outputs unchanged | `kernels.hip.cpp`, `executor.*`; `TOBU_FNX_EXPERT_USAGE` | ours (generic hook) | – |
| Hidden-state export for training MTP heads (prefill with the full MTP body, export of residual and head output) | `engine/tobu/teach.hpp`; TEACH command | ours (generic hook) | – |
| SCORE: per-position log p and argmax hit through the plain path, for perplexity | `tobu_fnx.cpp` SCORE | ours | – |
| state id over the compute-relevant sources, ignoring comments and whitespace; prefix caches survive rebuilds unless compute changes | `engine/cmake/state_id.cmake` | ours | – |
| Reproducible build: fixed prefix maps, fixed HIP CUID per source, no recorded command line. Same sources and image give the same binary byte for byte. | `engine/cmake/reproducible.cmake`; `install.sh` `BIN_SHA256` | ours | – |

### Server

| technique | where | origin | credit |
|---|---|---|---|
| OpenAI and Anthropic APIs from one server, streaming, llama.cpp-style `/props` | `server/api.py`, `server/server.py` | ours | behaviour modelled on llama.cpp `llama-server` |
| Prefix store on a snapshot grid: in-pass checkpoints every 4 tokens make a cache hit give exactly the cold result; LRU by token hash with refresh of matching prefixes | `engine_flashnext.py` `PrefixStore`; `TOBU_FNX_GRID`; SYNCCP | ours (store, grid exactness) | in-pass checkpoints and snapshot format: Gufo; prefix caching in general: SGLang RadixAttention (Zheng et al., 2024), vLLM |
| Prefix store on SSD: background writer, validity bound to the state id, hot entries flushed on stop | `DiskTier`, `persist_hot`; `TOBU_FNX_DISK_DIR`, `TOBU_FNX_DISK_MB` | ours | – |
| Fair prefill: prefill in grid-aligned chunks; between chunks the engine yields to decoding or waiting slots | `engine_flashnext.py` `CHUNK`, `yield_pipe`; `TOBU_FNX_PREFILL_YIELD_S` | ours | comparable to chunked prefill (Sarathi-Serve, Agrawal et al., OSDI 2024) |
| **K4**: one large prefill chunk (16,384 tokens) while a slot is alone, smaller chunks under concurrency | `CHUNK_ALONE`, `TOBU_FNX_PREFILL_CHUNK_ALONE` | ours | – |
| **Mode A**: speculative sampling with the engine's MTP drafts at full width for unseeded sampled requests; seeded requests stay reproducible | `server.py` `native_spec_sampling`; `--no-spec-sampling`, `X-Tobu-Spec-Sampling` | ours (server policy) | the rejection sampler is Gufo's; method: Leviathan et al. (ICML 2023), Chen et al. (2023) |
| Exact speculative sampling with point-mass drafts on the Python path; single-token grammar positions are not sampled | `server/sampling.py` | ours, known technique | Leviathan et al. 2023; Chen et al. 2023; penalty order as in vLLM |
| **Mode G**: greedy decoding for clients that send no temperature; `X-Tobu-Sampling: model` opts out | `--greedy-harness` | ours | – |
| Answer memory: finished answers become copy drafts for later requests (capped, evicted oldest first) | `engine_flashnext.py` `remember_output`; `TOBU_FNX_XSTORE_TOKENS` | ours, known technique | SuffixDecoding (Oliaro et al., 2024, arXiv 2411.04975); llama.cpp ngram cache |
| Lazy tool-call grammar: active only from the start of a tool call. It enforces tags, names, parameters and argument schemas; normal text stays plain greedy. | `server/tool_format.py` `Grammar`, `Matcher`, `FormatVerifier`; `--no-tool-grammar` | ours (Python implementation) | call format and grammar semantics follow llama.cpp (`qwen3_coder` chat format, `llama-grammar`; MIT, The ggml authors); `json_schema_to_grammar.py` is copied from llama.cpp |
| Forced grammar text proposed as a draft and verified by the model | `tool_format.py` `forced_text` | ours, variation of a known technique | jump-forward decoding (SGLang, 2024) |
| Streamed tool-call arguments (OpenAI `tool_calls` deltas, Anthropic `input_json_delta`) and deferred tools (`defer_loading`) | `server/api.py` | ours | – |
| Thinking budget with an answer reserve; `reasoning_effort` mapped to a budget | `server/think_budget.py`; `--default-thinking-budget`, `--thinking-answer-reserve` | ours | engine-side budget: Gufo `ReasoningBudget` |
| `presence_penalty` under greedy only when a request asks for it | `server/api.py` | ours | – |
| Chat template rendering and parsing in pure Python, byte-identical to the model's template | `server/chat.py`, `gguf_read.py`, `hftok.py` | ours (re-implementation) | template: the model's chat template, as shipped in llama.cpp `models/templates` |
| Image pipeline: decoding and URL fetch with SSRF guards; engine image cache; vision tower loaded on first use and freed when idle; prefix cache continues behind images | `server/media.py`; IMAGE/VPROMPT commands; `TOBU_FNX_IMAGE_MB`, `TOBU_FNX_VISION_IDLE_S` | ours (pipeline) | vision tower and resize: Gufo |
| Operations: `/health` returns 503 until warm; exit code 70 when the engine dies; `/tobu/metrics` and a request log with counts and hashes only; monitoring probes kept out of history; rewarm after idle; TCP_NODELAY; SSE keep-alive before the first token; 2 ms pipe polling | `server/server.py`; `--log`, `--rewarm-after` | ours | – |
| Hardening: binds 127.0.0.1; constant-time token check; with a token, `/chat` needs a one-time `?token=` login and then an HttpOnly, SameSite=Strict session cookie (same origin only, the token never appears in the page); token file created 0600; Host allowlist; CSP on `/chat`; engine container without network | `server/server.py`; `scripts/run-fnx.sh` | ours | – |

### Measurement and verification

| technique | where | origin |
|---|---|---|
| Bit-identity discipline: every speed-up is checked by the SHA-256 of greedy outputs across runs, restarts, MTP on/off, chunk sizes and kernel routes. A change that alters an output is not adopted. | `bench/tobu_bench.py`; `engine/tests/`; `tobu-fnx --ids` | ours (method; the MTP = plain property itself comes from Gufo) |
| Measurement method: one engine session, discarded warmup, interleaved A/B runs back to back, median of 4, GPU clock and power logged per run, caches off (or cache-busting for engines without a switch) | `bench/tobu_bench.py`, `bench/run_openai.py`, `results/` | ours |
| `tobu bench` and the engine-neutral bench kit, installer, build image | `bench/`, `install.sh`, `docker/`, `scripts/` | ours |

### Ideas from observing halogen

We observed the behaviour and timing of halogen, a commercial closed-source engine, and took two ideas from it:
- pairing expert gate and up for the IQ formats;
- keeping IQ4_NL expert-down weights in registers.

Both were implemented from scratch in Gufo's kernel style. No code was copied.

### Not ours, although tobu relies on it

- **Gufo / flash-next-iq-gufo-extension:** the model implementation, the attention/MoE/GDN kernels, MTP drafting
  with bit-exact verification, and target/draft rejection sampling with residual correction. The method behind the
  last two is from Leviathan et al. 2023 and Chen et al. 2023; MTP itself from Gloeckle et al. 2024 and DeepSeek-V3.
- **llama.cpp/ggml:** the MMQ kernels.

## engine/ (tobu-fnx, the compute core)

Taken from two MIT projects and trimmed to the parts this edition runs. Every taken file has a header line naming its
origin. tobu's own files are `engine/tobu/`, `engine/tests/`, `engine/cmake/` and `engine/CMakeLists.txt`. Changes by
tobu are marked in the files, for example the paired IQ3_S/IQ4_XS gate/up kernel and the register-resident IQ4_NL down
kernel in `src/kernels/rocm/kernels.hip.cpp`.

| path | origin | license |
|---|---|---|
| `engine/gufo/` | Gufo, https://github.com/gufo-org/gufo, tag v0.34.0 (`2845939b`): core headers and runtime pieces (GGUF reader, mapped files, host pages, weight upload, sampling) | MIT, Copyright (c) 2026 gufo contributors. `engine/LICENSE.gufo`; Gufo's NOTICE, THIRD_PARTY_NOTICES and license texts in `third_party/gufo/` |
| `engine/src/` (except `kernels/rocm/mmq/`) | flash-next-iq-gufo-extension, https://github.com/storm1er/flash-next-iq-gufo-extension, commit `9ce08ce` (v0.2.0), itself derived from Gufo's `qwen38_flash_next` model and Gufo's Qwen family libraries | MIT, Copyright (c) 2026 gufo contributors. `engine/LICENSE.flash_next_iq`, `third_party/flash-next-iq-gufo-extension/` |
| `engine/src/kernels/rocm/mmq/` | GGML MMQ/MMVQ kernels from llama.cpp/ggml, as vendored by the extension (`VENDOR.md`) | MIT, Copyright (c) 2023-2026 The ggml authors. `engine/LICENSE.llama.cpp` |


Clean-room note: some prefill techniques were chosen after observing the behaviour and timing of halogen, a
commercial engine. No code was copied from it. tobu implemented the ideas from scratch. Nothing is taken from AGPL projects.

## Release binary (`tobu-fnx`, gfx1151)

The binary is built from `engine/` in `docker/Dockerfile` (base image `rocm/dev-ubuntu-24.04:10.1.0-full`).
- It contains compiled code from the MIT parts above.
- It also contains header-only ROCm libraries compiled into it: hipCUB (BSD-3-Clause), rocPRIM (MIT) and rocWMMA (MIT).
  Their texts are in `third_party/gufo/licenses/{hipcub,rocprim,rocwmma}.txt`.
- At run time it links dynamically against ROCm (HIP, hipBLAS, hipBLASLt), ICU, OpenSSL (libcrypto) and libgomp. These
  come from the container image and are not part of the release.

## server/json_schema_to_grammar.py

Copied unchanged from llama.cpp, `examples/json_schema_to_grammar.py`, apart from a header comment. Used for
structured output and tool parameters. MIT License, Copyright (c) 2023-2026 The ggml authors (`engine/LICENSE.llama.cpp`).

## bench/prompts/ctx-8k.txt, ctx-32k.txt

These are the long-context benchmark inputs: the first 28,000 and 112,000 bytes of `ggml/src/ggml.c` from llama.cpp
commit `8af68d4`. They are used unchanged as prompt text. MIT License, Copyright (c) 2023-2026 The ggml authors.

## Model weights (not included)

`install.sh` downloads `unsloth/Qwen3.8-Flash-Next-GGUF` (revision `766911a6`, UD-IQ4_XS plus the shared Q8_0 MTP
head) from Hugging Face. The weights are not part of this repository. They are licensed under the
[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/LICENSE); the weights keep their publishers' terms and are not bundled with tobu. You accept
that license when you download them.
