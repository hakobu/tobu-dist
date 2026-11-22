# Where every number in the README comes from

Machine for all runs: Bosgame M5 (Ryzen AI Max+ 395, Radeon 8060S / gfx1151, 128 GB). Model: unsloth
Qwen3.8-Flash-Next UD-IQ4_XS (revision pinned in `install.sh`). Bench kit: `bench/run_openai.py` with the prompts in
`bench/prompts/`.
- prefill tok/s = `prompt_tokens / ttft_s`;
- decode tok/s = `decode_tps`;
- output identity is checked by the field `sha256` (first 16 hex digits of the SHA-256 of the output text).

## Engine comparison (README "Numbers", 85 W)

| README value | file | how it is computed |
|---|---|---|
| tobu prefill 1,635 / 1,596 and decode 73.5 / 80.2 / 40.4 | `2026-11-23-mtp-heads/shipped-base-head/bench.json` (raw: `runs.jsonl`) | `summary.tobu.<id>.median`: median of 4 runs after a discarded warmup (`tobu bench`, tobu-dist server, 85 W) |
| llama.cpp prefill 562 / 481 | `2026-11-19-reference/llamacpp-vulkan-9a080f4-ub2048-nocache-20261119-165711.jsonl` | ids `ctx-8k`, `ctx-32k`: mean of runs 0 and 1 (562 / 563 and 485 / 478), truncated |
| llama.cpp default ubatch 416 / 365 | `2026-11-22-compare/llamacpp-vulkan-8af68d4-nocache-20261122.jsonl` | `ctx-8k`, `ctx-32k`, run 0 |
| halogen prefill 8.7k 1,599 | `2026-11-22-compare/halogen-0.20.7-gguf-mtp-nocache-20261122.jsonl` | `ctx-8k`, run 0: 8,704 / 5.445 s |
| halogen prefill 36k 1,680 | `2026-11-19-reference/halogen-0.20.7-gguf-mtp-nocache-20261119-154740.jsonl` | `ctx-32k`: mean of runs 0 and 1 (1,682 / 1,679), prompt cache off |
| llama.cpp and halogen decode code / JSON / prose | `2026-11-22-compare/*-nocache-20261122.jsonl` | ids `en-code`, `json-extract`, `de-prosa`: run 0 (the first, cold run; repeats hit prompt caches) |
| same output on repeat (3 / 7 / 7 of 7) | same files; tobu: `2026-11-23-mtp-heads/*/bench.json` | runs 0 and 1 per prompt have equal `sha256`; tobu: `summary.tobu.<id>.same_output` |
| summary of all of the above | `2026-11-22-compare/compare.json` | – |

## MTP heads (README second table)

| README value | file | how |
|---|---|---|
| no MTP: decode ~31, prefill 1,706 / 1,669 | `2026-11-23-mtp-heads/no-mtp/bench.json` | `summary.tobu.<id>.median`; same `sha256` as all other configurations |
| shipped unsloth head | `2026-11-23-mtp-heads/shipped-base-head/bench.json` | as above |
| head P | `2026-11-23-mtp-heads/head-p/bench.json` | as above |
| (internal fine-tuned production head, reference only) | `2026-11-23-mtp-heads/internal-finetuned-head/bench.json` | decode prompts only |
| "same 7 hashes" | all four `bench.json` | `summary.tobu.<id>.sha256` equals `published_sha256` (`bench/prompts/prompts.jsonl`) for every prompt |

## Power profiles (README third table)

| README value | file | how |
|---|---|---|
| 85 W: 1,635 / 1,596 | `2026-11-23-mtp-heads/shipped-base-head/bench.json` | as above (both kernels, released binary) |
| ~130 W: 1,934 / 1,796 | `2026-11-22-prefill/ab2-20261122-1714.txt` | median of the 4 measured `ON` runs (both kernels) |
| same hashes in both profiles | both files | `sha` column: `36c10ec667f4276a` (8.7k), `084a049545951ae3` (36k) |

## Other README statements

| statement | file | how |
|---|---|---|
| time to first token 0.7 s on a cache hit | `2026-11-19-reference/tobu-e3-cache-20261119-195115.jsonl` | `ctx-8k` / `ctx-32k` `ttft_s` 0.69–0.73 (earlier engine build, 19 November); cold for comparison: `tobu-e3-cold-20261119-195008.jsonl` |
| greedy output identical across power profiles and old/new kernels | `2026-11-22-prefill/*.txt` | one `sha` per prompt across all ON/OFF runs |
| gufo 0.34 reference runs (with and without MTP) | `2026-11-19-reference/gufo034-fni-*.jsonl` | – |
