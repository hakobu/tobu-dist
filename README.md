# tobu – Qwen3.8-Flash-Next on AMD Strix Halo

**One model, one GPU, as fast as we can make it, and the same output every time.**

tobu (飛ぶ, “to fly”) is an inference engine built for exactly one model on exactly one GPU. This repository is the
`qwen38-flash-next-gfx1151` edition. It serves **Qwen3.8-Flash-Next** (MoE, unsloth UD-IQ4_XS GGUF) on an
**AMD Ryzen AI Max+ 395** (Strix Halo, gfx1151, 128 GB unified memory) through an OpenAI- and Anthropic-compatible
server. It has run in production for coding agents since 20 November 2026.

## Built on gufo

tobu exists because of **[Gufo](https://github.com/gufo-org/gufo)**, the open-source Strix Halo AI engine by
**Francesco Bozzo, Federico Izzo and the gufo contributors** ([gufo.blog](https://gufo.blog/docs), MIT license). They
wrote it and published it as open source, and large parts of tobu are built on that work. The compute core of this
edition, tobu-fnx, is Gufo's code:
- GGUF loading and weight upload;
- the Qwen3.8-Flash-Next model, with its HIP kernels for gfx1151;
- MTP drafting and the sampler.

The UD-IQ4_XS model support comes from
[flash-next-iq-gufo-extension](https://github.com/storm1er/flash-next-iq-gufo-extension) by Quentin Decaunes
(storm1er), also MIT. Thank you.

What tobu adds on top:
- **Agent server:** OpenAI and Anthropic APIs, streamed tool calls, grammar and `json_schema` output, parallel slots,
  prefix cache in RAM and on SSD, speculative sampling at full width, greedy mode per client.
- **Prefill kernels:** a paired IQ3_S/IQ4_XS gate/up kernel and an IQ4_NL down kernel that keeps the dequantised weights
  in registers. Both are bit-identical to the code paths they replace.
- **Bit-identity discipline:** every change is checked against tobu's own non-speculative greedy output (SHA-256 of the
  outputs), and the build is reproducible byte for byte.
- **`tobu bench`:** a one-command comparison with llama.cpp (and halogen).

[`THIRD_PARTY.md`](THIRD_PARTY.md) lists exactly which parts are tobu's own work and which come from upstream, file by
file, with credits. The licences and notices for everything taken over are there and in [`third_party/`](third_party/).

## Numbers

Same machine, same GGUF, batch 1, greedy, tokens/s. The tobu column is exactly what this repository ships
(`./tobu serve` with unsloth's MTP head), measured with `./tobu bench` on 23 November 2026.

**Bosgame M5 (Ryzen AI Max+ 395), default power profile (85 W):**

| task | llama.cpp (Vulkan) | halogen (commercial) | **tobu** |
|---|---:|---:|---:|
| cold prefill, 8.7k-token prompt | 562 | 1,599 | **1,635** |
| cold prefill, 36k-token prompt | 481 | 1,680 | **1,596** |
| decode: code | 28.5 | 67.5 | **73.5** |
| decode: JSON extraction | 28.5 | 79.8 | **80.2** |
| decode: German prose | 28.3 | 39.0 | **40.4** |
| same output on repeat (7 prompts) | 3 of 7 | 7 of 7 | **7 of 7** |

**MTP heads** (same run, tobu only, 85 W). Every configuration produced exactly the same 7 output hashes:

| tobu, decode tok/s | no MTP (plain) | unsloth MTP head (shipped) | [head P](https://huggingface.co/hakobuAi/tobu-mtp-head-p) (optional, `--head-p`, available) |
|---|---:|---:|---:|
| code / code refactor | 31.1 / 31.2 | 73.5 / 86.9 | 74.3 / 86.2 |
| JSON extraction | 31.1 | 80.2 | 83.7 |
| German prose / short German answer | 31.0 / 31.2 | 40.4 / 39.9 | 42.4 / 42.6 |
| cold prefill 8.7k / 36k | 1,706 / 1,669 | 1,635 / 1,596 | 1,637 / 1,601 |

Head P is an MTP head that tobu fine-tuned on public and synthetic data only. It is available on Hugging Face
([hakobuAi/tobu-mtp-head-p](https://huggingface.co/hakobuAi/tobu-mtp-head-p), Qwen Community License 1.0);
`./install.sh --head-p` downloads it (pinned revision, sha256-checked) and uses it instead of the unsloth head. Its output
is the same; only decode speed changes. Without MTP, prefill is about 4 %
faster, because with MTP the prefill also runs the MTP layer.

**Bosgame M5 (Ryzen AI Max+ 395), performance power profile (~130 W)**, next to the 85 W default:

| tobu, cold prefill | 85 W default | performance power profile (~130 W) |
|---|---:|---:|
| 8.7k-token prompt | 1,635 | **1,934** |
| 36k-token prompt | 1,596 | **1,796** |

Both columns are medians of 4 runs after a discarded warmup, with the same engine binary and both prefill kernels. The
~130 W runs are from 22 November; the GPU clock dropped during part of their 36k runs. The output hashes are the same in
both power profiles.

How these numbers were measured:
- tobu: `./tobu bench` through the tobu-dist server (answer memory off, prompt cache off): one discarded warmup, then 4
  back-to-back runs per prompt, median; GPU clock and power logged for every run (85.0 W during the runs).
- llama.cpp and halogen: the 22 November comparison run through each engine's own server, with thinking off, one cold
  run per prompt (their repeats hit prompt caches). So their figures are single runs, not medians.
- llama.cpp: decode with commit `8af68d4` (22 November), Vulkan backend, `-fa on`. Its prefill figures are from
  commit `9a080f4` with `-ub 2048` (19 November, mean of 2 runs); with the default ubatch it reaches 416 and 365 tok/s.
- halogen 0.20.7 with its MTP head. 8.7k prefill: one run on 22 November. 36k prefill: mean of 2 runs on 19 November
  with its prompt cache off (on the 22nd its prompt cache reused the 8.7k prefix).
- Four of the seven bench prompts are German, which matters for decode speed.
- Every number links to its raw data in [`results/README.md`](results/README.md).

### Honest limits

- **Long prompts:** at 85 W tobu and halogen are on par on a cold 8.7k-token prefill (within about 3 %; tobu's figure
  is a median of 4 runs, halogen's a single run), and halogen is about 5 % ahead on 36k. Most agent turns never pay a
  cold prefill, because tobu's prefix cache (RAM plus SSD) makes them warm: on a cache hit the time to first token was
  0.7 s for both prompts (measured 19 November with an earlier engine build).
- **Decode depends on the MTP head:** with the shipped unsloth head tobu is ahead of halogen on code and level on JSON
  and prose (halogen measured with a single run). A better-trained head adds up to about 7 % on prose; head P is
  available with `./install.sh --head-p`.
- **Hardware:** only gfx1151 (Strix Halo) with about 90 GB of GPU-visible memory. Batch 1 per slot (4 slots by default).

### Determinism

tobu's greedy output is identical to its own non-speculative (plain) greedy output, and bit-identical across:
- runs;
- restarts;
- speculation on or off (the MTP drafts; see the MTP-heads table: no MTP, two different heads, same 7 hashes);
- prefill chunk sizes;
- the new prefill kernels, which reproduce the old kernels' summation order exactly.

`tobu bench` checks that repeated runs agree and compares tobu's outputs with the published hashes in
`bench/prompts/prompts.jsonl`. Note that "identical" means identical to tobu's own plain path: llama.cpp and halogen
produce different (equally valid) greedy texts on some prompts, because their kernels sum in a different order. With
sampling (temperature >
0), speculation uses standard rejection sampling at full width, so the output distribution is exactly the model's. A
request with a seed is reproducible.

## Reproduce in 10 minutes

Prerequisites:
- A Strix Halo box with Linux and the amdgpu driver (`/dev/kfd`).
- Docker that your user can run.
- `python3` with the `venv` module (Ubuntu/Debian: `sudo apt install python3-venv`) or [uv](https://docs.astral.sh/uv/):
  the installer creates `.venv` with the packages in `server/requirements.txt` (numpy, tokenizers, jinja2; pillow for
  images). An existing Python with these packages can be given as `TOBU_PY`.
- About 95 GB of free disk for the model.
- The download itself (~87 GB) is not part of the 10 minutes. The weights are under the Qwen Community License 1.0
  (see [License](#license)).

One line (clones the repo into `./tobu-dist`, or `$TOBU_DIR`, and runs the installer from there):

```bash
curl -fsSL https://raw.githubusercontent.com/hakobu/tobu-dist/main/install.sh | bash
```

Or step by step:

```bash
git clone https://github.com/hakobu/tobu-dist && cd tobu-dist
./install.sh            # checks prerequisites, builds the ROCm image, gets binary + model (checksums), starts :8080
./tobu bench            # tobu: 7 fixed prompts, 1 warmup + 4 runs each, median, power + clock, SHA check (~9 min)
./tobu bench --quick    # 1 run per prompt (~3 min), same SHA check
# with llama.cpp for comparison (any llama-server build for this box):
./tobu serve stop && ./tobu bench --llama-server ~/llama.cpp/build/bin/llama-server
```

`tobu bench` starts and stops each engine itself, one after the other. Each engine needs most of the memory, so they
never run at the same time. It writes `results/bench-<time>/bench.json` (every run plus the medians) and `card.png`. If
you run halogen yourself, add `--halogen-url http://127.0.0.1:<port>`. halogen has no switch to turn its prompt cache
off, so the bench puts a unique first line on each prefill prompt for it.

The installer saves its choices (model folder and files, MTP head, container image) in `.tobu.env`; `./tobu serve`,
`./tobu bench` and a later `./install.sh` use them, and a variable set in the environment still wins. Options:
`--no-start`, `--build` (build the engine yourself), `--head-p` (optional MTP head P), `--help`.

Check `power cap` in the bench output (many kernels do not expose it; the bench then says so, and the measured package
power per run is in `bench.json`). The engine comparison above is at 85 W, the box's default. On a Bosgame M5
(Ryzen AI Max+ 395) with the performance power profile (~130 W), tobu's cold prefill measured 1,934 / 1,796 tok/s
(8.7k / 36k), compared with 1,635 / 1,596 at the 85 W default.

## Use it

```bash
./tobu serve start      # 127.0.0.1:8080 (TOBU_HOST / TOBU_PORT to change); ./tobu serve stop | status
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write merge_intervals in Python"}],"max_tokens":512}'
```

- OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, streaming, tool calls (arguments stream too),
  `json_schema` / grammar output, thinking with a token budget, images (optional mmproj file).
- 4 parallel slots; prefix store in RAM and on SSD (`TOBU_FNX_DISK_DIR`), so long agent sessions stay warm across
  restarts.
- Clients that send no temperature can be served greedy with `--greedy-harness <name>`. The header
  `X-Tobu-Sampling: model` opts a request back into the model card's sampling.
- Built-in chat page at `/chat`.
- PID file and log live in `~/.local/state/tobu` (`TOBU_STATE`), shared by every checkout of the same user. To run a
  second checkout next to the first, give it its own `TOBU_STATE` and `TOBU_PORT`.

## Build from source

```bash
docker build -t tobu-build:gfx1151 docker/
engine/build.sh                 # -> engine/build/tobu-fnx (10–20 min, CPU only)
```

The build is reproducible: the same image and the same sources give the same binary, byte for byte
(`engine/cmake/reproducible.cmake`). `./install.sh --build` builds instead of downloading the release binary.

## What's inside

| folder | what |
|---|---|
| `engine/` | tobu-fnx, the C++/HIP compute core for gfx1151 (MoE expert GEMMs, attention, GDN, MTP), derived from Gufo and flash-next-iq-gufo-extension |
| `server/` | the agent server (Python): API, slots, prefix store, speculation and sampling control, tool/grammar handling |
| `bench/` | `tobu bench` and the fixed prompt set |
| `scripts/` | container wrapper for the engine, server start/stop |
| `docker/` | build and runtime image (ROCm 10.1.0) |
| `results/` | the measurements behind the numbers above |

## How it was built

Developed with AI coding agents; every change is verified bit-for-bit against the plain greedy reference.

## Contact

David Müller, hakobu GmbH, david.mueller@hakobu.de

## License

Apache License 2.0, see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The engine contains code from Gufo, from
flash-next-iq-gufo-extension and from llama.cpp/ggml, all under the MIT license. Details and notices are in
[`THIRD_PARTY.md`](THIRD_PARTY.md) and [`third_party/`](third_party/). The model weights are not included. They are
downloaded from Hugging Face and are licensed under the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/LICENSE); the weights keep their
publishers' terms and are not bundled with tobu.

halogen is a commercial product. It is named here only as a point of comparison, and tobu contains none of
its code.
