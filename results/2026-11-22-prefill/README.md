# Prefill kernels, 22 November 2026 (Bosgame M5, Ryzen AI Max+ 395)

Setup:
- Machine: Bosgame M5 (Ryzen AI Max+ 395, Radeon 8060S / gfx1151, 128 GB).
- Model: Qwen3.8-Flash-Next UD-IQ4_XS, greedy decoding.

Method:
- One engine session, with one warmup run that is discarded.
- Then back-to-back runs. In the A/B files the kernel is switched on and off between runs, and the order alternates
  with each repetition.
- The tables report the median. GPU clock (sclk) and package power are logged for every run.

| file | what | power profile | result (median, cold prefill tok/s) |
|---|---|---|---|
| `ab-20261122-1653.*` | A/B test: step 1 (paired IQ gate/up kernel) on vs. off | **85 W default** | 8.7k: 1,639 vs. 1,536 · 36k: 1,590 vs. 1,492 |
| `ab2-20261122-1714.*` | A/B test: step 2 (IQ4_NL down kernel) on vs. off, with step 1 on | **performance power profile (~130 W)** | 8.7k: 1,934 vs. 1,920 · 36k: 1,796 vs. 1,803 (the GPU clock dropped during this run; pairs with matching clocks show +1.4 to 1.5 %) |
| `step1-130w-20261122-1702.{jsonl,txt}` | step 1, cold prefill + decode | **performance power profile (~130 W)** | 8.7k: 1,908 · 36k: 1,799 · decode code / JSON / prose: 78.5 / 87.3 / 43.9 tok/s |

The ~130 W numbers belong next to the 85 W default. At 85 W, the A/B test above had the first kernel only (1,639 / 1,590);
the released binary with both kernels measured 1,635 / 1,596 tok/s (8.7k / 36k) on 23 November
(`../2026-11-23-mtp-heads/shipped-base-head/`).
The output hashes (`sha256` for each run) are identical across all files and both power profiles.
