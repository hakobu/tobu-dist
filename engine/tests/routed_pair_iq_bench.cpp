// Times the routed expert gate/up projection of tobu-fnx's wide prefill per
// layer, paired (RoutedGatedF16Gemm, 128-token map, one launch) against the
// two-pass route (RoutedF16Gemm gate -> F32, then up with SwiGLU, on the
// executor's 16/48-token map), alternating the two so clock drift hits both.
// Prints per layer: tokens, active experts, tile counts, median ms and
// TFLOPS of both, the speed-up and whether both outputs hash the same; then
// the totals over all layers.
//
// Layers come from a counts file (one line of 512 per-expert counts per
// layer; tokens = sum / 10), e.g. phase 0 of Engine::ExpertUsage after a
// single prefill pass with TOBU_FNX_EXPERT_USAGE=1, or are synthetic: per
// layer a Zipf-like top-10 routing with weights 1 / (rank + s), s varying
// per layer between 4 (hot experts) and 64 (flat), over a random expert
// permutation (gufo measured ~66 % of the 512 experts active at pp2048).
//
// Ported from gufo-tobu tests/models/qwen38_flash_next/routed_gemm_bench.cpp.
//
// tobu_fnx_routed_pair_iq_bench [reps] [tokens | counts-file] [type] [layers]
//   reps 10, tokens 4096, type iq3_s (| iq4_xs | q4_k | q5_k), layers 40
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/rocm/kernels.hpp"

namespace {

namespace q = gufo::models::flash_next_iq::rocm;

constexpr std::uint32_t kExperts = 512;
constexpr std::uint32_t kUsed = 10;
constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kFf = 640;
q::WeightType gate_type = q::WeightType::kIQ3_S;

std::size_t BlockBytes(q::WeightType type) {
  return type == q::WeightType::kQ4_K    ? 144
         : type == q::WeightType::kQ5_K  ? 176
         : type == q::WeightType::kIQ3_S ? 110
                                         : 136;
}
q::WeightType ParseType(const std::string& name) {
  if (name == "iq3_s")
    return q::WeightType::kIQ3_S;
  if (name == "iq4_xs")
    return q::WeightType::kIQ4_XS;
  if (name == "q4_k")
    return q::WeightType::kQ4_K;
  if (name == "q5_k")
    return q::WeightType::kQ5_K;
  throw std::runtime_error("unknown gate/up type " + name);
}

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t Next(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, host.size() * sizeof(T)), "hipMalloc");
  CheckHip(
      hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
      "hipMemcpy");
  return static_cast<T*>(p);
}

template<typename T>
T* Alloc(std::size_t count) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, count * sizeof(T)), "hipMalloc");
  CheckHip(hipMemset(p, 0, count * sizeof(T)), "hipMemset");
  return static_cast<T*>(p);
}

template<typename T>
std::uint64_t Hash(const T* device, std::size_t count) {
  std::vector<std::uint8_t> host(count * sizeof(T));
  CheckHip(hipMemcpy(host.data(), device, host.size(), hipMemcpyDeviceToHost),
           "hipMemcpy");
  std::uint64_t h = 1469598103934665603ULL;
  for (std::uint8_t c : host)
    h = (h ^ c) * 1099511628211ULL;
  return h;
}

/// Random superblocks with model-like F16 scales (weights near 0.02).
std::vector<std::uint8_t> MakeBlocks(q::WeightType type, std::size_t elems,
                                     std::uint32_t seed) {
  const std::size_t bytes = BlockBytes(type);
  const std::size_t blocks = elems / 256;
  std::vector<std::uint8_t> w(blocks * bytes);
  for (std::size_t i = 0; i < w.size(); i += 4) {
    const std::uint32_t r = Next(&seed);
    std::memcpy(&w[i], &r, std::min<std::size_t>(4, w.size() - i));
  }
  const float base = type == q::WeightType::kIQ3_S    ? 2e-4F
                     : type == q::WeightType::kIQ4_XS ? 2.5e-5F
                                                      : 4e-4F;
  for (std::size_t b = 0; b < blocks; ++b) {
    const float d =
        base * (1.0F + 0.5F * static_cast<float>(Next(&seed) % 97) / 97);
    const __half dh = __float2half(d);
    std::memcpy(&w[b * bytes], &dh, 2);
    if (type == q::WeightType::kQ4_K || type == q::WeightType::kQ5_K) {
      const __half mh = __float2half(0.5F * d);
      std::memcpy(&w[b * bytes + 2], &mh, 2);
    }
  }
  return w;
}

/// Synthetic top-10 routing: expert e has weight 1 / (rank + s).
std::vector<std::uint32_t> ZipfCounts(std::uint32_t tokens, double s,
                                      std::uint32_t seed) {
  std::vector<std::uint32_t> rank(kExperts);
  std::iota(rank.begin(), rank.end(), 0U);
  for (std::uint32_t i = kExperts - 1; i > 0; --i)
    std::swap(rank[i], rank[Next(&seed) % (i + 1)]);
  std::vector<double> cdf(kExperts);
  double sum = 0.0;
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    sum += 1.0 / (rank[e] + s);
    cdf[e] = sum;
  }
  std::vector<std::uint32_t> counts(kExperts, 0);
  std::vector<std::uint32_t> row;
  for (std::uint32_t t = 0; t < tokens; ++t) {
    row.clear();
    while (row.size() < kUsed) {
      const double x = sum * (Next(&seed) / 4294967296.0);
      const auto e = std::min<std::uint32_t>(
          kExperts - 1, static_cast<std::uint32_t>(
                            std::lower_bound(cdf.begin(), cdf.end(), x) -
                            cdf.begin()));
      if (std::find(row.begin(), row.end(), e) == row.end())
        row.push_back(e);
    }
    for (std::uint32_t e : row)
      ++counts[e];
  }
  return counts;
}

template<typename F>
void Alternate(int reps, F&& a, F&& b, float* ms_a, float* ms_b) {
  hipEvent_t e0, e1, e2;
  CheckHip(hipEventCreate(&e0), "event");
  CheckHip(hipEventCreate(&e1), "event");
  CheckHip(hipEventCreate(&e2), "event");
  std::vector<float> ta, tb;
  for (int r = 0; r < reps; ++r) {
    CheckHip(hipEventRecord(e0), "record");
    a();
    CheckHip(hipEventRecord(e1), "record");
    b();
    CheckHip(hipEventRecord(e2), "record");
    CheckHip(hipEventSynchronize(e2), "sync");
    float x = 0.0F, y = 0.0F;
    CheckHip(hipEventElapsedTime(&x, e0, e1), "elapsed");
    CheckHip(hipEventElapsedTime(&y, e1, e2), "elapsed");
    ta.push_back(x);
    tb.push_back(y);
  }
  std::sort(ta.begin(), ta.end());
  std::sort(tb.begin(), tb.end());
  *ms_a = ta[ta.size() / 2];
  *ms_b = tb[tb.size() / 2];
  for (hipEvent_t e : {e0, e1, e2})
    (void)hipEventDestroy(e);
}

struct Result {
  float pair_ms = 0.0F;
  float two_ms = 0.0F;
  double tflop = 0.0;
  bool same = true;
};

Result RunLayer(const std::vector<std::uint32_t>& counts, int reps,
                const void* d_gate, const void* d_up) {
  const std::uint32_t slots = std::accumulate(counts.begin(), counts.end(), 0U);
  const std::uint32_t tokens = slots / kUsed;
  // Expert e takes the next counts[e] tokens cyclically, so every token
  // receives kUsed distinct experts (counts[e] <= tokens).
  std::vector<std::int32_t> ids(slots);
  std::vector<std::uint32_t> filled(tokens, 0);
  std::uint32_t cursor = 0;
  std::uint32_t active = 0;
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    active += counts[e] != 0;
    for (std::uint32_t i = 0; i < counts[e]; ++i) {
      const std::uint32_t t = cursor++ % tokens;
      ids[std::size_t{t} * kUsed + filled[t]++] = static_cast<std::int32_t>(e);
    }
  }
  // The executor's maps (RouteHints): 128-token tiles for the paired route;
  // the two-pass route's 48-token tiles once slots >= 16 per expert, else 16.
  const std::uint32_t two_rows = slots >= 16 * kExperts ? 48 : 16;
  const auto map = [&](std::uint32_t rows) {
    std::vector<std::int32_t> tiles;
    for (std::uint32_t e = 0; e < kExperts; ++e)
      for (std::uint32_t j = 0; j < (counts[e] + rows - 1) / rows; ++j)
        tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
    return tiles;
  };
  const auto pair_tiles = map(128);
  const auto two_tiles = map(two_rows);
  const auto n_pair = static_cast<std::uint32_t>(pair_tiles.size());
  const auto n_two = static_cast<std::uint32_t>(two_tiles.size());

  const std::size_t compact = q::RoutedCompactRows(slots, kExperts);
  auto* d_ids = Upload(ids);
  auto* d_counts = Upload(counts);
  auto* d_pair_tiles = Upload(pair_tiles);
  auto* d_two_tiles = Upload(two_tiles);
  auto* d_bounds = Alloc<std::int32_t>(kExperts + 1);
  auto* d_cursors = Alloc<std::int32_t>(kExperts);
  auto* d_rows_token = Alloc<std::int32_t>(compact);
  auto* d_rows_slot = Alloc<std::int32_t>(compact);
  q::RoutedCompact(d_ids, d_counts, d_bounds, d_cursors, d_rows_token,
                   d_rows_slot, tokens, kUsed, kExperts, nullptr);

  std::uint32_t seed = 0x12345678U;
  std::vector<__half> x(std::size_t{tokens} * kHidden);
  for (auto& v : x)
    v = __float2half(
        static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
        1000.0F);
  auto* d_x = Upload(x);
  auto* d_pair_out = Alloc<__half>(std::size_t{slots} * kFf);
  auto* d_two_out = Alloc<__half>(std::size_t{slots} * kFf);
  auto* d_gate_f32 = Alloc<float>(std::size_t{slots} * kFf);
  CheckHip(hipDeviceSynchronize(), "setup");

  const auto pair = [&] {
    if (!q::RoutedGatedF16Gemm(d_gate, d_up, gate_type, d_x, d_pair_tiles,
                               n_pair, d_bounds, d_rows_token, d_rows_slot,
                               d_pair_out, kFf, kHidden, nullptr))
      throw std::runtime_error("paired gate/up GEMM rejected");
  };
  const auto two_pass = [&] {
    if (!q::RoutedF16Gemm(d_gate, gate_type, d_x, d_two_tiles, n_two, two_rows,
                          d_bounds, d_rows_token, d_rows_slot, nullptr,
                          d_gate_f32, nullptr, kFf, kHidden, nullptr) ||
        !q::RoutedF16Gemm(d_up, gate_type, d_x, d_two_tiles, n_two, two_rows,
                          d_bounds, d_rows_token, d_rows_slot, d_gate_f32,
                          nullptr, d_two_out, kFf, kHidden, nullptr))
      throw std::runtime_error("two-pass gate/up GEMM rejected");
  };
  const std::function<void()> fa = pair, fb = two_pass;
  fa();
  fb();
  CheckHip(hipDeviceSynchronize(), "warm-up");
  Result r;
  Alternate(reps, fa, fb, &r.pair_ms, &r.two_ms);
  const std::uint64_t h_pair = Hash(d_pair_out, std::size_t{slots} * kFf);
  const std::uint64_t h_two = Hash(d_two_out, std::size_t{slots} * kFf);
  r.same = h_pair == h_two;
  r.tflop = 4.0 * slots * kFf * kHidden / 1e12;
  std::printf(
      "tokens %5u active %3u tiles %4u/%4u(%u)  pair %7.3f ms %5.1f TFLOPS  "
      "two-pass %7.3f ms %5.1f TFLOPS  x%.3f  %016llx %s\n",
      tokens, active, n_pair, n_two, two_rows, r.pair_ms,
      r.tflop / r.pair_ms * 1e3, r.two_ms, r.tflop / r.two_ms * 1e3,
      r.two_ms / r.pair_ms, static_cast<unsigned long long>(h_pair),
      r.same ? "identical" : "DIFFERENT");
  for (void* p : std::initializer_list<void*>{
           d_ids, d_counts, d_pair_tiles, d_two_tiles, d_bounds, d_cursors,
           d_rows_token, d_rows_slot, d_x, d_pair_out, d_two_out, d_gate_f32})
    CheckHip(hipFree(p), "hipFree");
  return r;
}

}  // namespace

int main(int argc, char** argv) try {
  const int reps = argc > 1 ? std::max(1, std::atoi(argv[1])) : 10;
  const std::string source = argc > 2 ? argv[2] : "4096";
  if (argc > 3)
    gate_type = ParseType(argv[3]);
  const int n_layers = argc > 4 ? std::max(1, std::atoi(argv[4])) : 40;
  std::vector<std::vector<std::uint32_t>> layers;
  if (!source.empty() && std::all_of(source.begin(), source.end(), [](char c) {
        return c >= '0' && c <= '9';
      })) {
    const auto tokens = static_cast<std::uint32_t>(std::stoul(source));
    for (int l = 0; l < n_layers; ++l) {
      // Skew from hot (s = 4) to flat (s = 64) across the layers.
      const double s = 4.0 * std::pow(16.0, n_layers > 1
                                                ? double(l) / (n_layers - 1)
                                                : 0.5);
      layers.push_back(ZipfCounts(tokens, s, 0x9E3779B9U + 7919U * l));
    }
  } else {
    std::ifstream in(source);
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream fields(line);
      std::vector<std::uint32_t> counts;
      std::uint32_t n;
      while (fields >> n)
        counts.push_back(n);
      if (counts.size() == kExperts)
        layers.push_back(std::move(counts));
    }
    if (layers.empty())
      throw std::runtime_error("no 512-expert count lines in " + source);
  }
  const std::size_t elems = std::size_t{kExperts} * kFf * kHidden;
  auto* d_gate = Upload(MakeBlocks(gate_type, elems, 1U));
  auto* d_up = Upload(MakeBlocks(gate_type, elems, 2U));
  Result total;
  std::size_t different = 0;
  for (const auto& counts : layers) {
    const Result r = RunLayer(counts, reps, d_gate, d_up);
    total.pair_ms += r.pair_ms;
    total.two_ms += r.two_ms;
    total.tflop += r.tflop;
    different += !r.same;
  }
  std::printf(
      "total %zu layers: pair %.2f ms (%.1f TFLOPS)  two-pass %.2f ms "
      "(%.1f TFLOPS)  speed-up x%.3f  outputs %s\n",
      layers.size(), total.pair_ms, total.tflop / total.pair_ms * 1e3,
      total.two_ms, total.tflop / total.two_ms * 1e3,
      total.two_ms / total.pair_ms,
      different == 0 ? "identical" : "DIFFERENT (run the test)");
  (void)hipFree(d_gate);
  (void)hipFree(d_up);
  return different == 0 ? 0 : 2;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
