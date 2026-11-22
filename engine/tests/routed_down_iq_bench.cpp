// Times the routed expert down projection of tobu-fnx's wide prefill per
// layer for IQ4_NL weights: the register-resident RoutedDownF16Kernel
// (RoutedDownWideF16Gemm) against the format-generic 128-token kernel
// (RoutedF16Gemm, tile_rows 128; the route with TOBU_FNX_IQ_DOWN=0), both on
// the executor's 128-token map, alternating so clock drift hits both. Prints
// per layer the median ms, TFLOPS, speed-up and whether both outputs hash
// the same; then the totals. Layers: as routed_pair_iq_bench (synthetic Zipf
// layers, hot -> flat, or a counts file with 512 per-expert counts a line).
//
// tobu_fnx_routed_down_iq_bench [reps] [tokens | counts-file] [layers]
//   reps 10, tokens 4096, layers 40
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
constexpr std::size_t kBlockBytes = 18;  // block_iq4_nl, 32 elements

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

/// Random IQ4_NL blocks with model-like F16 scales (weights near 0.02).
std::vector<std::uint8_t> MakeBlocks(std::size_t elems, std::uint32_t seed) {
  const std::size_t blocks = elems / 32;
  std::vector<std::uint8_t> w(blocks * kBlockBytes);
  for (std::size_t i = 0; i < w.size(); i += 4) {
    const std::uint32_t r = Next(&seed);
    std::memcpy(&w[i], &r, std::min<std::size_t>(4, w.size() - i));
  }
  for (std::size_t b = 0; b < blocks; ++b) {
    const float d =
        4e-4F * (1.0F + 0.5F * static_cast<float>(Next(&seed) % 97) / 97);
    const __half dh = __float2half(d);
    std::memcpy(&w[b * kBlockBytes], &dh, 2);
  }
  return w;
}

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

struct Result {
  float reg_ms = 0.0F;
  float gen_ms = 0.0F;
  double tflop = 0.0;
  bool same = true;
};

Result RunLayer(const std::vector<std::uint32_t>& counts, int reps,
                const void* d_w) {
  const std::uint32_t slots = std::accumulate(counts.begin(), counts.end(), 0U);
  const std::uint32_t tokens = slots / kUsed;
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
  std::vector<std::int32_t> tiles;
  for (std::uint32_t e = 0; e < kExperts; ++e)
    for (std::uint32_t j = 0; j < (counts[e] + 127) / 128; ++j)
      tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
  const auto n_tiles = static_cast<std::uint32_t>(tiles.size());

  const std::size_t compact = q::RoutedCompactRows(slots, kExperts);
  auto* d_ids = Upload(ids);
  auto* d_counts = Upload(counts);
  auto* d_tiles = Upload(tiles);
  auto* d_bounds = Alloc<std::int32_t>(kExperts + 1);
  auto* d_cursors = Alloc<std::int32_t>(kExperts);
  auto* d_rows_token = Alloc<std::int32_t>(compact);
  auto* d_rows_slot = Alloc<std::int32_t>(compact);
  q::RoutedCompact(d_ids, d_counts, d_bounds, d_cursors, d_rows_token,
                   d_rows_slot, tokens, kUsed, kExperts, nullptr);

  // SwiGLU-like F16 input rows, one per slot.
  std::uint32_t seed = 0x12345678U;
  std::vector<__half> x(std::size_t{slots} * kFf);
  for (auto& v : x)
    v = __float2half(
        static_cast<float>(static_cast<int>(Next(&seed) % 2001) - 1000) /
        2000.0F);
  auto* d_x = Upload(x);
  auto* d_reg = Alloc<__half>(std::size_t{slots} * kHidden);
  auto* d_gen = Alloc<__half>(std::size_t{slots} * kHidden);
  CheckHip(hipDeviceSynchronize(), "setup");

  const auto reg = [&] {
    if (!q::RoutedDownWideF16Gemm(d_w, q::WeightType::kIQ4_NL, d_x, d_tiles,
                                  n_tiles, d_bounds, d_rows_slot, d_rows_slot,
                                  d_reg, kHidden, kFf, nullptr))
      throw std::runtime_error("register-resident down rejected");
  };
  const auto gen = [&] {
    if (!q::RoutedF16Gemm(d_w, q::WeightType::kIQ4_NL, d_x, d_tiles, n_tiles,
                          128, d_bounds, d_rows_slot, d_rows_slot, nullptr,
                          nullptr, d_gen, kHidden, kFf, nullptr))
      throw std::runtime_error("generic down rejected");
  };
  reg();
  gen();
  CheckHip(hipDeviceSynchronize(), "warm-up");
  hipEvent_t e0, e1, e2;
  CheckHip(hipEventCreate(&e0), "event");
  CheckHip(hipEventCreate(&e1), "event");
  CheckHip(hipEventCreate(&e2), "event");
  std::vector<float> ta, tb;
  for (int r = 0; r < reps; ++r) {
    CheckHip(hipEventRecord(e0), "record");
    reg();
    CheckHip(hipEventRecord(e1), "record");
    gen();
    CheckHip(hipEventRecord(e2), "record");
    CheckHip(hipEventSynchronize(e2), "sync");
    float a = 0.0F, b = 0.0F;
    CheckHip(hipEventElapsedTime(&a, e0, e1), "elapsed");
    CheckHip(hipEventElapsedTime(&b, e1, e2), "elapsed");
    ta.push_back(a);
    tb.push_back(b);
  }
  for (hipEvent_t e : {e0, e1, e2})
    (void)hipEventDestroy(e);
  std::sort(ta.begin(), ta.end());
  std::sort(tb.begin(), tb.end());
  Result r;
  r.reg_ms = ta[ta.size() / 2];
  r.gen_ms = tb[tb.size() / 2];
  const std::uint64_t h_reg = Hash(d_reg, std::size_t{slots} * kHidden);
  const std::uint64_t h_gen = Hash(d_gen, std::size_t{slots} * kHidden);
  r.same = h_reg == h_gen;
  r.tflop = 2.0 * slots * kHidden * kFf / 1e12;
  std::printf(
      "tokens %5u active %3u tiles %4u  register %7.3f ms %5.1f TFLOPS  "
      "generic %7.3f ms %5.1f TFLOPS  x%.3f  %016llx %s\n",
      tokens, active, n_tiles, r.reg_ms, r.tflop / r.reg_ms * 1e3, r.gen_ms,
      r.tflop / r.gen_ms * 1e3, r.gen_ms / r.reg_ms,
      static_cast<unsigned long long>(h_reg),
      r.same ? "identical" : "DIFFERENT");
  for (void* p : std::initializer_list<void*>{d_ids, d_counts, d_tiles,
                                              d_bounds, d_cursors, d_rows_token,
                                              d_rows_slot, d_x, d_reg, d_gen})
    CheckHip(hipFree(p), "hipFree");
  return r;
}

}  // namespace

int main(int argc, char** argv) try {
  const int reps = argc > 1 ? std::max(1, std::atoi(argv[1])) : 10;
  const std::string source = argc > 2 ? argv[2] : "4096";
  const int n_layers = argc > 3 ? std::max(1, std::atoi(argv[3])) : 40;
  std::vector<std::vector<std::uint32_t>> layers;
  if (!source.empty() && std::all_of(source.begin(), source.end(), [](char c) {
        return c >= '0' && c <= '9';
      })) {
    const auto tokens = static_cast<std::uint32_t>(std::stoul(source));
    for (int l = 0; l < n_layers; ++l) {
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
  auto* d_w = Upload(MakeBlocks(std::size_t{kExperts} * kHidden * kFf, 3U));
  Result total;
  std::size_t different = 0;
  for (const auto& counts : layers) {
    const Result r = RunLayer(counts, reps, d_w);
    total.reg_ms += r.reg_ms;
    total.gen_ms += r.gen_ms;
    total.tflop += r.tflop;
    different += !r.same;
  }
  std::printf(
      "total %zu layers: register %.2f ms (%.1f TFLOPS)  generic %.2f ms "
      "(%.1f TFLOPS)  speed-up x%.3f  outputs %s\n",
      layers.size(), total.reg_ms, total.tflop / total.reg_ms * 1e3,
      total.gen_ms, total.tflop / total.gen_ms * 1e3,
      total.gen_ms / total.reg_ms,
      different == 0 ? "identical" : "DIFFERENT (run the test)");
  (void)hipFree(d_w);
  return different == 0 ? 0 : 2;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
