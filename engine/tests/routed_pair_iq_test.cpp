// Bitwise check of the paired routed gate/up GEMM (RoutedGatedF16Gemm, one
// launch, gate kept in F32 registers, SwiGLU epilogue) against the two-pass
// route (RoutedF16Gemm: gate to F32, then up with the SwiGLU epilogue) that
// tobu-fnx runs when the pair is off (TOBU_FNX_IQ_PAIR=0). Every F16 output
// must match bit for bit, for IQ3_S and IQ4_XS (the production expert
// gate/up formats) and Q4_K/Q5_K as controls (the paired K-quant kernel is
// known to match), over 1/7/33/48/128/4096-token passes, several expert
// distributions, the 16- and 48-token two-pass tiles, shorter passes over a
// subset of the tokens (prefix-cache splits), and a ragged shape (m = 201).
//
// Ported from gufo-tobu extensions/flash_next_iq/tests/routed_wmma_ops_test.cpp
// (Routing, PairedSwiglu, SeparateSwiglu, CountDiffs, CheckPassRoutes),
// trimmed to the gate/up route and adapted to tobu's launchers.
//
// tobu_fnx_routed_pair_iq_test [--quick]   (exit 0 = every check passed)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gufo/ext/experimental/ggml_dequant.hpp>

#include "kernels/rocm/kernels.hpp"

namespace q = gufo::models::flash_next_iq::rocm;
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

float Uniform(std::uint32_t* state, float scale) {
  return scale *
         static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                            32768) /
         32768.0F;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  T* device = nullptr;
  CheckHip(hipMalloc(&device, host.size() * sizeof(T) + 4096), "hipMalloc");
  CheckHip(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                     hipMemcpyHostToDevice),
           "upload");
  return device;
}

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  CheckHip(
      hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost),
      "download");
  return host;
}

const char* TypeName(q::WeightType type) {
  return type == q::WeightType::kQ4_K    ? "Q4_K"
         : type == q::WeightType::kQ5_K  ? "Q5_K"
         : type == q::WeightType::kIQ3_S ? "IQ3_S"
                                         : "IQ4_XS";
}

std::size_t BlockBytes(q::WeightType type) {
  return type == q::WeightType::kQ4_K    ? 144
         : type == q::WeightType::kQ5_K  ? 176
         : type == q::WeightType::kIQ3_S ? 110
                                         : 136;
}

/// Packed random 256-element superblocks: every byte random (grid indices,
/// signs, high bits, sub-block scales, codes), then model-like F16 scales so
/// weights sit near 0.02 and the outputs stay well inside F16 range.
/// `scale_div` shrinks the scales further (the up projection).
std::vector<std::uint8_t> MakePacked(q::WeightType type, std::size_t e,
                                     std::size_t m, std::size_t k,
                                     std::uint32_t seed,
                                     float scale_div = 1.0F) {
  const std::size_t bytes = BlockBytes(type);
  std::vector<std::uint8_t> packed(e * m * (k / 256) * bytes);
  for (std::size_t i = 0; i < packed.size(); i += 4) {
    const std::uint32_t r = NextRandom(&seed);
    std::memcpy(packed.data() + i, &r, std::min<std::size_t>(4, packed.size() - i));
  }
  // IQ3_S: |w| ~ d (1 + 2 sc) grid (~ d * 100); IQ4_XS: d |ls - 32| |value|
  // (~ d * 800); K-quants as gufo's routed tests.
  const float base = type == q::WeightType::kIQ3_S    ? 2e-4F
                     : type == q::WeightType::kIQ4_XS ? 2.5e-5F
                                                      : 4e-4F;
  for (std::size_t b = 0; b < packed.size(); b += bytes) {
    const float d = base * (1.0F + Uniform(&seed, 0.5F)) / scale_div;
    const __half dh = __float2half(d);
    std::memcpy(packed.data() + b, &dh, 2);
    if (type == q::WeightType::kQ4_K || type == q::WeightType::kQ5_K) {
      const __half mh = __float2half(0.5F * d);
      std::memcpy(packed.data() + b + 2, &mh, 2);
    }
  }
  return packed;
}

/// Expert choice per token (top-`used`, distinct experts).
enum class Dist { kSkewed, kUniform, kZipf, kHot };
const char* DistName(Dist d) {
  return d == Dist::kSkewed    ? "skewed"
         : d == Dist::kUniform ? "uniform"
         : d == Dist::kZipf    ? "zipf"
                               : "hot";
}

std::vector<std::int32_t> MakeIds(Dist dist, std::size_t n_tokens,
                                  std::size_t used, std::size_t experts,
                                  std::uint32_t seed) {
  std::uint32_t state = seed;
  // Zipf-like weights 1 / (rank + 16) over a random expert permutation (the
  // routed GEMM bench's synthetic routing).
  std::vector<double> cdf(experts);
  {
    std::vector<std::uint32_t> rank(experts);
    std::iota(rank.begin(), rank.end(), 0U);
    for (std::size_t i = experts - 1; i > 0; --i)
      std::swap(rank[i], rank[NextRandom(&state) % (i + 1)]);
    double sum = 0.0;
    for (std::size_t e = 0; e < experts; ++e) {
      sum += 1.0 / (rank[e] + 16.0);
      cdf[e] = sum;
    }
  }
  std::vector<std::int32_t> ids(n_tokens * used);
  for (std::size_t t = 0; t < n_tokens; ++t) {
    std::vector<std::int32_t> chosen;
    while (chosen.size() < used) {
      const std::uint32_t r = NextRandom(&state);
      std::int32_t e = 0;
      switch (dist) {
        case Dist::kSkewed:  // a quarter of the picks among 16 experts
          e = static_cast<std::int32_t>((r % 3 == 0 ? r / 3 % 16 : r / 3) %
                                        experts);
          break;
        case Dist::kUniform:
          e = static_cast<std::int32_t>(r % experts);
          break;
        case Dist::kZipf: {
          const double x = cdf.back() * (r / 4294967296.0);
          e = static_cast<std::int32_t>(
              std::lower_bound(cdf.begin(), cdf.end(), x) - cdf.begin());
          e = std::min<std::int32_t>(e, static_cast<std::int32_t>(experts) - 1);
          break;
        }
        case Dist::kHot:  // every token in a few experts: full 128-row tiles
          e = static_cast<std::int32_t>(r % std::min(experts, used + 3));
          break;
      }
      if (std::find(chosen.begin(), chosen.end(), e) == chosen.end())
        chosen.push_back(e);
    }
    std::copy(chosen.begin(), chosen.end(), ids.begin() + t * used);
  }
  return ids;
}

/// Compacted routing of tokens [first, first + n_tokens) of `all_ids`, as
/// MoeExperts builds it (RoutedCompact, 16-row padded buckets).
struct Routing {
  std::vector<std::int32_t> ids;
  std::vector<std::uint32_t> counts;
  std::size_t slots;
  std::int32_t* d_bounds = nullptr;
  std::int32_t* d_rows_token = nullptr;
  std::int32_t* d_rows_slot = nullptr;
  Routing(const std::vector<std::int32_t>& all_ids, std::size_t first,
          std::size_t n_tokens, std::size_t used, std::size_t experts)
      : ids(all_ids.begin() + first * used,
            all_ids.begin() + (first + n_tokens) * used),
        counts(experts, 0),
        slots(n_tokens * used) {
    for (const auto e : ids)
      ++counts[e];
    auto* d_ids = Upload(ids);
    auto* d_counts = Upload(counts);
    auto* d_cursors = Upload(std::vector<std::int32_t>(experts, 0));
    const std::size_t compact = q::RoutedCompactRows(slots, experts);
    d_bounds = Upload(std::vector<std::int32_t>(experts + 1, 0));
    d_rows_token = Upload(std::vector<std::int32_t>(compact, 0));
    d_rows_slot = Upload(std::vector<std::int32_t>(compact, 0));
    q::RoutedCompact(d_ids, d_counts, d_bounds, d_cursors, d_rows_token,
                     d_rows_slot, static_cast<std::uint32_t>(n_tokens),
                     static_cast<std::uint32_t>(used),
                     static_cast<std::uint32_t>(experts), nullptr);
    CheckHip(hipDeviceSynchronize(), "routing compaction");
    for (void* p : {static_cast<void*>(d_ids), static_cast<void*>(d_counts),
                    static_cast<void*>(d_cursors)})
      (void)hipFree(p);
  }
  Routing(const Routing&) = delete;
  Routing& operator=(const Routing&) = delete;
  ~Routing() {
    for (void* p :
         {static_cast<void*>(d_bounds), static_cast<void*>(d_rows_token),
          static_cast<void*>(d_rows_slot)})
      (void)hipFree(p);
  }
  /// The executor's tile map: expert | (tile << 16) per tile_rows tokens.
  std::int32_t* Tiles(std::uint32_t tile_rows, std::uint32_t* n) const {
    std::vector<std::int32_t> tiles;
    for (std::size_t e = 0; e < counts.size(); ++e)
      for (std::uint32_t j = 0; j < (counts[e] + tile_rows - 1) / tile_rows;
           ++j)
        tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
    *n = static_cast<std::uint32_t>(tiles.size());
    return tiles.empty() ? nullptr : Upload(tiles);
  }
};

constexpr std::size_t kGuard = 16;
const __half kFill = __float2half(-7.0F);

/// Paired 128-row gate/up launch (the wide prefill route). Throws if the
/// route rejects the type (no paired kernel for it).
std::vector<__half> PairedSwiglu(const Routing& r, const void* gate,
                                 const void* up, q::WeightType type,
                                 const __half* x, std::size_t m,
                                 std::size_t k) {
  std::uint32_t n = 0;
  auto* tiles = r.Tiles(128, &n);
  auto* out = Upload(std::vector<__half>(r.slots * m + kGuard, kFill));
  if (!q::RoutedGatedF16Gemm(gate, up, type, x, tiles, n, r.d_bounds,
                             r.d_rows_token, r.d_rows_slot, out, m, k, nullptr))
    throw std::runtime_error(std::string("paired routed launch rejected ") +
                             TypeName(type));
  CheckHip(hipDeviceSynchronize(), "paired routed launch");
  auto host = Download(out, r.slots * m + kGuard);
  (void)hipFree(tiles);
  (void)hipFree(out);
  for (std::size_t i = r.slots * m; i < host.size(); ++i)
    if (std::memcmp(&host[i], &kFill, 2) != 0)
      throw std::runtime_error("paired launch overwrote its guard");
  host.resize(r.slots * m);
  return host;
}

/// Separate gate (F32) and up (SwiGLU, F16) launches at one tile height (the
/// two-pass route). `gate_out` receives the F32 gate.
std::vector<__half> SeparateSwiglu(const Routing& r, const void* gate,
                                   const void* up, q::WeightType type,
                                   const __half* x, std::size_t m,
                                   std::size_t k, std::uint32_t tile_rows,
                                   std::vector<float>* gate_out = nullptr) {
  std::uint32_t n = 0;
  auto* tiles = r.Tiles(tile_rows, &n);
  auto* g = Upload(std::vector<float>(r.slots * m, -7.0F));
  auto* out = Upload(std::vector<__half>(r.slots * m + kGuard, kFill));
  if (!q::RoutedF16Gemm(gate, type, x, tiles, n, tile_rows, r.d_bounds,
                        r.d_rows_token, r.d_rows_slot, nullptr, g, nullptr, m,
                        k, nullptr) ||
      !q::RoutedF16Gemm(up, type, x, tiles, n, tile_rows, r.d_bounds,
                        r.d_rows_token, r.d_rows_slot, g, nullptr, out, m, k,
                        nullptr))
    throw std::runtime_error("separate routed launch rejected");
  CheckHip(hipDeviceSynchronize(), "separate routed launch");
  auto host = Download(out, r.slots * m);
  if (gate_out != nullptr)
    *gate_out = Download(g, r.slots * m);
  (void)hipFree(tiles);
  (void)hipFree(g);
  (void)hipFree(out);
  return host;
}

/// Counts differing F16 bit patterns between `a` rows and `b` rows, where
/// b's slot s corresponds to a's slot s + a_offset; reports the first and
/// how many differ only in the sign of zero.
std::size_t CountDiffs(const std::vector<__half>& a, std::size_t a_offset,
                       const std::vector<__half>& b, std::size_t m,
                       const std::string& what) {
  std::size_t diffs = 0;
  std::size_t zero_sign = 0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    std::uint16_t x, y;
    std::memcpy(&x, &a[a_offset * m + i], 2);
    std::memcpy(&y, &b[i], 2);
    if (x != y) {
      if (diffs == 0)
        std::cout << "  " << what << ": first diff slot " << i / m << " row "
                  << i % m << ": 0x" << std::hex << x << " vs 0x" << y
                  << std::dec << " (" << __half2float(a[a_offset * m + i])
                  << " vs " << __half2float(b[i]) << ")\n";
      zero_sign += ((x | y) & 0x7FFFU) == 0;
      ++diffs;
    }
  }
  if (diffs != 0)
    std::cout << "  FAIL " << what << ": " << diffs << " of " << b.size()
              << " values differ (" << zero_sign << " only in the sign of 0)\n";
  return diffs;
}

/// The outputs must be finite and mostly non-zero, or a match proves little.
bool Nontrivial(const std::vector<__half>& v, const std::string& what) {
  std::size_t nonzero = 0;
  for (const __half h : v) {
    const float f = __half2float(h);
    if (!std::isfinite(f)) {
      std::cout << "  FAIL " << what << ": non-finite output\n";
      return false;
    }
    nonzero += f != 0.0F;
  }
  if (v.empty() || nonzero * 2 < v.size()) {
    std::cout << "  FAIL " << what << ": only " << nonzero << " of "
              << v.size() << " outputs are non-zero\n";
    return false;
  }
  return true;
}

/// Independent F64 check of a few paired rows over the CPU dequantization:
/// silu(gate) * up within F16 rounding of the F16-weight route.
bool ReferenceRows(const std::vector<__half>& paired,
                   const std::vector<std::int32_t>& ids, std::size_t used,
                   q::WeightType type, const std::vector<std::uint8_t>& gate,
                   const std::vector<std::uint8_t>& up,
                   const std::vector<__half>& x, std::size_t m,
                   std::size_t k, const std::string& what) {
  const auto decode = type == q::WeightType::kIQ3_S
                          ? gufo::quant::DequantizeIQ3_S
                      : type == q::WeightType::kIQ4_XS
                          ? gufo::quant::DequantizeIQ4_XS
                      : type == q::WeightType::kQ5_K
                          ? gufo::quant::DequantizeQ5_K
                          : gufo::quant::DequantizeQ4_K;
  const std::size_t row_bytes = (k / 256) * BlockBytes(type);
  std::vector<float> wg(k), wu(k);
  double worst = 0.0;
  double scale = 0.0;
  const std::size_t slots = ids.size();
  for (std::size_t probe = 0; probe < 24; ++probe) {
    const std::size_t s = (probe * 7919) % slots;
    const std::size_t t = s / used;
    const std::size_t e = static_cast<std::size_t>(ids[s]);
    const std::size_t row = (probe * 104729) % m;
    decode(gate.data() + (e * m + row) * row_bytes, wg.data(), k);
    decode(up.data() + (e * m + row) * row_bytes, wu.data(), k);
    double g = 0.0, u = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      const double xv = __half2float(x[t * k + i]);
      g += static_cast<double>(wg[i]) * xv;
      u += static_cast<double>(wu[i]) * xv;
    }
    const double ref = g / (1.0 + std::exp(-g)) * u;
    const double got = __half2float(paired[s * m + row]);
    worst = std::max(worst, std::abs(ref - got));
    scale = std::max(scale, std::abs(ref));
  }
  // F16 weights (rounded d * scale) and F16 output: a few 1e-3 relative.
  if (!(worst <= 1e-2 * scale + 1e-6)) {
    std::cout << "  FAIL " << what << ": paired rows vs F64 reference, worst "
              << worst << " at scale " << scale << '\n';
    return false;
  }
  return true;
}

struct Shape {
  std::size_t m, k, used, experts;
};

/// All checks for one gate/up format at one shape.
bool CheckFormat(q::WeightType type, const Shape& sh,
                 const std::vector<std::size_t>& token_counts,
                 const std::vector<Dist>& dists) {
  const std::size_t m = sh.m, k = sh.k, used = sh.used, experts = sh.experts;
  const std::string name = std::string(TypeName(type)) + " m=" +
                           std::to_string(m) + " k=" + std::to_string(k) +
                           " E=" + std::to_string(experts);
  std::cout << name << '\n';
  const auto gate = MakePacked(type, experts, m, k,
                         0xA11CEU ^ static_cast<std::uint32_t>(m));
  const auto up = MakePacked(type, experts, m, k,
                       0xB0B0U ^ static_cast<std::uint32_t>(m), 16.0F);
  auto* d_gate = Upload(gate);
  auto* d_up = Upload(up);
  const std::size_t max_tokens =
      *std::max_element(token_counts.begin(), token_counts.end());
  std::uint32_t state = 0xC2C0C2C0U ^ static_cast<std::uint32_t>(type);
  std::vector<__half> x(max_tokens * k);
  // Mixed row magnitudes, as gufo's route checks: quiet rows and loud rows.
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = __float2half(Uniform(&state, (i / k) % 3 == 0 ? 0.05F : 3.0F));
  auto* d_x = Upload(x);
  bool ok = true;
  std::size_t checks = 0;
  for (const Dist dist : dists) {
    for (const std::size_t n : token_counts) {
      const auto ids = MakeIds(dist, n, used, experts,
                               0x5EED0000U ^ static_cast<std::uint32_t>(n * 31) ^
                                   static_cast<std::uint32_t>(dist));
      const std::string what =
          name + " " + DistName(dist) + " n=" + std::to_string(n);
      const Routing all(ids, 0, n, used, experts);
      const auto paired = PairedSwiglu(all, d_gate, d_up, type, d_x, m, k);
      ok = Nontrivial(paired, what) && ok;
      // Both two-pass tile heights (16: short batches, 48: wide prefill).
      for (const std::uint32_t tile : {16U, 48U}) {
        ok = CountDiffs(paired, 0,
                        SeparateSwiglu(all, d_gate, d_up, type, d_x, m, k,
                                       tile),
                        m, what + " paired vs two-pass " +
                               std::to_string(tile)) == 0 &&
             ok;
        ++checks;
      }
      // Replays are deterministic.
      const auto replay = PairedSwiglu(all, d_gate, d_up, type, d_x, m, k);
      if (std::memcmp(paired.data(), replay.data(),
                      paired.size() * sizeof(__half)) != 0) {
        std::cout << "  FAIL " << what << ": paired replay changed bits\n";
        ok = false;
      }
      if (dist == dists.front())
        ok = ReferenceRows(paired, ids, used, type, gate, up, x, m, k, what) &&
             ok;
      // Shorter passes over some of the tokens (a restored prefix's tail,
      // ragged chunks): the executor takes the two-pass 16-row route up to
      // eight tokens and the wide (paired) route above.
      if (n >= 64) {
        const std::vector<std::pair<std::size_t, std::size_t>> passes = {
            {n - 7, 7},
            {n - 1, 1},
            {n / 3, 9},
            {n / 2, 33},
            {n / 5, std::min<std::size_t>(n - n / 5, 96)}};
        for (const auto& [first, count] : passes) {
          const Routing part(ids, first, count, used, experts);
          const std::string sub = what + " pass " + std::to_string(count) +
                                  "@" + std::to_string(first);
          const auto short_pass =
              count <= 8 ? SeparateSwiglu(part, d_gate, d_up, type,
                                          d_x + first * k, m, k, 16)
                         : PairedSwiglu(part, d_gate, d_up, type,
                                        d_x + first * k, m, k);
          ok = CountDiffs(paired, first * used, short_pass, m, sub) == 0 && ok;
          ++checks;
        }
      }
    }
  }
  std::cout << "  " << checks << " comparisons " << (ok ? "ok" : "FAILED")
            << '\n';
  for (void* p : {static_cast<void*>(d_gate), static_cast<void*>(d_up),
                  static_cast<void*>(d_x)})
    (void)hipFree(p);
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
  try {
    // Model shape: expert ffn 640, hidden 2560, top-10 of 512 experts.
    const Shape model{640, 2560, 10, 512};
    // Ragged: odd row count (row tiles past m, odd-m stores), few experts.
    const Shape ragged{201, 512, 4, 8};
    const std::vector<std::size_t> tokens =
        quick ? std::vector<std::size_t>{1, 7, 33, 128}
              : std::vector<std::size_t>{1, 7, 33, 48, 128, 4096};
    const std::vector<Dist> dists =
        quick ? std::vector<Dist>{Dist::kSkewed, Dist::kHot}
              : std::vector<Dist>{Dist::kSkewed, Dist::kUniform, Dist::kZipf,
                                  Dist::kHot};
    bool ok = true;
    // Controls first: the K-quant pair is known to match its two-pass route.
    ok = CheckFormat(q::WeightType::kQ4_K, model, {33, 4096},
                     {Dist::kSkewed}) &&
         ok;
    for (const auto type : {q::WeightType::kIQ3_S, q::WeightType::kIQ4_XS}) {
      ok = CheckFormat(type, model, tokens, dists) && ok;
      ok = CheckFormat(type, ragged, {1, 7, 97, 300}, {Dist::kSkewed}) && ok;
    }
    std::cout << (ok ? "routed pair IQ test: all bit-identical\n"
                     : "routed pair IQ test: FAILED\n");
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "routed pair IQ test: " << error.what() << '\n';
    return 1;
  }
}
