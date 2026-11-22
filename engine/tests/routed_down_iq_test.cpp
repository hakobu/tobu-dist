// Bitwise check of the register-resident wide routed down projection
// (RoutedDownWideF16Gemm -> RoutedDownF16Kernel) against the routes tobu-fnx
// runs otherwise: for IQ4_NL the format-generic 128-token kernel
// (RoutedF16Gemm, tile_rows 128; TOBU_FNX_IQ_DOWN=0) and the 16/48-token
// tiles of short and decode-time batches; Q8_0 (and Q5_1) as controls, whose
// wide route was already register-resident. Every F16 output must match bit
// for bit over 1/7/33/48/128/4096-token passes, several expert distributions,
// shorter passes over a subset of the tokens (prefix-cache splits) and a
// ragged shape (m = 201, k = 128).
//
// Same harness as routed_pair_iq_test (ported from gufo-tobu's
// routed_wmma_ops_test: Routing, CountDiffs, RoutedDown, CheckPassRoutes).
//
// tobu_fnx_routed_down_iq_test [--quick]   (exit 0 = every check passed)
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
  return type == q::WeightType::kIQ4_NL ? "IQ4_NL"
         : type == q::WeightType::kQ8_0 ? "Q8_0"
                                        : "Q5_1";
}

std::size_t BlockBytes(q::WeightType type) {
  return type == q::WeightType::kIQ4_NL ? 18
         : type == q::WeightType::kQ8_0 ? 34
                                        : 24;
}

/// Packed random 32-element blocks: every byte random, then model-like F16
/// scales (weights near 0.02; Q5_1 m = -16 d centres the codes).
std::vector<std::uint8_t> MakePacked(q::WeightType type, std::size_t e,
                                     std::size_t m, std::size_t k,
                                     std::uint32_t seed) {
  const std::size_t bytes = BlockBytes(type);
  std::vector<std::uint8_t> packed(e * m * (k / 32) * bytes);
  for (std::size_t i = 0; i < packed.size(); i += 4) {
    const std::uint32_t r = NextRandom(&seed);
    std::memcpy(packed.data() + i, &r,
                std::min<std::size_t>(4, packed.size() - i));
  }
  // IQ4_NL |w| ~ d * 60; Q8_0 ~ d * 64; Q5_1 ~ d * 8.
  const float base = type == q::WeightType::kIQ4_NL ? 4e-4F : 1.5e-3F;
  for (std::size_t b = 0; b < packed.size(); b += bytes) {
    const float d = base * (1.0F + Uniform(&seed, 0.5F));
    const __half dh = __float2half(d);
    std::memcpy(packed.data() + b, &dh, 2);
    if (type == q::WeightType::kQ5_1) {
      const __half mh = __float2half(-16.0F * d);
      std::memcpy(packed.data() + b + 2, &mh, 2);
    }
  }
  return packed;
}

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
        case Dist::kSkewed:
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
        case Dist::kHot:
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

/// Compacted routing of tokens [first, first + n_tokens), as MoeExperts.
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

/// Down projection (slot rows in and out). `wide_kernel`: the register-
/// resident RoutedDownWideF16Gemm over the 128-token map; otherwise
/// RoutedF16Gemm at `tile_rows` (128 = format-generic wide kernel for IQ4_NL).
std::vector<__half> RoutedDown(const Routing& r, const void* w,
                               q::WeightType type, const __half* x,
                               std::size_t m, std::size_t k,
                               std::uint32_t tile_rows, bool wide_kernel) {
  std::uint32_t n = 0;
  auto* tiles = r.Tiles(wide_kernel ? 128 : tile_rows, &n);
  auto* out = Upload(std::vector<__half>(r.slots * m + kGuard, kFill));
  const bool launched =
      wide_kernel
          ? q::RoutedDownWideF16Gemm(w, type, x, tiles, n, r.d_bounds,
                                     r.d_rows_slot, r.d_rows_slot, out, m, k,
                                     nullptr)
          : q::RoutedF16Gemm(w, type, x, tiles, n, tile_rows, r.d_bounds,
                             r.d_rows_slot, r.d_rows_slot, nullptr, nullptr,
                             out, m, k, nullptr);
  if (!launched)
    throw std::runtime_error(std::string("routed down launch rejected ") +
                             TypeName(type) +
                             (wide_kernel ? " (register-resident)"
                                          : " tile " + std::to_string(tile_rows)));
  CheckHip(hipDeviceSynchronize(), "routed down launch");
  auto host = Download(out, r.slots * m + kGuard);
  (void)hipFree(tiles);
  (void)hipFree(out);
  for (std::size_t i = r.slots * m; i < host.size(); ++i)
    if (std::memcmp(&host[i], &kFill, 2) != 0)
      throw std::runtime_error("down launch overwrote its guard");
  host.resize(r.slots * m);
  return host;
}

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

/// F64 spot check of a few rows over the CPU dequantization.
bool ReferenceRows(const std::vector<__half>& got,
                   const std::vector<std::int32_t>& ids, q::WeightType type,
                   const std::vector<std::uint8_t>& w,
                   const std::vector<__half>& x, std::size_t m, std::size_t k,
                   const std::string& what) {
  const auto decode = type == q::WeightType::kIQ4_NL
                          ? gufo::quant::DequantizeIQ4_NL
                      : type == q::WeightType::kQ8_0
                          ? gufo::quant::DequantizeQ8_0
                          : gufo::quant::DequantizeQ5_1;
  const std::size_t row_bytes = (k / 32) * BlockBytes(type);
  std::vector<float> wr(k);
  double worst = 0.0;
  double scale = 0.0;
  for (std::size_t probe = 0; probe < 24; ++probe) {
    const std::size_t s = (probe * 7919) % ids.size();
    const std::size_t e = static_cast<std::size_t>(ids[s]);
    const std::size_t row = (probe * 104729) % m;
    decode(w.data() + (e * m + row) * row_bytes, wr.data(), k);
    double ref = 0.0;
    for (std::size_t i = 0; i < k; ++i)
      ref += static_cast<double>(wr[i]) * __half2float(x[s * k + i]);
    worst = std::max(
        worst, std::abs(ref - static_cast<double>(__half2float(got[s * m + row]))));
    scale = std::max(scale, std::abs(ref));
  }
  if (!(worst <= 1e-2 * scale + 1e-6)) {
    std::cout << "  FAIL " << what << ": rows vs F64 reference, worst "
              << worst << " at scale " << scale << '\n';
    return false;
  }
  return true;
}

struct Shape {
  std::size_t m, k, used, experts;
};

bool CheckFormat(q::WeightType type, const Shape& sh,
                 const std::vector<std::size_t>& token_counts,
                 const std::vector<Dist>& dists) {
  const std::size_t m = sh.m, k = sh.k, used = sh.used, experts = sh.experts;
  const std::string name = std::string(TypeName(type)) + " m=" +
                           std::to_string(m) + " k=" + std::to_string(k) +
                           " E=" + std::to_string(experts);
  std::cout << name << '\n';
  const auto w = MakePacked(type, experts, m, k,
                            0xD0D0U ^ static_cast<std::uint32_t>(m));
  auto* d_w = Upload(w);
  const std::size_t max_slots =
      *std::max_element(token_counts.begin(), token_counts.end()) * used;
  std::uint32_t state = 0xD0E0F0A0U ^ static_cast<std::uint32_t>(type);
  // Down inputs are SwiGLU rows (F16, one per slot): quiet and loud rows.
  std::vector<__half> x(max_slots * k);
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = __float2half(Uniform(&state, (i / k) % 3 == 0 ? 0.02F : 1.5F));
  auto* d_x = Upload(x);
  // The generic wide kernel is IQ4_NL's previous route; for Q5_1/Q8_0
  // RoutedF16Gemm's 128 route is the register-resident kernel itself.
  const bool generic = type == q::WeightType::kIQ4_NL;
  bool ok = true;
  std::size_t checks = 0;
  for (const Dist dist : dists) {
    for (const std::size_t n : token_counts) {
      const auto ids =
          MakeIds(dist, n, used, experts,
                  0x5EED1000U ^ static_cast<std::uint32_t>(n * 31) ^
                      static_cast<std::uint32_t>(dist));
      const std::string what =
          name + " " + DistName(dist) + " n=" + std::to_string(n);
      const Routing all(ids, 0, n, used, experts);
      const auto wide = RoutedDown(all, d_w, type, d_x, m, k, 128, true);
      ok = Nontrivial(wide, what) && ok;
      std::vector<std::uint32_t> heights = {16U, 48U};
      if (generic)
        heights.push_back(128U);
      for (const std::uint32_t tile : heights) {
        ok = CountDiffs(wide, 0,
                        RoutedDown(all, d_w, type, d_x, m, k, tile, false), m,
                        what + " register-resident vs " +
                            (tile == 128 ? std::string("generic 128")
                                         : "tile " + std::to_string(tile))) ==
                 0 &&
             ok;
        ++checks;
      }
      const auto replay = RoutedDown(all, d_w, type, d_x, m, k, 128, true);
      if (std::memcmp(wide.data(), replay.data(),
                      wide.size() * sizeof(__half)) != 0) {
        std::cout << "  FAIL " << what << ": replay changed bits\n";
        ok = false;
      }
      if (dist == dists.front())
        ok = ReferenceRows(wide, ids, type, w, x, m, k, what) && ok;
      // Shorter passes over some tokens: up to eight rows the executor takes
      // 16-row tiles, above that the wide route again.
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
              RoutedDown(part, d_w, type, d_x + first * used * k, m, k,
                         count <= 8 ? 16 : 128, count > 8);
          ok = CountDiffs(wide, first * used, short_pass, m, sub) == 0 && ok;
          ++checks;
        }
      }
    }
  }
  std::cout << "  " << checks << " comparisons " << (ok ? "ok" : "FAILED")
            << '\n';
  (void)hipFree(d_w);
  (void)hipFree(d_x);
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
  try {
    // Model shape: hidden 2560 rows over expert ffn 640, top-10 of 512.
    const Shape model{2560, 640, 10, 512};
    // Ragged: odd row count (row tile past m, unaligned stores), short k.
    const Shape ragged{201, 128, 4, 8};
    const std::vector<std::size_t> tokens =
        quick ? std::vector<std::size_t>{1, 7, 33, 128}
              : std::vector<std::size_t>{1, 7, 33, 48, 128, 4096};
    const std::vector<Dist> dists =
        quick ? std::vector<Dist>{Dist::kSkewed, Dist::kHot}
              : std::vector<Dist>{Dist::kSkewed, Dist::kUniform, Dist::kZipf,
                                  Dist::kHot};
    bool ok = true;
    ok = CheckFormat(q::WeightType::kIQ4_NL, model, tokens, dists) && ok;
    ok = CheckFormat(q::WeightType::kIQ4_NL, ragged, {1, 7, 97, 300},
                     {Dist::kSkewed}) &&
         ok;
    // Controls: the existing register-resident formats.
    ok = CheckFormat(q::WeightType::kQ8_0, model, {33, 4096},
                     {Dist::kSkewed}) &&
         ok;
    ok = CheckFormat(q::WeightType::kQ5_1, ragged, {97}, {Dist::kSkewed}) && ok;
    std::cout << (ok ? "routed down IQ test: all bit-identical\n"
                     : "routed down IQ test: FAILED\n");
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "routed down IQ test: " << error.what() << '\n';
    return 1;
  }
}
