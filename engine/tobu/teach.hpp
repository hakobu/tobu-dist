// tobu (MTP-head training, TOB-38): teacher-data sink for the TEACH command. Copyright (c) 2026 hakobu, Apache-2.0.
#ifndef TOBU_TEACH_HPP_
#define TOBU_TEACH_HPP_

#include <cstdint>
#include <utility>
#include <vector>

namespace tobu {

/// Set only during TEACH: the prefill computes the MTP block with the full body over all rows and, for
/// MTP positions in `spans` ([begin, end) absolute; position i reads token i+1 and trunk row i, predicts token i+2),
/// stores the residual before `hc_head` (r, 4 × 2560) and its output (y, 2560) as fp16.
struct TeachSink {
  std::vector<std::pair<std::uint32_t, std::uint32_t>> spans;
  std::vector<std::uint32_t> pos;
  std::vector<std::uint16_t> r;
  std::vector<std::uint16_t> y;
  /// Stage B: additionally per row the residual after attention (a, before hc_ffn), the MoE input (m, hidden) and
  /// the MoE output (o, hidden, routed + shared, before the residual update).
  bool stage_b{false};
  std::vector<std::uint16_t> a;
  std::vector<std::uint16_t> m;
  std::vector<std::uint16_t> o;
};

}  // namespace tobu

#endif
