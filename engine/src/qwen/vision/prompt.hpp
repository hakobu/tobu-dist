// Parts from gufo / flash-next-iq-gufo-extension (storm1er 9ce08ce, MIT); see THIRD_PARTY.md. Changes: tobu.
#ifndef GUFO_MODELS_QWEN_VISION_PROMPT_HPP_
#define GUFO_MODELS_QWEN_VISION_PROMPT_HPP_

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <gufo/ext/image.hpp>
#include "qwen/text/chat_template.hpp"

namespace gufo::models::qwen_iq::vision {


inline constexpr std::uint32_t kImageToken = 248056;
inline constexpr std::uint32_t kPatchSize = 16;
inline constexpr std::uint32_t kMergeSize = 2;

struct ImageGrid {
  std::uint32_t offset{0};  ///< physical first image-pad token
  std::uint32_t height{0};  ///< merged language-token grid, not ViT patches
  std::uint32_t width{0};
  bool operator==(const ImageGrid&) const = default;
};

/// Compact request-owned position state, sufficient to restore RoPE without
/// retaining pixels or encoder scratch in a continuation snapshot.
struct RopeLayout {
  std::vector<ImageGrid> images;
  [[nodiscard]] std::array<std::int32_t, 3> Position(
      std::uint32_t physical) const;
  [[nodiscard]] std::int32_t Delta() const;
  [[nodiscard]] std::uint32_t PrefixLength() const;
  /// Images contributing at least one embedding to the computed prefix.
  [[nodiscard]] RopeLayout Prefix(std::uint32_t token_count) const;
  void Validate(std::uint32_t max_context) const;
  bool operator==(const RopeLayout&) const = default;
};

struct PreparedImage {
  /// Immutable resized RGB8 shared with CPU preprocessing reuse. Temporal
  /// repetition happens on GPU; continuation snapshots do not retain pixels.
  std::shared_ptr<const core::Image> pixels;
  ImageGrid grid;
  std::array<std::uint8_t, 32> prefix_identity{};
};

struct Prompt {
  std::vector<tokenization_iq::TokenId> tokens;
  /// Frontier unchanged when an interrupted assistant is followed by a user.
  std::size_t stable_prefix_tokens{0};
  RopeLayout rope;
  std::vector<PreparedImage> images;
  /// SHA-256 covers decoded pixels, grid placement, preprocessing version,
  /// and the model-specific encoder identity. Tokens remain a separate key.
  std::vector<std::uint8_t> cache_identity;
  [[nodiscard]] std::span<const std::uint8_t> IdentityForPrefix(
      std::size_t token_count) const;
};

/// `kLow` limits both sides of the resized image to 512 pixels, which is at
/// most 16 x 16 merged tokens, unless the model's minimum size is larger. It
/// never exceeds the `kAuto` size. `kAuto` uses the model's full range.
/// tobu: `max_pixels` limits the pixels after scaling (default as in gufo 16.7 MP; production profile: 1 MP = 1,024 tokens).
[[nodiscard]] core::Image ResizeImage(
    const core::Image& image, tokenization_iq::ChatMessage::ImageDetail detail,
    std::uint32_t max_pixels = 16777216);

/// tobu: prompt from already scaled images and the token offsets of their first image token. The server renders the
/// template and tokenizes itself (image pads already expanded to gh*gw); identities as in gufo's Prepare (same
/// hash sequence: version, encoder identity, per image offset/height/width/pixels).
struct PlacedImage {
  std::shared_ptr<const core::Image> pixels;
  std::uint32_t offset{0};
};
[[nodiscard]] Prompt BuildPrompt(std::span<const PlacedImage> images,
                                 std::string_view encoder_identity,
                                 std::uint32_t max_context);

}  // namespace gufo::models::qwen_iq::vision
#endif
