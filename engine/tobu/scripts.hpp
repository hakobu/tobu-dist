// tobu-fnx: script ban "no-CJK": code point classifier and check of individual token texts.
// Copyright (c) 2026 hakobu, Apache-2.0 (tobu). Standard library only: tobu/test_scripts.cpp checks it on macOS.
//
// A token is banned if its text (Model::TokenText: the raw bytes of the token, byte BPE already resolved)
// contains a CJK character, or can begin one:
//
//  * Complete characters in the token: banned if the code point lies in IsCjk.
//  * Incomplete sequence at the token end (lead byte, possibly with continuation bytes, rest in the next token): banned if
//    ANY completion yields a CJK character (range of possible code points intersects a CJK range).
//    So a CJK character cannot arise from byte pieces either: every character split across tokens begins
//    with a token that ends in a prefix of that character, and that one is banned. Consequence: the single lead bytes
//    E1 (Hangul Jamo), E2 (CJK radicals from U+2E80), E3–ED, EF (fullwidth, compatibility ideographs) and F0 (planes
//    2/3) are banned; longer prefixes without a CJK completion (e.g. E2 80 = U+2000–203F, F0 9F = emoji) stay free.
//    The model still reaches characters of these blocks via whole tokens or two-byte pieces (byte BPE has its own
//    tokens for the frequent lead/continuation byte pairs); at most rare characters are lost that would only exist via the
//    single lead byte, accepted deliberately (better tight than a CJK character slipping through).
//  * Continuation bytes at the token start (rest of a character from the previous token): irrelevant, skipped; the
//    character was begun by an earlier token, and its prefix was already checked (prompts end on character boundaries).
//  * Invalid bytes (stray continuation bytes mid-token, F5–FF, truncated sequence mid-token): no character.
//    Overlong encodings are treated like their value (they do not occur in the vocabulary; banned when in doubt).
#ifndef TOBU_FNX_SCRIPTS_HPP_
#define TOBU_FNX_SCRIPTS_HPP_

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace tobu::scripts {

struct Range {
  char32_t lo, hi;  // inclusive
};

// CJK blocks (Unicode 15/16), ascending. Not included: Yi (A000–A4CF), Yijing hexagrams (4DC0–4DFF), enclosed
// ideographic supplement U+1F200 (emoji-like).
inline constexpr Range kCjk[] = {
    {0x1100, 0x11FF},    // Hangul Jamo
    {0x2E80, 0x2EFF},    // CJK Radicals Supplement
    {0x2F00, 0x2FDF},    // Kangxi Radicals
    {0x2FF0, 0x2FFF},    // Ideographic Description Characters
    {0x3000, 0x303F},    // CJK Symbols and Punctuation (incl. U+3000 ideographic space, 。「」)
    {0x3040, 0x309F},    // Hiragana
    {0x30A0, 0x30FF},    // Katakana
    {0x3100, 0x312F},    // Bopomofo
    {0x3130, 0x318F},    // Hangul Compatibility Jamo
    {0x3190, 0x319F},    // Kanbun
    {0x31A0, 0x31BF},    // Bopomofo Extended
    {0x31C0, 0x31EF},    // CJK Strokes
    {0x31F0, 0x31FF},    // Katakana Phonetic Extensions
    {0x3200, 0x32FF},    // Enclosed CJK Letters and Months
    {0x3300, 0x33FF},    // CJK Compatibility
    {0x3400, 0x4DBF},    // CJK Unified Ideographs Extension A
    {0x4E00, 0x9FFF},    // CJK Unified Ideographs
    {0xA960, 0xA97F},    // Hangul Jamo Extended-A
    {0xAC00, 0xD7AF},    // Hangul Syllables
    {0xD7B0, 0xD7FF},    // Hangul Jamo Extended-B
    {0xF900, 0xFAFF},    // CJK Compatibility Ideographs
    {0xFE10, 0xFE1F},    // Vertical Forms
    {0xFE30, 0xFE4F},    // CJK Compatibility Forms
    {0xFF00, 0xFFEF},    // Halfwidth and Fullwidth Forms (，：（）！ Ａ ｱ …)
    {0x1AFF0, 0x1B16F},  // Kana: Extended-B, Supplement, Extended-A, Small Kana Extension
    {0x20000, 0x3FFFF},  // Planes 2 and 3: CJK Extensions B–I, Compatibility Supplement
};

/// Does the code point lie in a CJK block (Han, Kana, Hangul, Bopomofo, CJK punctuation, fullwidth)?
[[nodiscard]] constexpr bool IsCjk(char32_t cp) noexcept {
  for (const auto& r : kCjk)
    if (cp >= r.lo && cp <= r.hi) return true;
  return false;
}

/// Does [lo, hi] intersect a CJK block?
[[nodiscard]] constexpr bool CjkIn(char32_t lo, char32_t hi) noexcept {
  for (const auto& r : kCjk)
    if (lo <= r.hi && r.lo <= hi) return true;
  return false;
}

/// Length of the UTF-8 sequence from its lead byte; 0 for continuation bytes (80–BF) and invalid lead bytes (C0, C1, F5–FF).
[[nodiscard]] constexpr int SequenceLength(std::uint8_t lead) noexcept {
  if (lead < 0x80) return 1;
  if (lead < 0xC2) return 0;
  if (lead < 0xE0) return 2;
  if (lead < 0xF0) return 3;
  if (lead < 0xF5) return 4;
  return 0;
}

[[nodiscard]] constexpr bool IsContinuation(std::uint8_t b) noexcept { return (b & 0xC0) == 0x80; }

/// A prefix (lead byte + 0..len-2 continuation bytes) of a len-byte sequence: can it be completed to a CJK character?
/// The missing continuation bytes range over 80..BF; the range of possible code points is checked against kCjk.
[[nodiscard]] constexpr bool CjkCompletionPossible(const std::uint8_t* p, int have, int len) noexcept {
  const char32_t lead_bits = len == 2 ? 0x1F : len == 3 ? 0x0F : 0x07;
  char32_t v = p[0] & lead_bits;
  for (int i = 1; i < have; ++i) v = (v << 6) | (p[i] & 0x3F);
  const int missing = len - have;
  const char32_t lo = v << (6 * missing);
  const char32_t hi = lo | ((char32_t{1} << (6 * missing)) - 1);
  return CjkIn(lo, hi);
}

/// Does the token text contain a CJK character or end in a prefix that can begin one? (Rules: see file header.)
[[nodiscard]] inline bool TokenHasCjk(std::string_view text) noexcept {
  const auto* b = reinterpret_cast<const std::uint8_t*>(text.data());
  const std::size_t n = text.size();
  std::size_t i = 0;
  while (i < n && IsContinuation(b[i])) ++i;  // rest of a character from the previous token
  while (i < n) {
    const int len = SequenceLength(b[i]);
    if (len <= 1) {  // ASCII or invalid byte
      ++i;
      continue;
    }
    int have = 1;
    while (have < len && i + have < n && IsContinuation(b[i + have])) ++have;
    if (have == len) {
      char32_t cp = b[i] & (len == 2 ? 0x1F : len == 3 ? 0x0F : 0x07);
      for (int k = 1; k < len; ++k) cp = (cp << 6) | (b[i + k] & 0x3F);
      if (IsCjk(cp)) return true;
      i += static_cast<std::size_t>(len);
      continue;
    }
    if (i + have == n) return CjkCompletionPossible(b + i, have, len);  // cut off at the token end
    ++i;  // truncated sequence mid-token: no character
  }
  return false;
}

/// Ban mask over the vocabulary: mask[id] = 1 if TokenHasCjk(text(id)). `count` = number of banned tokens.
template<class TextOf>
[[nodiscard]] std::vector<std::uint8_t> CjkMask(std::size_t vocab, TextOf&& text_of, std::size_t* count = nullptr) {
  std::vector<std::uint8_t> mask(vocab, 0);
  std::size_t c = 0;
  for (std::size_t id = 0; id < vocab; ++id) {
    if (TokenHasCjk(text_of(id))) {
      mask[id] = 1;
      ++c;
    }
  }
  if (count != nullptr) *count = c;
  return mask;
}

}  // namespace tobu::scripts

#endif  // TOBU_FNX_SCRIPTS_HPP_
