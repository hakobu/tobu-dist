// Test of the script ban (tobu/scripts.hpp) without GPU, without model; runs with the system compiler:
//
//   clang++ -std=c++20 -Wall -Wextra -I engine \
//     engine/tobu/test_scripts.cpp -o /tmp/test_scripts && /tmp/test_scripts
//
// Checks block boundaries of the classifier, whole tokens (CJK, Kana, Hangul, fullwidth vs. Latin, umlauts,
// Cyrillic, Arabic, emoji, dash), byte pieces (prefixes at the token end, continuation bytes at the start), and that
// every split of a CJK character into byte pieces is banned at at least one piece (exhaustive over all
// CJK code points and all split points).
#include <cstdio>
#include <string>
#include <vector>

#include "tobu/scripts.hpp"

namespace {

int fails = 0;

void check(bool ok, const char* name) {
  std::printf("%s %s\n", ok ? "ok    " : "FEHLER", name);
  if (!ok) ++fails;
}

std::string Utf8(char32_t cp) {
  std::string s;
  if (cp < 0x80) {
    s += static_cast<char>(cp);
  } else if (cp < 0x800) {
    s += static_cast<char>(0xC0 | (cp >> 6));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    s += static_cast<char>(0xE0 | (cp >> 12));
    s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    s += static_cast<char>(0xF0 | (cp >> 18));
    s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return s;
}

bool Banned(const std::string& s) { return tobu::scripts::TokenHasCjk(s); }

}  // namespace

int main() {
  using tobu::scripts::IsCjk;
  // 1. Classifier: block boundaries
  check(IsCjk(0x4E00) && IsCjk(0x9FFF) && !IsCjk(0x4DFF) && IsCjk(0x4DBF) && IsCjk(0x3400) && !IsCjk(0x4DC0),
        "1 Han: 4E00–9FFF, Ext A 3400–4DBF, Yijing 4DC0–4DFF frei");
  check(IsCjk(0x3000) && IsCjk(0x303F) && IsCjk(0x3002) && !IsCjk(0x2E7F),
        "1 CJK-Satzzeichen 3000–303F (。, ideographisches Leerzeichen)");
  check(IsCjk(0x3041) && IsCjk(0x30A2) && IsCjk(0x31F0) && IsCjk(0xFF71) && IsCjk(0x1B000),
        "1 Hiragana, Katakana, phonetische Ergänzung, Halbbreite Katakana, Kana-Ergänzung");
  check(IsCjk(0xAC00) && IsCjk(0xD7A3) && IsCjk(0x1100) && IsCjk(0x3131) && IsCjk(0xA960) && IsCjk(0xD7FB),
        "1 Hangul: Silben, Jamo, Kompatibilitäts-Jamo, Ext A/B");
  check(IsCjk(0x3105) && IsCjk(0x31A0) && IsCjk(0x2E80) && IsCjk(0x2F00) && IsCjk(0xF900) && IsCjk(0xFE30),
        "1 Bopomofo, Radikale, Kompatibilitätsideogramme und -formen");
  check(IsCjk(0xFF0C) && IsCjk(0xFF21) && IsCjk(0xFFEF) && !IsCjk(0xFFF0) && !IsCjk(0xFFFD) && !IsCjk(0xFEFF),
        "1 Vollbreite FF00–FFEF gesperrt, FFFD/BOM frei");
  check(IsCjk(0x20000) && IsCjk(0x2A6DF) && IsCjk(0x30000) && IsCjk(0x3FFFF) && !IsCjk(0x40000),
        "1 Ebenen 2/3 (Ext B–I)");
  check(!IsCjk('A') && !IsCjk(0xE4) && !IsCjk(0x2014) && !IsCjk(0x0416) && !IsCjk(0x0627) && !IsCjk(0x0E01) &&
            !IsCjk(0x1EB9) && !IsCjk(0x1F600) && !IsCjk(0x1F200) && !IsCjk(0xA000) && !IsCjk(0xFE0F) &&
            !IsCjk(0x10FFFF),
        "1 frei: ASCII, ä, —, Kyrillisch, Arabisch, Thai, Vietnamesisch, Emoji, Yi, VS16");

  // 2. Whole tokens
  check(Banned("中") && Banned("中文") && Banned(" 的") && Banned("Hello世界") && Banned("。") && Banned("，") &&
            Banned("の") && Banned("カタカナ") && Banned("한국어") && Banned("ㄅ") && Banned("ｱ") && Banned("Ａ") &&
            Banned("𠀀") && Banned("：") && Banned("（"),
        "2 gesperrt: Han, Kana, Hangul, Bopomofo, CJK-/Vollbreite-Satzzeichen, Ext B");
  check(!Banned("Hello") && !Banned(" Größe") && !Banned("—") && !Banned("“") && !Banned("Привет") &&
            !Banned("مرحبا") && !Banned("สวัสดี") && !Banned("Tiếng Việt") && !Banned("😀") && !Banned("❤️") &&
            !Banned("\xEF\xBB\xBF") && !Banned("\xEF\xBF\xBD") && !Banned("") && !Banned("<|im_end|>") &&
            !Banned("→") && !Banned("€"),
        "2 frei: Latein, Umlaute, Gedankenstrich, Anführung, Kyrillisch, Arabisch, Thai, Vietnamesisch, Emoji, "
        "BOM, U+FFFD, Spezial-Token, Pfeil, Euro");

  // 3. Byte pieces
  check(Banned("\xE4") && Banned("\xE4\xB8") && Banned("ab\xE5") && Banned("x\xE5\x8F") && Banned("\xEA") &&
            Banned("\xEB") && Banned("\xED") && Banned("\xE3") && Banned("\xE9\xBF"),
        "3 Präfix am Ende, nur CJK-Ergänzungen (E3–E9, EA–ED): gesperrt");
  check(Banned("\xE1") && Banned("\xE2") && Banned("\xEF") && Banned("\xF0") && Banned("\xF0\xA0") &&
            Banned("\xE1\x84") && Banned("\xE2\xBA") && Banned("\xEF\xBC") && Banned("\xF0\x9B\x80"),
        "3 Präfix mit mindestens einer CJK-Ergänzung (E1, E2, EF, F0): gesperrt");
  check(!Banned("\xE2\x80") && !Banned("\xE1\xBA") && !Banned("\xF0\x9F") && !Banned("\xF0\x9F\x98") &&
            !Banned("\xC3") && !Banned("a\xD0") && !Banned("\xE0") && !Banned("\xF4") && !Banned("\xEF\xBB"),
        "3 Präfix ohne CJK-Ergänzung (U+2000–203F, Vietnamesisch, Emoji, 2-Byte, E0, F4, BOM): frei");
  check(!Banned("\xB8\xAD") && !Banned("\xAD") && !Banned("\x80\x80 text") && Banned("\xAD文") &&
            Banned("\xAD\xE6"),
        "3 Folgebytes am Anfang zählen nicht; was danach kommt, schon");
  check(!Banned("a\x80" "b") && !Banned("\xFF\xFE") && !Banned("\xE4x") && !Banned("\xC0\xAF"),
        "3 ungültige Bytes mitten im Token: kein Zeichen");

  // 4. Every split of every CJK character: the piece with the start of the character is banned (exhaustive).
  long chars = 0, leaks = 0;
  for (char32_t cp = 0x80; cp <= 0x3FFFF; ++cp) {
    if ((cp >= 0xD800 && cp <= 0xDFFF) || !IsCjk(cp)) continue;
    ++chars;
    const std::string u = Utf8(cp);
    for (std::size_t cut = 1; cut <= u.size(); ++cut) {
      // Piece: arbitrary ASCII lead-in + the first `cut` bytes of the character (rest in the next token)
      if (!Banned("ab" + u.substr(0, cut))) ++leaks;
    }
  }
  check(chars > 100000 && leaks == 0, "4 jedes Präfix jedes CJK-Zeichens gesperrt (alle Zerlegungen)");
  // Counter-check: characters outside whose prefixes have no CJK completion stay free
  long free_ok = 0, free_all = 0;
  for (char32_t cp : {U'ä', U'—', U'€', U'→', U'ệ', U'😀', U'Ж', U'ا'}) {
    const std::string u = Utf8(cp);
    ++free_all;
    if (!Banned(u)) ++free_ok;
  }
  check(free_ok == free_all, "4 ganze Nicht-CJK-Zeichen frei");

  // 5. Mask over a toy vocabulary
  const std::vector<std::string> vocab = {"a", "中", "\xE4", " the", "，", "\xE2\x80", "😀", "\xB8\xAD"};
  std::size_t count = 0;
  const auto mask = tobu::scripts::CjkMask(vocab.size(), [&](std::size_t id) { return vocab[id]; }, &count);
  check(count == 3 && mask == std::vector<std::uint8_t>{0, 1, 1, 0, 1, 0, 0, 0}, "5 CjkMask und Zählung");

  std::printf("FEHLER: %d\n", fails);
  return fails == 0 ? 0 : 1;
}
