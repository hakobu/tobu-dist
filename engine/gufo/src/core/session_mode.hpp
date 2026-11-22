// Parts from gufo (gufo-org/gufo v0.34.0 2845939b, MIT); see THIRD_PARTY.md. Changes: tobu.
#ifndef GUFO_CORE_SESSION_MODE_HPP_
#define GUFO_CORE_SESSION_MODE_HPP_

namespace gufo::core {

/// Per-session execution choice, independent of resident model capabilities.
enum class SessionMode { kAutoregressive, kSpeculative };

}  // namespace gufo::core

#endif  // GUFO_CORE_SESSION_MODE_HPP_
