// TOB-59: escape hatch of the native tool grammar (gufo JsonConstraint::WithTools, ToolSyntax::unknown_name_escape):
// a name in no tool list is not bent onto a listed one. Without GPU, without model. From
// engine (ICU as in test_sampling_ban.cpp):
//
//   clang++ -std=c++20 -O1 -I gufo -I gufo/include -I src -I . -I $ICU/include tobu/test_tool_escape.cpp \
//     gufo/src/core/json_constraint.cpp gufo/src/core/json_schema_lexeme.cpp gufo/src/core/json_schema_regex.cpp \
//     -L $ICU/lib -licuuc -licui18n -o /tmp/test_tool_escape && /tmp/test_tool_escape
//
// Checked: unknown name -> header allowed, then end (complete); without the escape the same header is rejected
// (counter-check: bending); a listed name keeps its argument grammar (not complete after the header, call
// all the way through); a prefix of a listed name is unknown; a named tool (without escape) stays strict.
#include <cstdio>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/core/json_constraint.hpp"

namespace gs = gufo::sampling;
using JC = gs::JsonConstraint;

static int fails = 0;

static void check(const char* name, bool ok) {
  std::printf("%s %s\n", ok ? "ok    " : "FEHLER", name);
  if (!ok) ++fails;
}

// State after the text; empty = rejected.
static JC::State run(const JC& g, const std::string& text) {
  auto st = g.Start();
  for (const unsigned char c : text) {
    if (st.empty()) break;
    st = g.Advance(st, c);
  }
  return st;
}

static std::shared_ptr<const JC> tools(bool escape, bool required = false) {
  const auto bash = gufo::json::parse(
      R"({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]})");
  const auto plan = gufo::json::parse(
      R"({"type":"object","properties":{"steps":{"type":"string"}},"required":["steps"]})");
  std::vector<JC::Tool> ts;
  ts.emplace_back("Bash", JC::ToolParameters(bash, false, JC::ToolFormat::kQwen));
  ts.emplace_back("mcp__work-board__plan_steps", JC::ToolParameters(plan, false, JC::ToolFormat::kQwen));
  JC::ToolSyntax syntax = JC::Syntax(JC::ToolFormat::kQwen);
  syntax.unknown_name_escape = escape;
  return JC::WithTools(nullptr, std::move(ts), required, true, syntax);
}

int main() {
  const auto g = tools(true);
  const auto off = tools(false);
  const std::string head = "Ich rufe es auf.\n<tool_call>\n<function=";

  auto st = run(*g, head + "mcp__work-board__plan_stepz>\n");
  check("unbekannter Name: Kopf erlaubt und vollständig (Ende)", !st.empty() && g->Complete(st));
  st = run(*g, head + "mcp__other__thing>");
  check("unbekannter Name, nur '>': vollständig", !st.empty() && g->Complete(st));
  st = run(*g, head + "mcp__other__thing>\n<parameter=");
  check("unbekannter Name: danach nichts mehr (keine Argumente)", st.empty());
  st = run(*off, head + "mcp__other__thing>");
  check("Gegenprobe ohne Ausweg: unbekannter Name abgelehnt (Umbiegen)", st.empty());

  st = run(*g, head + "Bash>\n");
  check("gelisteter Name: nach dem Kopf nicht vollständig (Argumente folgen)", !st.empty() && !g->Complete(st));
  st = run(*g, head + "Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>");
  check("gelisteter Aufruf ganz: vollständig", !st.empty() && g->Complete(st));
  st = run(*g, head + "Bash>\n<parameter=nope>\n");
  check("gelisteter Name: unbekannter Parameter weiter abgelehnt", st.empty());
  st = run(*g, head + "Bas>");
  check("Präfix eines gelisteten Namens gilt als unbekannt", !st.empty() && g->Complete(st));
  st = run(*g, head + "Bash2>");
  check("Verlängerung eines gelisteten Namens gilt als unbekannt", !st.empty() && g->Complete(st));
  st = run(*g, head + ">");
  check("leerer Name abgelehnt", st.empty());

  const auto req = tools(true, true);
  st = run(*req, "<tool_call>\n<function=unknown_tool>\n");
  check("tool_choice required: Ausweg beendet ebenfalls (Server meldet Fehler)", !st.empty() && req->Complete(st));

  std::printf("FEHLER: %d\n", fails);
  return fails ? 1 : 0;
}
