// Test of the script ban in the gufo sampler (SamplingConfig::banned) without GPU, without model; macOS/Linux, system compiler.
// From engine (the grammar parts need ICU: brew install icu4c; ICU = the
// prefix containing include/unicode, e.g. ICU=/usr/local/opt/icu4c@78):
//
//   clang++ -std=c++20 -O1 -ffp-contract=off -I gufo -I gufo/include -I src -I . -I $ICU/include \
//     tobu/test_sampling_ban.cpp gufo/src/core/sampling.cpp gufo/src/core/sampling_logit_row.cpp \
//     gufo/src/core/json_constraint.cpp gufo/src/core/json_schema_lexeme.cpp gufo/src/core/json_schema_regex.cpp \
//     -L $ICU/lib -licuuc -licui18n -o /tmp/test_sampling_ban && /tmp/test_sampling_ban
//
// Checked (without tool grammar):
//  1. Mask without banned tokens = no ban: same distributions and same draws (greedy, penalties, top-k/
//     top-p/min-p, linear, top-p over > 1024 tokens, min_keep).
//  2. Ban = sampler without ban on the hand-masked row (−inf): same distributions/draws, never a
//     banned token; greedy takes the best allowed one, also with penalties.
//  3. CanSelectArgmax: banned -> false (GPU argmax falls back to the host path), allowed -> true.
//  4. Compact MTP verification (DistributionFromCandidates, candidates like the GPU: top-64 unpenalized + penalized) =
//     whole row; too few allowed candidates -> CompactCandidatesCover false (load whole rows).
//  5. VerifyMtpProposal (whole row, compact): a banned proposal is never accepted, the correction is allowed.
//  6. MTP draft (SampleMtpProposal): without banned proposals; if all candidates are banned, unbanned (no error).
//  7. Forced takes precedence: thinking-budget end made of banned tokens is emitted, CanSelectArgmax true.
//  8. If no allowed finite token remains, the row applies without the ban (no error).
//  9. BuildDistribution with ban; WithoutConstraint keeps the ban.
// 10. Grammar (schema enum ["中"]): where it also admits allowed tokens, the ban applies; where it admits only banned
//     ones (grammar literal), the grammar takes precedence, greedy and sampled, no error. Output {"a":"中"}.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "mtp_sampling.hpp"
#include "src/core/json.hpp"
#include "src/core/json_constraint.hpp"
#include "src/core/sampling.hpp"

namespace gs = gufo::sampling;
namespace fni = gufo::models::flash_next_iq;
using Mask = std::shared_ptr<const std::vector<std::uint8_t>>;

namespace {

int fails = 0;

void check(bool ok, const char* name) {
  std::printf("%s %s\n", ok ? "ok    " : "FEHLER", name);
  if (!ok) ++fails;
}

std::vector<float> Row(std::size_t vocab, std::uint64_t seed, float scale = 3.0F) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> n(0.0F, scale);
  std::vector<float> row(vocab);
  for (auto& x : row) x = n(rng);
  return row;
}

Mask MakeMask(std::size_t vocab, const std::vector<gs::TokenId>& ids, std::size_t every = 0) {
  auto m = std::make_shared<std::vector<std::uint8_t>>(vocab, 0);
  for (auto id : ids) (*m)[id] = 1;
  if (every)
    for (std::size_t i = 0; i < vocab; i += every) (*m)[i] = 1;
  return m;
}

std::vector<float> Masked(std::vector<float> row, const Mask& m) {
  for (std::size_t i = 0; i < row.size(); ++i)
    if ((*m)[i]) row[i] = -std::numeric_limits<float>::infinity();
  return row;
}

bool Same(const gs::SamplingDistribution& a, const gs::SamplingDistribution& b) {
  const auto x = a.entries(), y = b.entries();
  if (x.size() != y.size()) return false;
  for (std::size_t i = 0; i < x.size(); ++i)
    if (x[i].token != y[i].token || x[i].value != y[i].value) return false;
  return true;
}

bool AnyBanned(const gs::SamplingDistribution& d, const Mask& m) {
  for (const auto& e : d.entries())
    if ((*m)[e.token]) return true;
  return false;
}

std::vector<gs::TokenId> TopTokens(const std::vector<float>& row, std::size_t k) {
  std::vector<gs::TokenId> ids(row.size());
  for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<gs::TokenId>(i);
  std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](auto a, auto b) {
    return row[a] == row[b] ? a < b : row[a] > row[b];
  });
  ids.resize(k);
  return ids;
}

struct Cfg {
  const char* name;
  gs::SamplingConfig c;
};

std::vector<Cfg> Configs() {
  std::vector<Cfg> out;
  gs::SamplingConfig c;
  c.seed = 7;
  out.push_back({"greedy", c});
  auto p = c;
  p.presence_penalty = 1.5F;
  p.frequency_penalty = 0.3F;
  p.repeat_penalty = 1.1F;
  out.push_back({"greedy+Strafen", p});
  auto s = c;
  s.temperature = 0.8F;
  s.top_k = 40;
  s.top_p = 0.95F;
  s.min_p = 0.05F;
  out.push_back({"top-k/top-p/min-p", s});
  auto sp = s;
  sp.presence_penalty = 1.5F;
  out.push_back({"top-k+Strafen", sp});
  auto l = c;
  l.temperature = 1.0F;
  out.push_back({"linear", l});
  auto tp = c;
  tp.temperature = 0.7F;
  tp.top_p = 0.9F;
  out.push_back({"top-p ganze Zeile", tp});
  auto mk = c;
  mk.temperature = 0.6F;
  mk.min_p = 0.2F;
  mk.min_keep = 3;
  out.push_back({"min-p+min_keep", mk});
  return out;
}

// Draws with acceptance (penalty history grows along), one new row per step.
std::vector<gs::TokenId> Draws(gs::SamplerState st, std::size_t vocab, int n, const Mask& hand = nullptr) {
  std::vector<gs::TokenId> out;
  for (int i = 0; i < n; ++i) {
    auto row = Row(vocab, 1000 + static_cast<std::uint64_t>(i));
    if (hand) row = Masked(row, hand);
    out.push_back(st.Sample(row));
    st.Accept(out.back());
  }
  return out;
}

}  // namespace

int main() {
  constexpr std::size_t kVocab = 5000;
  const std::vector<gs::TokenId> history = {3, 17, 42, 99, 1234};
  const auto row = Row(kVocab, 1);
  const auto top = TopTokens(row, 200);
  // Ban: the 30 best tokens, every 7th token and a few from the history
  std::vector<gs::TokenId> ban_ids(top.begin(), top.begin() + 30);
  ban_ids.push_back(17);
  ban_ids.push_back(1234);
  const Mask ban = MakeMask(kVocab, ban_ids, 7);
  const Mask none = MakeMask(kVocab, {});

  // 1 + 2
  bool same_null = true, same_hand = true, never = true;
  for (auto& [name, base] : Configs()) {
    auto with_none = base;
    with_none.banned = none;
    auto with_ban = base;
    with_ban.banned = ban;
    const gs::SamplerState a(base, history), b(with_none, history), c(with_ban, history);
    const auto da = a.Distribution(row), db = b.Distribution(row);
    const auto ra = Draws(a, kVocab, 300), rb = Draws(b, kVocab, 300);
    if (!Same(da, db) || ra != rb) {
      std::printf("   (1: %s)\n", name);
      same_null = false;
    }
    const auto dc = c.Distribution(row), dh = a.Distribution(Masked(row, ban));
    const auto rc = Draws(c, kVocab, 300), rh = Draws(a, kVocab, 300, ban);
    if (!Same(dc, dh) || rc != rh) {
      std::printf("   (2: %s)\n", name);
      same_hand = false;
    }
    for (auto t : rc) never = never && !(*ban)[t];
    never = never && !AnyBanned(dc, ban);
  }
  check(same_null, "1 Maske ohne gesperrte Tokens: Verteilungen und Ziehungen wie ohne Sperre (7 Einstellungen)");
  check(same_hand, "2 Sperre = von Hand maskierte Zeile: Verteilungen und Ziehungen gleich (7 Einstellungen)");
  check(never, "2 nie ein gesperrtes Token gezogen oder in der Verteilung");
  {
    gs::SamplingConfig g;
    g.banned = ban;
    gs::SamplerState st(g, history);
    gs::TokenId best_allowed = 0;
    for (auto t : top)
      if (!(*ban)[t]) {
        best_allowed = t;
        break;
      }
    check(st.Sample(row) == best_allowed && (*ban)[top[0]], "2 greedy: bestes erlaubtes Token statt gesperrtem Argmax");
  }

  // 3
  {
    gs::SamplingConfig g;
    g.banned = ban;
    const gs::SamplerState st(g, history);
    gs::SamplingConfig plain;
    const gs::SamplerState sp(plain, history);
    gs::TokenId allowed = 0;
    while ((*ban)[allowed]) ++allowed;
    check(!st.CanSelectArgmax(top[0]) && st.CanSelectArgmax(allowed) &&
              sp.CanSelectArgmax(top[0]) && !g.can_use_unmodified_argmax(),
          "3 CanSelectArgmax: gesperrt false, erlaubt true, ohne Sperre unverändert");
  }

  // 4: compact candidates like the GPU (VerificationCandidates): top-64 of the unpenalized by raw logit, plus all
  // penalized ones with raw logit.
  {
    gs::SamplingConfig s;
    s.temperature = 0.8F;
    s.top_k = 20;
    s.top_p = 0.9F;
    s.presence_penalty = 1.0F;
    s.seed = 3;
    auto compact_for = [&](const gs::SamplerState& st, const std::vector<float>& r) {
      std::vector<gs::LogitCandidate> c;
      std::vector<float> unpen = r;
      for (const auto& p : st.penalties()) unpen[p.token] = -std::numeric_limits<float>::infinity();
      for (auto t : TopTokens(unpen, 64)) c.push_back({t, r[t]});
      for (const auto& p : st.penalties()) c.push_back({p.token, r[p.token]});
      return c;
    };
    bool equal = true, covered = true;
    for (std::uint64_t seed = 0; seed < 50; ++seed) {
      auto cfg = s;
      cfg.banned = ban;
      gs::SamplerState st(cfg, history);
      for (auto t : {5U, 17U, 2000U}) st.Accept(t);
      const auto r = Row(kVocab, 500 + seed);
      // only the best 10 banned (ban bans every 7th token, so enough allowed ones remain among the top-64)
      const auto c = compact_for(st, r);
      if (!st.CompactCandidatesCover(c)) {
        covered = false;
        continue;
      }
      equal = equal && Same(st.DistributionFromCandidates(c), st.Distribution(r));
    }
    check(covered && equal, "4 kompakt mit Sperre = ganze Zeile (50 Zeilen, Strafen, top-k 20, top-p)");
    // the best 60 of a row banned: fewer than 20 allowed ones remain among the GPU top-64
    const auto r = Row(kVocab, 77);
    const auto t = TopTokens(r, 60);
    auto cfg = s;
    cfg.banned = MakeMask(kVocab, std::vector<gs::TokenId>(t.begin(), t.end()));
    gs::SamplerState st(cfg, history);
    const auto c = compact_for(st, r);
    bool threw = false;
    try {
      (void)st.DistributionFromCandidates(c);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    gs::SamplerState plain(s, history);
    check(!st.CompactCandidatesCover(c) && threw && plain.CompactCandidatesCover(c),
          "4 zu wenige erlaubte Kandidaten: Cover false (ganze Zeilen), DistributionFromCandidates verweigert");
  }

  // 5: verification of a banned proposal
  {
    gs::SamplingConfig s;
    s.temperature = 0.8F;
    s.top_k = 40;
    s.seed = 11;
    s.banned = ban;
    bool ok = true;
    for (int i = 0; i < 200; ++i) {
      gs::SamplerState st(s, history);
      st.SetRngState(1234 + static_cast<std::uint64_t>(i));
      fni::MtpProposal p;
      p.size = 2;
      p.ids[0] = top[0];  // banned
      p.ids[1] = top[30];
      p.probabilities[0] = 0.5F;
      p.probabilities[1] = 0.5F;
      p.token = top[0];
      p.probability = 0.5F;
      const auto v = fni::VerifyMtpProposal(row, p, st);
      ok = ok && !v.accepted && !(*ban)[v.token];
      std::vector<gs::LogitCandidate> c;
      for (auto t : TopTokens(row, 64)) c.push_back({t, row[t]});
      gs::SamplerState sc(s, history);
      sc.SetRngState(1234 + static_cast<std::uint64_t>(i));
      if (sc.CompactCandidatesCover(c)) {
        const auto w = fni::VerifyMtpProposal(c, kVocab, p, sc);
        ok = ok && !w.accepted && !(*ban)[w.token] && w.token == v.token;
      }
    }
    check(ok, "5 VerifyMtpProposal: gesperrter Vorschlag nie angenommen, Korrektur erlaubt, kompakt = ganze Zeile");
  }

  // 6: MTP draft
  {
    gs::SamplingConfig s;
    s.temperature = 0.8F;
    s.top_k = 40;
    s.seed = 5;
    s.banned = ban;
    const gs::SamplerState st(s, history);
    fni::MtpCandidateLogits c;
    const auto t = TopTokens(row, 64);
    for (std::size_t i = 0; i < 64; ++i) {
      c.ids[i] = t[i];
      c.logits[i] = row[t[i]];
    }
    c.size = 64;
    std::uint64_t rng = 9;
    bool clean = true;
    for (int i = 0; i < 100; ++i) {
      const auto p = fni::SampleMtpProposal(c, st, &rng);
      for (std::size_t k = 0; k < p.size; ++k) clean = clean && !((*ban)[p.ids[k]] && p.probabilities[k] > 0);
      clean = clean && !(*ban)[p.token];
    }
    c.size = 20;  // the best 20: all banned (ban contains the 30 best)
    bool no_throw = true;
    try {
      const auto p = fni::SampleMtpProposal(c, st, &rng);
      no_throw = p.size > 0;
    } catch (const std::exception&) {
      no_throw = false;
    }
    check(clean && no_throw, "6 Entwurf ohne gesperrte Vorschläge; alle Kandidaten gesperrt -> ungesperrt, kein Fehler");
  }

  // 7: forced thinking-budget end made of banned tokens
  {
    auto budget = std::make_shared<gs::ReasoningBudget>();
    budget->tokens = 0;
    budget->forced = {top[0], top[1]};  // both banned
    gs::SamplingConfig g;
    g.banned = ban;
    g.reasoning_budget = budget;
    gs::SamplerState st(g, {});
    const bool can = st.CanSelectArgmax(top[0]);
    const auto a = st.Sample(row);
    st.Accept(a);
    const auto b = st.Sample(row);
    st.Accept(b);
    const auto c = st.Sample(row);
    check(can && a == top[0] && b == top[1] && !(*ban)[c], "7 erzwungen geht vor: Budget-Ende aus gesperrten Tokens");
  }

  // 8: only banned tokens finite
  {
    std::vector<float> r(kVocab, -std::numeric_limits<float>::infinity());
    for (gs::TokenId i = 0; i < kVocab; i += 7) r[i] = static_cast<float>(i % 13);
    gs::SamplingConfig g;
    g.banned = ban;
    gs::SamplerState gr(g, {});
    auto s = g;
    s.temperature = 0.8F;
    s.top_k = 10;
    gs::SamplerState sa(s, {});
    bool ok = true;
    try {
      ok = (*ban)[gr.Sample(r)] && (*ban)[sa.Sample(r)];
    } catch (const std::exception&) {
      ok = false;
    }
    check(ok, "8 kein erlaubtes endliches Token: Zeile ohne Sperre, kein Fehler");
  }

  // 9
  {
    gs::SamplingConfig s;
    s.temperature = 0.9F;
    s.top_k = 50;
    s.banned = ban;
    const auto d = gs::BuildDistribution(row, s, history, {});
    gs::SamplerState st(s, history);
    check(!AnyBanned(d, ban) && st.WithoutConstraint().config().banned == ban && !st.WithoutBan().config().banned,
          "9 BuildDistribution mit Sperre; WithoutConstraint behält sie, WithoutBan nicht");
  }

  // 10: grammar forces a banned literal
  try {
    const std::vector<std::string> pieces = {"{\"a\":\"", "中", "\"}", "{\"a\":\"中\"}", "{\"a\":\"中", "<e>", "x", " "};
    auto vocab = std::make_shared<const gs::ConstraintVocabulary>(
        static_cast<std::uint32_t>(pieces.size()),
        [&](std::uint32_t id) { return gs::ConstraintVocabulary::Piece{pieces[id], id == 5}; });
    const auto schema = gufo::json::parse(
        R"({"type": "object", "properties": {"a": {"type": "string", "enum": ["中"]}}, "required": ["a"],
            "additionalProperties": false})");
    auto tc = std::make_shared<gs::TokenConstraint>();
    tc->grammar = gs::JsonConstraint::Compile(schema, true);
    tc->vocabulary = vocab;
    const Mask cjk = MakeMask(pieces.size(), {1, 3, 4});  // like TokenHasCjk
    const std::vector<float> r = {10.0F, 50.0F, 0.0F, 50.0F, 50.0F, 1.0F, 50.0F, -30.0F};  // banned ones first
    bool ok = true;
    for (const float temperature : {0.0F, 0.7F}) {
      gs::SamplingConfig g;
      g.temperature = temperature;
      g.seed = 1;
      g.min_p = temperature > 0 ? 0.01F : 0.0F;
      g.constraint = tc;
      g.banned = cjk;
      gs::SamplerState st(g, {});
      std::vector<gs::TokenId> out;
      try {
        for (int i = 0; i < 4; ++i) {
          out.push_back(st.Sample(r));
          if (out.back() == 5) break;
          st.Accept(out.back());
        }
      } catch (const std::exception& e) {
        std::printf("   (10: %s)\n", e.what());
        ok = false;
      }
      ok = ok && out == std::vector<gs::TokenId>{0, 1, 2, 5};
      if (out != std::vector<gs::TokenId>{0, 1, 2, 5}) {
        std::printf("   (10: T=%.1f:", temperature);
        for (auto t : out) std::printf(" %u", t);
        std::printf(")\n");
      }
    }
    check(ok, "10 Grammatik: Sperre wirkt neben erlaubten Tokens, ein nur gesperrt erfüllbares Literal geht vor");
  } catch (const std::exception& e) {
    std::printf("   (10: %s)\n", e.what());
    check(false, "10 Grammatik aufbauen");
  }

  std::printf("FEHLER: %d\n", fails);
  return fails == 0 ? 0 : 1;
}
