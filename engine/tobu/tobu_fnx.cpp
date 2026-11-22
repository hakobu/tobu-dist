// tobu-fnx: compute engine of the edition qwen38-flash-next-gfx1151 as a child process of the tobu agent server
// (server/agent/engine_flashnext.py). Copyright (c) 2026 hakobu, Apache-2.0 (tobu).
//
// The engine underneath (src/: Model/Session/Executor/Kernels) comes from gufo (gufo-org/gufo 2845939b, MIT) or the
// flash-next-iq-gufo-extension (storm1er 9ce08ce, MIT); see THIRD_PARTY_NOTICES.md and the file headers.
//
// Protocol (stdin -> stdout, little endian, int32), as in run/tobu_llama.cpp:
//   Request  cmd, n, data[n]      Response  status (0 = ok, < 0 = error), n, data[n]
//   0 INFO               -> vocab, max_context, has_mtp, eos, position, state_id[4]
//                           state_id = first 16 bytes of the hash over the state-determining sources (CMake,
//                           state_id.h): equal -> SSD snapshots from another build are valid (server fingerprint)
//   1 RESET              -> -                         clear state
//   2 SYNC  ids...       -> position, us              prompt (common prefix with the state is kept, rest prefilled)
//   3 STEP  max_tokens   -> stop, n, tok[n], cycles, drafted, accepted, us
//                           one decode round (greedy; with MTP: draft + verification + acceptance in the engine)
//   4 EVAL  token        -> position                  append token without selection (forced tokens, thinking budget)
//   5 LOGITS             -> float32[vocab]            logits of the current row (sampling, grammar)
//   6 SAVE  slot         -> bytes_lo, bytes_hi        state into host memory (slot)
//   7 LOAD  slot         -> position
//   8 DROP  slot         -> -
//  12 CORPUS_SET id, tokens... -> docs, tokens        workspace document (copy source, E4)
//  13 CORPUS_DROP id (-1 all) -> docs, tokens
//  14 SELECT k             -> position                  slot (session) for the following commands (--sessions N)
//  11 PEEK               -> token                     first token of the next round (without feeding it)
//  10 CONFIG len, json   -> -                         grammar/thinking budget/sampling/script ban per request
//                           (in the engine)
//  15 STEPB k, (slot, max_tokens)[k] -> per request: status, stop, n, tok[n], cycles, drafted, accepted; then us
//                           one decode round for k slots batched (S2): MTP slots via gufo's Session::DecodeBatch,
//                           single-token slots via EvaluateBatch; per slot bit-identical to STEP (gufo verifies this exactly)
//  16 USAGE reset        -> layers, experts, u32[3][layers][experts]   expert usage (TOBU_FNX_EXPERT_USAGE=1;
//                           otherwise layers = experts = 0). Phases 0 prefill, 1 decode/verify rows, 2 MTP head. Process-wide
//                           (all slots), reset = 1 zeroes it afterwards.
//  17 TEACH ns, (b, e)[ns], ids... -> rows, hc_dim, hidden, pos[rows], fp16 r[rows][hc_dim], fp16 y[rows][hidden]
//                           (ns < 0: stage B with |ns| spans, additionally fp16 a[rows][hc_dim], m[rows][hidden],
//                           o[rows][hidden])
//                           MTP-head training: reset slot, prefill prompt with the full MTP body, for MTP positions in
//                           the spans the residual before hc_head (r) and its output (y); slot reset afterwards
//  18 SCORE ids...       -> float32 logp[n-1], int32 argmax_ok[n-1]   quality measurement (perplexity): reset slot, per
//                           position log p(ids[i+1] | ids[..i]) from the logits of the plain path, feed token; plus whether the
//                           argmax hits the next token. Slot reset afterwards.
//  19 IMAGE id, detail, w, h, max_pixels, rgb[w*h*3 bytes, padded to int32] -> gh, gw, tokens, sha256[8]
//                           scale image (RGB8, decoded by the server) to the model grid (gufo ResizeImage, detail 0 auto /
//                           1 low, max_pixels 0 = 16.7 MP) and store it in the image store under id (LRU, TOBU_FNX_IMAGE_MB=64)
//  20 VPROMPT k, (id, offset)[k] -> prefix_length, delta   attach image layout to the selected slot (offset = first
//                           image token in the prompt, server expands <|image_pad|> to gh*gw); k = 0 -> without images. The
//                           embeddings are computed only by the next SYNC (lazy). Send before SYNC; LOAD resets the layout.
//  21 IMAGE_DROP id (-1 all) -> n (remaining images)
//  25 STATS               -> (lo, hi)[13]: cycles, drafted, accepted, lookup, lookup_accepted, lookup_cycles,
//                           corpus, corpus_accepted, corpus_cycles, store, store_accepted, store_cycles, reserved
//                           draft counters of the selected slot since start (u64 as two int32); lookup* = all copies,
//                           corpus* = of these from workspace files, store* = from the answer store. The server computes per
//                           request the difference (benefit per source in usage.tobu)
//                           Vision tower (--mmproj): weights go to the GPU only with the first image, freed again after TOBU_FNX_VISION_IDLE_S
//                           (600) without an image (TOB-37).
//  22 PERSIST slot, len, path -> MiB          snapshot of storage slot `slot` to the SSD (gufo format with check header):
//                           materialize now, a background thread writes path.tmp and renames; DROP afterwards is ok
//  23 LOADFILE len, path     -> position         load snapshot file into the slot (like LOAD; gufo checks format/size)
//  24 PERSIST_WAIT           -> open writers      waits until all PERSIST writers are done (shutdown, tests)
//   9 SYNCCP nb, b[nb], slot[nb], ids... -> position, us, MiB[nb]   prompt like SYNC, checkpoints at grid points
//                           in the same pass (server prefix store; resume with LOAD + SYNC/SYNCCP)
//
// Start: tobu-fnx --model <gguf part 1> [--mtp <gguf>] [--ple <engram-gguf>] [--mmproj <mmproj-BF16.gguf>] [--context N] [--draft N]
// Check mode without protocol: tobu-fnx ... --ids <file: token ids, space-separated> --n <max> -> generated ids on stdout.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <poll.h>
#include <sys/stat.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unistd.h>

#include <hip/hip_runtime.h>

#include "engine.hpp"
#include "qwen/vision/prompt.hpp"
#include "src/core/crypto/sha256.hpp"
#include "tobu/corpus.hpp"
#include "state_id.h"   // generated (cmake/state_id.cmake): state identifier for INFO
#include "tobu/scripts.hpp"
#include "src/core/json.hpp"
#include "src/core/json_constraint.hpp"

namespace fni = gufo::models::flash_next_iq;
using gufo::core::SessionMode;

namespace {

bool rd(void* p, size_t n) { return fread(p, 1, n, stdin) == n; }

void reply(int32_t status, const int32_t* d, int32_t n) {
  int32_t h[2] = {status, n};
  fwrite(h, 4, 2, stdout);
  if (n > 0) fwrite(d, 4, static_cast<size_t>(n), stdout);
  fflush(stdout);
}

void reply_bytes(int32_t status, const void* d, size_t nbytes) {
  int32_t h[2] = {status, static_cast<int32_t>(nbytes / 4)};
  fwrite(h, 4, 2, stdout);
  if (nbytes) fwrite(d, 1, nbytes, stdout);
  fflush(stdout);
}

int32_t us_since(std::chrono::steady_clock::time_point t0) {
  return static_cast<int32_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
}

struct Args {
  std::string model, mtp, ids, splits, memory;
  uint32_t context = 65536, draft = 0, sessions = 1;
  std::string ple;   // TOB-40: n-gram table from a separate GGUF (engram)
  std::string mmproj;   // TOB-37: vision tower (mmproj-BF16.gguf); empty = text only
  float presence = 0.0F;  // only for comparisons with gufo's server default (thinking off: 1.5); tobu-exact = 0
  long n = 256;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto v = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (k == "--model") a.model = v();
    else if (k == "--mtp") a.mtp = v();
    else if (k == "--context") a.context = static_cast<uint32_t>(std::stoul(v()));
    else if (k == "--draft") a.draft = static_cast<uint32_t>(std::stoul(v()));
    else if (k == "--sessions") a.sessions = static_cast<uint32_t>(std::stoul(v()));
    else if (k == "--ple") a.ple = v();
    else if (k == "--mmproj") a.mmproj = v();
    else if (k == "--memory") a.memory = v();
    else if (k == "--ids") a.ids = v();
    else if (k == "--n") a.n = std::stol(v());
    else if (k == "--splits") a.splits = v();
    else if (k == "--presence-penalty") a.presence = std::stof(v());
  }
  return a;
}

// One decode round, greedy; the same computation path as gufo's server (runner.cpp DecodeStep):
//  * with MTP and max_tokens > 1: Session::DecodeStep (draft + verification + acceptance),
//  * otherwise gufo's base path (TextModelRunner::DecodeStep): pick a token from the logits (SamplerState::Sample),
//    stop at the stop token (EndsTurn, do not feed it), otherwise Session::Evaluate.
// Session::DecodeStep in autoregressive mode is a third path and does not compute bit-identically to the base path
// (E2 measurement 2026-11-19: 6 of 7 bench outputs differ), so never use it.
bool step(fni::Model& m, fni::Session& s, gufo::sampling::SamplerState& sampler, size_t max_tokens,
          fni::Session::DecodeResult* out, std::string* err) {
  out->tokens.clear();
  out->stop = false;
  if (m.HasMtp() && max_tokens > 1) {
    // DecodeStep adds the tokens to the sampler itself (FinishDecode); gufo's runner works on a copy and
    // adopts it afterwards; the same here, so that grammar/penalties see each token exactly once.
    auto working = sampler;
    if (!s.DecodeStep(max_tokens, working, out, err, /*stop_at_eos=*/true)) return false;
    sampler = std::move(working);
    return true;
  }
  const auto logits = s.Logits();
  if (logits.empty()) {
    *err = "keine Logits";
    return false;
  }
  const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
  out->tokens.push_back(token);
  if (m.IsStopToken(token)) {
    out->stop = true;
    return true;
  }
  sampler.Accept(static_cast<gufo::sampling::TokenId>(token));
  return s.Evaluate(token, err);
}

// Thinking budget with tobu semantics (TOB-32, server/agent/think_budget.py): after `tokens` generated thinking tokens,
// "\n</think>\n\n" is forced (gufo otherwise counts from <think> in the prompt and forces only "\n</think>"). If thinking ends
// by itself (</think>, <tool_call>), nothing is counted anymore. The server deducts the answer reserve (TOB-33) beforehand.
std::shared_ptr<const gufo::sampling::ReasoningBudget> make_budget(const fni::Model& m, int32_t tokens) {
  auto b = std::make_shared<gufo::sampling::ReasoningBudget>();
  b->tokens = static_cast<std::size_t>(tokens);
  for (auto t : m.Tokenize("\n</think>\n\n")) b->forced.push_back(static_cast<gufo::sampling::TokenId>(t));
  for (const char* e : {"</think>", "<tool_call>"}) {
    const auto ids = m.Tokenize(e);
    if (ids.size() == 1) b->ends.push_back(static_cast<gufo::sampling::TokenId>(ids[0]));
  }
  return b;  // prompt empty: counting starts at the first generated token; partial_utf8/blank off (tobu semantics)
}

// Tool grammar like gufo's server (text_model_runner.cpp ConstrainChatRequest, Qwen XML) from OpenAI tools.
// tool_choice: "auto" | "required" | "none" | {"function": {"name": ...}} (only this function, required, single).
// Structured output (response_format json_schema) like gufo's server (response_format.hpp): JsonConstraint::Compile with
// open_objects (llama.cpp semantics), not strict; with tools the base grammar of WithTools (then not parallel).
// `tools` may be null (schema only).
std::shared_ptr<gufo::sampling::TokenConstraint> make_constraint(
    const fni::Model& m, const gufo::json::Value* tools, const gufo::json::Value* choice, bool parallel, bool think,
    const gufo::json::Value* schema, std::string* err) {
  using JC = gufo::sampling::JsonConstraint;
  static std::shared_ptr<const gufo::sampling::ConstraintVocabulary> vocab;   // once per process (248k pieces)
  if (!vocab)
    vocab = std::make_shared<const gufo::sampling::ConstraintVocabulary>(m.VocabSize(), [&](std::uint32_t id) {
      return gufo::sampling::ConstraintVocabulary::Piece{m.TokenText(static_cast<int32_t>(id)),
                                                         m.IsStopToken(static_cast<int32_t>(id))};
    });
  std::shared_ptr<const JC> base;
  if (schema != nullptr && schema->is_object()) base = JC::Compile(*schema, false, true);
  const auto bind = [&](std::shared_ptr<const JC> g) -> std::shared_ptr<gufo::sampling::TokenConstraint> {
    if (!g) return nullptr;
    if (think) g = JC::WithReasoning(g, "</think>");
    auto tc = std::make_shared<gufo::sampling::TokenConstraint>();
    tc->grammar = g;
    tc->vocabulary = vocab;
    return tc;
  };
  if (tools == nullptr || !tools->is_array() || tools->empty()) return bind(base);
  bool required = false;
  std::string only;
  if (choice != nullptr && choice->is_string()) {
    if (choice->get_str() == "none") return bind(base);
    required = choice->get_str() == "required";
  } else if (choice != nullptr && choice->is_object()) {
    if (const auto* fn = choice->find("function")) only = fn->member_str("name");
    required = true;
    parallel = false;
  }
  std::vector<JC::Tool> ts;
  for (const auto& t : tools->items()) {
    const auto* fn = t.find("function");
    if (fn == nullptr) continue;
    const std::string name = fn->member_str("name");
    if (!only.empty() && name != only) continue;
    const auto* strict = fn->find("strict");
    const auto* params = fn->find("parameters");
    const auto empty = gufo::json::parse("{\"type\":\"object\"}");
    ts.emplace_back(name, JC::ToolParameters(params ? *params : empty, strict && strict->as_bool(),
                                             JC::ToolFormat::kQwen));
  }
  if (ts.empty()) {
    if (required) *err = "tool_choice verlangt ein Werkzeug, das nicht in tools steht";
    return required ? nullptr : bind(base);
  }
  // TOB-59: a name in no list is not bent onto a listed one; the output ends after the header and the
  // server reports "unknown tool". Not with a named tool_choice (there only that tool applies).
  JC::ToolSyntax syntax = JC::Syntax(JC::ToolFormat::kQwen);
  syntax.unknown_name_escape = only.empty();
  return bind(JC::WithTools(base, std::move(ts), required, parallel && !base, syntax));
}

// Script ban "no-CJK" (decision): mask over the whole vocabulary from
// the token texts (tobu/scripts.hpp, including byte pieces), once per process on first use; count on stderr.
// Stop tokens always stay free.
std::shared_ptr<const std::vector<std::uint8_t>> cjk_mask(const fni::Model& m) {
  static std::shared_ptr<const std::vector<std::uint8_t>> mask;
  if (!mask) {
    const auto t0 = std::chrono::steady_clock::now();
    std::size_t count = 0;
    auto v = tobu::scripts::CjkMask(m.VocabSize(), [&](std::size_t id) {
      const auto t = static_cast<int32_t>(id);
      return m.IsStopToken(t) ? std::string() : m.TokenText(t);
    }, &count);
    mask = std::make_shared<const std::vector<std::uint8_t>>(std::move(v));
    fprintf(stderr, "tobu-fnx: Schriften-Sperre cjk: %zu von %u Tokens gesperrt (%.0f ms)\n", count, m.VocabSize(),
            us_since(t0) / 1e3);
  }
  return mask;
}

// CONFIG field "ban_scripts": ["cjk"] -> ban mask (null = no ban). Unknown script -> error (CONFIG -3).
std::shared_ptr<const std::vector<std::uint8_t>> banned_for(const fni::Model& m, const gufo::json::Value* scripts) {
  if (scripts == nullptr || scripts->is_null()) return nullptr;
  if (!scripts->is_array()) throw std::invalid_argument("ban_scripts muss eine Liste sein");
  std::shared_ptr<const std::vector<std::uint8_t>> out;
  for (const auto& s : scripts->items()) {
    if (!s.is_string() || s.get_str() != "cjk")
      throw std::invalid_argument("ban_scripts: unbekannte Schrift (nur \"cjk\")");
    out = cjk_mask(m);  // only one script so far; further ones would be OR-ed in here
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);
  if (a.model.empty()) {
    fprintf(stderr, "usage: tobu-fnx --model <gguf> [--mtp <gguf>] [--context N] [--ids file --n N]\n");
    return 2;
  }
  fni::ModelOptions mo;
  mo.max_context = a.context;
  mo.mtp_model_path = a.mtp;
  mo.ple_path = a.ple;
  mo.vision_model_path = a.mmproj;
  if (a.draft) mo.max_draft_tokens = a.draft;
  // Cost model of the MTP drafts for batched decode (S2) like gufo's server (runner.hpp
  // DecodeConcurrencyForSessions): sessions, at most 8. Only changes draft lengths, never the tokens.
  mo.decode_concurrency = std::min<uint32_t>(std::max<uint32_t>(a.sessions, 1), 8);
  std::string err;
  const auto t_load = std::chrono::steady_clock::now();
  auto model = fni::Model::Load(a.model, mo, &err);
  if (!model) {
    fprintf(stderr, "tobu-fnx: Modell nicht geladen: %s\n", err.c_str());
    return 1;
  }
  const SessionMode mode = model->HasMtp() ? SessionMode::kSpeculative : SessionMode::kAutoregressive;
  if (!a.memory.empty()) {
    // --memory c1,c2,...: memory requirement by gufo's own calculation (MiB), without creating sessions.
    std::size_t free_b = 0, total_b = 0;
    (void)hipMemGetInfo(&free_b, &total_b);
    printf("{\"resident_mib\": %zu, \"deferred_scratch_mib\": %zu, \"gpu_free_mib\": %zu, \"gpu_total_mib\": %zu, "
           "\"mtp\": %d, \"sessions\": [", model->ResidentBytes() >> 20, model->DeferredScratchBytes() >> 20,
           free_b >> 20, total_b >> 20, model->HasMtp() ? 1 : 0);
    bool first = true;
    for (size_t pos = 0; pos < a.memory.size();) {
      size_t end = a.memory.find(',', pos);
      if (end == std::string::npos) end = a.memory.size();
      const uint32_t c = static_cast<uint32_t>(std::stoul(a.memory.substr(pos, end - pos)));
      pos = end + 1;
      printf("%s{\"context\": %u, \"speculative_mib\": %zu, \"autoregressive_mib\": %zu}", first ? "" : ", ", c,
             model->SessionBytes(SessionMode::kSpeculative, c) >> 20,
             model->SessionBytes(SessionMode::kAutoregressive, c) >> 20);
      first = false;
    }
    printf("]}\n");
    return 0;
  }
  // Slots (parallel requests): per slot a separate gufo session (state stays on the GPU) with its own sampler and
  // own CONFIG; SELECT k selects the slot for the following commands.
  struct Slot {
    std::unique_ptr<fni::Session> session;
    gufo::sampling::SamplingConfig cfg;                       // temperature 0 = greedy (tobu exact)
    std::shared_ptr<gufo::sampling::TokenConstraint> constraint;  // tool grammar (Qwen XML), otherwise empty
    int32_t think_budget{-1};                                 // thinking budget in tokens, -1 = none
    bool thinking{false};
    bool vision{false};                                       // image layout set (VPROMPT k > 0)
    gufo::sampling::SamplerState sampler;
  };
  std::vector<Slot> sl(std::max<uint32_t>(1, a.sessions));
  for (auto& S : sl) {
    S.session = model->CreateSession(mode, a.context, &err);
    if (!S.session) {
      fprintf(stderr, "tobu-fnx: Session nicht angelegt: %s\n", err.c_str());
      return 1;
    }
    S.cfg.presence_penalty = a.presence;   // --presence-penalty mimics gufo's server default (comparisons)
  }
  int32_t cur = 0;
  fprintf(stderr, "tobu-fnx: bereit in %.1f s (vocab %u, context %u, mtp %d, sessions %zu)\n", us_since(t_load) / 1e6,
          model->VocabSize(), a.context, model->HasMtp() ? 1 : 0, sl.size());
  auto make_sampler_for = [&](Slot& S, std::span<const int32_t> prompt) {
    auto c = S.cfg;
    c.constraint = S.constraint;
    if (S.thinking && S.think_budget >= 0) c.reasoning_budget = make_budget(*model, S.think_budget);
    return gufo::sampling::SamplerState(c, std::span<const gufo::sampling::TokenId>(
        reinterpret_cast<const gufo::sampling::TokenId*>(prompt.data()), prompt.size()));
  };
  for (auto& S : sl) S.sampler = make_sampler_for(S, {});
// Bind the command names to the current slot (session, sampler, cfg, ... as before the slots).
#define TOBU_BIND_SLOT(k)                                                                  \
  Slot& S_ = sl[k];                                                                         \
  auto& session = S_.session;                                                               \
  auto& sampler = S_.sampler;                                                               \
  auto& cfg = S_.cfg;                                                                       \
  auto& constraint = S_.constraint;                                                         \
  auto& think_budget = S_.think_budget;                                                     \
  auto& thinking = S_.thinking;                                                             \
  auto make_sampler = [&](std::span<const int32_t> prompt) { return make_sampler_for(S_, prompt); }; \
  (void)cfg; (void)constraint; (void)think_budget; (void)thinking;

  if (!a.ids.empty()) {  // check mode (slot 0)
    TOBU_BIND_SLOT(0)
    std::ifstream f(a.ids);
    std::vector<int32_t> ids;
    for (int32_t t; f >> t;) ids.push_back(t);
    const auto t0 = std::chrono::steady_clock::now();
    // --splits a,b,...: prefill in steps (Sync(prompt[:a]), Sync(prompt[:b]), ..., Sync(prompt)); checks whether
    // splitting the prefill changes the output.
    for (size_t pos = 0; !a.splits.empty() && pos < a.splits.size();) {
      size_t end = a.splits.find(',', pos);
      if (end == std::string::npos) end = a.splits.size();
      const size_t k = std::stoul(a.splits.substr(pos, end - pos));
      pos = end + 1;
      if (k == 0 || k >= ids.size()) continue;
      const auto ts = std::chrono::steady_clock::now();
      if (!session->Sync(std::span<const int32_t>(ids.data(), k), &err)) {
        fprintf(stderr, "tobu-fnx: Sync(%zu): %s\n", k, err.c_str());
        return 1;
      }
      fprintf(stderr, "tobu-fnx: Sync bis %zu: %.3f s\n", k, us_since(ts) / 1e6);
    }
    const auto t_last = std::chrono::steady_clock::now();
    const size_t from = session->Position();
    if (!session->Sync(ids, &err)) {
      fprintf(stderr, "tobu-fnx: Sync: %s\n", err.c_str());
      return 1;
    }
    if (!a.splits.empty())
      fprintf(stderr, "tobu-fnx: Rest %zu -> %zu (%zu Tokens): %.3f s (%.0f tok/s)\n", from, ids.size(),
              ids.size() - from, us_since(t_last) / 1e6, (ids.size() - from) / (us_since(t_last) / 1e6));
    const double pf = us_since(t0) / 1e6;
    sampler = make_sampler(ids);
    std::vector<int32_t> gen;
    fni::Session::DecodeResult r;
    const auto t1 = std::chrono::steady_clock::now();
    while (static_cast<long>(gen.size()) < a.n) {
      if (!step(*model, *session, sampler, static_cast<size_t>(a.n - static_cast<long>(gen.size())), &r, &err)) {
        fprintf(stderr, "tobu-fnx: DecodeStep: %s\n", err.c_str());
        return 1;
      }
      gen.insert(gen.end(), r.tokens.begin(), r.tokens.end());
      if (r.stop || r.tokens.empty()) break;
    }
    const double dc = us_since(t1) / 1e6;
    for (auto t : gen) printf("%d ", t);
    printf("\n");
    const auto& st = session->Statistics();
    fprintf(stderr, "tobu-fnx: prompt %zu, prefill %.3f s (%.0f tok/s), %zu tokens in %.3f s (%.2f tok/s), "
            "cycles %llu drafted %llu accepted %llu small_head %llu lookup %llu/%llu\n", ids.size(), pf,
            ids.size() / pf, gen.size(), dc, gen.size() > 1 ? (gen.size() - 1) / dc : 0.0,
            (unsigned long long)st.cycles, (unsigned long long)st.drafted, (unsigned long long)st.accepted,
            (unsigned long long)st.small_head_cycles, (unsigned long long)st.lookup_accepted,
            (unsigned long long)st.lookup);
    return 0;
  }

  std::map<int32_t, std::shared_ptr<fni::SessionSnapshot>> slots;   // shared: writer threads (PERSIST) co-own it
  tobu::Corpus corpus;                 // E4: workspace (client files) as a second copy source
  for (auto& S : sl) S.session->SetCorpus(&corpus);
  if (const char* cm = getenv("TOBU_FNX_CORPUS_MIN"))          // minimum length of corpus hits (greedy), default 12
    for (auto& S : sl) S.session->SetCorpusMinMatch(static_cast<size_t>(atol(cm)));
  // TOB-37 vision: image store (scaled, LRU by bytes) and idle release of the tower.
  namespace qv = gufo::models::qwen_iq::vision;
  const auto& encoder = model->VisionEncoder();
  struct StoredImage {
    std::shared_ptr<const gufo::core::Image> img;
    uint64_t used;
  };
  std::map<int32_t, StoredImage> images;
  uint64_t image_tick = 0;
  size_t image_bytes = 0;
  const size_t image_limit = static_cast<size_t>(getenv("TOBU_FNX_IMAGE_MB") ? atol(getenv("TOBU_FNX_IMAGE_MB")) : 64) << 20;
  const double vision_idle_s = getenv("TOBU_FNX_VISION_IDLE_S") ? atof(getenv("TOBU_FNX_VISION_IDLE_S")) : 600.0;
  auto vision_used = std::chrono::steady_clock::now();
  if (encoder) setvbuf(stdin, nullptr, _IONBF, 0);   // otherwise poll() below does not see buffered commands
  // Persistent prefix store: writer threads (only read the materialized host buffer, no GPU)
  std::mutex wmu;
  std::condition_variable wcv;
  int writers = 0;
  auto path_of = [&](const int32_t* w, int32_t nw) -> std::string {
    if (nw < 1 || w[0] < 0 || static_cast<int64_t>(w[0]) > 4 * static_cast<int64_t>(nw - 1)) return {};
    return std::string(reinterpret_cast<const char*>(w + 1), static_cast<size_t>(w[0]));
  };
  std::vector<int32_t> in, out;
  for (;;) {
    while (encoder && encoder->ResidentBytes() > 0) {   // tower on the GPU: release it when idle after vision_idle_s
      const double idle = std::chrono::duration<double>(std::chrono::steady_clock::now() - vision_used).count();
      if (idle >= vision_idle_s) {
        encoder->Release();
        fprintf(stderr, "tobu-fnx: Vision-Turm freigegeben (%.0f s ohne Bild)\n", idle);
        break;
      }
      pollfd pf{0, POLLIN, 0};
      const int ms = static_cast<int>(std::min(60.0, vision_idle_s - idle + 0.5) * 1000);
      if (poll(&pf, 1, ms) != 0) break;              // command available (or error): read normally
    }
    int32_t h[2];
    if (!rd(h, 8)) break;
    const int32_t cmd = h[0], n = h[1];
    in.assign(n > 0 ? n : 0, 0);
    if (n > 0 && !rd(in.data(), 4 * static_cast<size_t>(n))) break;
    out.clear();
    err.clear();
    TOBU_BIND_SLOT(cur)
    switch (cmd) {
      case 14: {
        // SELECT k: slot for the following commands -> position of the slot
        if (n < 1 || in[0] < 0 || in[0] >= static_cast<int32_t>(sl.size())) {
          reply(-1, nullptr, 0);
          break;
        }
        cur = in[0];
        const int32_t p = static_cast<int32_t>(sl[cur].session->Position());
        reply(0, &p, 1);
      } break;
      case 0: {
        int32_t d[9] = {static_cast<int32_t>(model->VocabSize()), static_cast<int32_t>(a.context),
                        model->HasMtp() ? 1 : 0, model->EosToken(), static_cast<int32_t>(session->Position())};
        std::memcpy(d + 5, kTobuStateId, 16);
        reply(0, d, 9);
      } break;
      case 1:
        if (S_.vision) session->ConfigureVision(nullptr);   // image layout belongs to the old prompt (M-RoPE positions)
        session->Reset();
        sampler = make_sampler({});
        reply(0, nullptr, 0);
        break;
      case 2: {
        const auto t0 = std::chrono::steady_clock::now();
        if (!session->Sync(in, &err)) {
          fprintf(stderr, "tobu-fnx: Sync: %s\n", err.c_str());
          reply(-2, nullptr, 0);
          break;
        }
        sampler = make_sampler(in);
        int32_t d[2] = {static_cast<int32_t>(session->Position()), us_since(t0)};
        reply(0, d, 2);
      } break;
      case 3: {
        const auto t0 = std::chrono::steady_clock::now();
        const auto before = session->Statistics();
        fni::Session::DecodeResult r;
        bool ok = false;
        try {
          ok = n >= 1 && in[0] >= 1 && step(*model, *session, sampler, static_cast<size_t>(in[0]), &r, &err);
        } catch (const std::exception& e) {
          err = e.what();
        }
        if (!ok) {
          fprintf(stderr, "tobu-fnx: DecodeStep: %s\n", err.c_str());
          reply(-2, nullptr, 0);
          break;
        }

        const auto& st = session->Statistics();
        out.push_back(r.stop ? 1 : 0);
        out.push_back(static_cast<int32_t>(r.tokens.size()));
        out.insert(out.end(), r.tokens.begin(), r.tokens.end());
        out.push_back(static_cast<int32_t>(st.cycles - before.cycles));
        out.push_back(static_cast<int32_t>(st.drafted - before.drafted));
        out.push_back(static_cast<int32_t>(st.accepted - before.accepted));
        out.push_back(us_since(t0));
        reply(0, out.data(), static_cast<int32_t>(out.size()));
      } break;
      case 4: {
        if (n < 1 || !session->Evaluate(in[0], &err)) {
          fprintf(stderr, "tobu-fnx: Evaluate: %s\n", err.c_str());
          reply(-2, nullptr, 0);
          break;
        }
        try {
          sampler.Accept(static_cast<gufo::sampling::TokenId>(in[0]));   // throws on a forbidden token (grammar)
        } catch (const std::exception& e) {
          fprintf(stderr, "tobu-fnx: EVAL %d: %s\n", in[0], e.what());
          reply(-4, nullptr, 0);
          break;
        }
        const int32_t p = static_cast<int32_t>(session->Position());
        reply(0, &p, 1);
      } break;
      case 5: {
        const auto lg = session->Logits();
        if (lg.empty()) {
          reply(-1, nullptr, 0);
          break;
        }
        reply_bytes(0, lg.data(), lg.size() * 4);
      } break;
      case 6: {
        const int32_t s = n > 0 ? in[0] : 0;
        auto snap = session->SaveSnapshot(&err, fni::Session::SnapshotMode::kMaterialized);
        if (!snap) {
          fprintf(stderr, "tobu-fnx: SaveSnapshot: %s\n", err.c_str());
          reply(-2, nullptr, 0);
          break;
        }
        const uint64_t bytes = session->SnapshotBytes();
        slots[s] = std::move(snap);
        int32_t d[2] = {static_cast<int32_t>(bytes & 0x7fffffff), static_cast<int32_t>(bytes >> 31)};
        reply(0, d, 2);
      } break;
      case 7: {
        const int32_t s = n > 0 ? in[0] : 0;
        auto it = slots.find(s);
        if (it == slots.end() || !session->RestoreSnapshot(*it->second, &err)) {
          fprintf(stderr, "tobu-fnx: RestoreSnapshot: %s\n", err.c_str());
          reply(-2, nullptr, 0);
          break;
        }
        const int32_t p = static_cast<int32_t>(session->Position());
        reply(0, &p, 1);
      } break;
      case 8:
        slots.erase(n > 0 ? in[0] : 0);
        reply(0, nullptr, 0);
        break;
      case 12: {
        // CORPUS_SET id, tokens... : set/replace a workspace document -> documents, total tokens
        if (n < 1) {
          reply(-1, nullptr, 0);
          break;
        }
        corpus.Set(in[0], std::vector<int32_t>(in.begin() + 1, in.end()));
        const int32_t d[2] = {static_cast<int32_t>(corpus.documents()), static_cast<int32_t>(corpus.tokens())};
        reply(0, d, 2);
      } break;
      case 13: {
        // CORPUS_DROP id (-1 = all)
        if (n < 1 || in[0] < 0) corpus.Clear(); else corpus.Drop(in[0]);
        const int32_t d[2] = {static_cast<int32_t>(corpus.documents()), static_cast<int32_t>(corpus.tokens())};
        reply(0, d, 2);
      } break;
      case 11: {
        // PEEK -> token: the token the next round will pick first (sampler copy over the current
        // logits, with grammar/thinking budget/penalties), without feeding it; first token right after the prefill.
        const auto lg = session->Logits();
        if (lg.empty()) {
          reply(-1, nullptr, 0);
          break;
        }
        try {
          auto copy = sampler;
          const int32_t t = static_cast<int32_t>(copy.Sample(lg));
          reply(0, &t, 1);
        } catch (const std::exception& e) {
          fprintf(stderr, "tobu-fnx: PEEK: %s\n", e.what());
          reply(-4, nullptr, 0);
        }
      } break;
      case 10: {
        // CONFIG len, UTF-8 JSON in int32 words: {"tools":[...]|null,"tool_choice":...,"parallel":bool,
        // "thinking":bool,"think_budget":N|-1,"temperature","top_p","top_k","min_p","presence_penalty",
        // "frequency_penalty","repeat_penalty","seed","ban_scripts":["cjk"],"json_schema":{…}}; applies to following SYNC/SYNCCP
        // (sampler per request). ban_scripts: hard ban in the sampler (SamplingConfig::banned), missing/empty = none.
        const int32_t len = n > 0 ? in[0] : -1;
        if (len < 0 || 4 * static_cast<int64_t>(n - 1) < len) {
          reply(-1, nullptr, 0);
          break;
        }
        try {
          const std::string text(reinterpret_cast<const char*>(in.data() + 1), static_cast<size_t>(len));
          const auto j = gufo::json::parse(text);
          gufo::sampling::SamplingConfig c;
          c.temperature = static_cast<float>(j.member_double("temperature", 0.0));
          c.top_p = static_cast<float>(j.member_double("top_p", 1.0));
          c.top_k = static_cast<std::int32_t>(j.member_double("top_k", 0.0));
          c.min_p = static_cast<float>(j.member_double("min_p", 0.0));
          c.presence_penalty = static_cast<float>(j.member_double("presence_penalty", a.presence));
          c.frequency_penalty = static_cast<float>(j.member_double("frequency_penalty", 0.0));
          c.repeat_penalty = static_cast<float>(j.member_double("repeat_penalty", 1.0));
          c.seed = static_cast<std::int64_t>(j.member_double("seed", -1.0));
          c.banned = banned_for(*model, j.find("ban_scripts"));
          c.Validate();
          thinking = j.find("thinking") != nullptr && j.find("thinking")->as_bool();
          think_budget = static_cast<int32_t>(j.member_double("think_budget", -1.0));
          constraint.reset();
          const auto* tools = j.find("tools");
          const auto* schema = j.find("json_schema");
          if ((tools != nullptr && tools->is_array() && !tools->empty()) || (schema != nullptr && schema->is_object())) {
            const auto* par = j.find("parallel");
            constraint = make_constraint(*model, tools, j.find("tool_choice"), par == nullptr || par->as_bool(),
                                         thinking, schema, &err);
            if (!err.empty()) throw std::invalid_argument(err);
          }
          cfg = c;
          reply(0, nullptr, 0);
        } catch (const std::exception& e) {
          fprintf(stderr, "tobu-fnx: CONFIG: %s\n", e.what());
          reply(-3, nullptr, 0);
        }
      } break;
      case 9: {
        // SYNCCP nb, b[nb], slot[nb], ids... : feed the prompt (like SYNC) and at the boundaries b (positions in the prompt,
        // ascending, beyond the state) capture checkpoints in the same pass (gufo SyncThrough, in-pass
        // checkpoints: the same computation as without). Response: position, us, bytes[nb] (0 = not captured).
        const int32_t nb = n > 0 ? in[0] : -1;
        if (nb < 1 || n < 1 + 2 * nb + 1) {
          reply(-1, nullptr, 0);
          break;
        }
        std::vector<uint32_t> bounds(in.begin() + 1, in.begin() + 1 + nb);
        std::vector<int32_t> slot_ids(in.begin() + 1 + nb, in.begin() + 1 + 2 * nb);
        std::span<const int32_t> prompt(in.data() + 1 + 2 * nb, in.size() - 1 - 2 * nb);
        std::vector<std::unique_ptr<fni::SessionSnapshot>> cps;
        const auto t0 = std::chrono::steady_clock::now();
        if (!session->SyncThrough(prompt, bounds, &cps, &err)) {
          fprintf(stderr, "tobu-fnx: SyncThrough: %s\n", err.c_str());
          reply(-2, nullptr, 0);
          break;
        }
        sampler = make_sampler(prompt);
        out.push_back(static_cast<int32_t>(session->Position()));
        out.push_back(us_since(t0));
        for (int32_t i = 0; i < nb; ++i) {
          int32_t mb = 0;
          if (i < static_cast<int32_t>(cps.size()) && cps[i]) {
            mb = static_cast<int32_t>((session->PrefillCheckpointBytes(bounds[i]) + (1u << 20) - 1) >> 20);
            slots[slot_ids[i]] = std::move(cps[i]);
          }
          out.push_back(mb);  // MiB, 0 = not captured
        }
        reply(0, out.data(), static_cast<int32_t>(out.size()));
      } break;
      case 15: {
        // STEPB k, (slot, max_tokens)[k]: one round for several slots at once (S2, batched decode). The same
        // paths as step(): MTP (max_tokens > 1) -> DecodeBatch on sampler copies, otherwise gufo's base path with
        // EvaluateBatch. gufo verifies batches against single calls exactly (logits, tokens; session_test.cpp).
        const int32_t k = n > 0 ? in[0] : 0;
        bool bad = k < 1 || k > 8 || n != 1 + 2 * k;
        for (int32_t i = 0; !bad && i < k; ++i) {
          const int32_t s = in[1 + 2 * i];
          bad = s < 0 || s >= static_cast<int32_t>(sl.size()) || in[2 + 2 * i] < 1;
          for (int32_t j = 0; !bad && j < i; ++j) bad = in[1 + 2 * j] == s;
        }
        if (bad) {
          reply(-1, nullptr, 0);
          break;
        }
        const auto t0 = std::chrono::steady_clock::now();
        struct Item {
          int32_t slot;
          size_t max_tokens;
          bool mtp;
          fni::Session::DecodeResult r;
          fni::Session::SpeculativeStats before;
          fni::Session::BatchOutcome outcome;
          std::optional<gufo::sampling::SamplerState> working;
        };
        std::vector<Item> it(static_cast<size_t>(k));
        std::vector<fni::Session::DecodeRequest> dec;
        std::vector<fni::Session::AdvanceRequest> adv;
        for (int32_t i = 0; i < k; ++i) {
          Item& x = it[i];
          x.slot = in[1 + 2 * i];
          x.max_tokens = static_cast<size_t>(in[2 + 2 * i]);
          Slot& S = sl[x.slot];
          x.before = S.session->Statistics();
          x.mtp = model->HasMtp() && x.max_tokens > 1;
          if (x.mtp) {
            x.working.emplace(S.sampler);   // copy as in step(): DecodeStep adds the tokens itself
            dec.push_back({S.session.get(), x.max_tokens, &*x.working, &x.r, true, &x.outcome});
            continue;
          }
          const auto logits = S.session->Logits();
          if (logits.empty()) {
            x.outcome.error = "keine Logits";
            continue;
          }
          const auto token = static_cast<std::int32_t>(S.sampler.Sample(logits));
          x.r.tokens.push_back(token);
          if (model->IsStopToken(token)) {
            x.r.stop = true;
            x.outcome.completed = true;
            continue;
          }
          S.sampler.Accept(static_cast<gufo::sampling::TokenId>(token));
          adv.push_back({S.session.get(), token, &x.outcome});
        }
        try {
          if (!dec.empty() && !fni::Session::DecodeBatch(dec, &err))
            fprintf(stderr, "tobu-fnx: DecodeBatch: %s\n", err.c_str());
          if (!adv.empty() && !fni::Session::EvaluateBatch(adv, &err))
            fprintf(stderr, "tobu-fnx: EvaluateBatch: %s\n", err.c_str());
        } catch (const std::exception& e) {
          fprintf(stderr, "tobu-fnx: STEPB: %s\n", e.what());
        }
        for (auto& x : it) {
          Slot& S = sl[x.slot];
          if (x.mtp && x.outcome.completed) S.sampler = std::move(*x.working);
          const auto& st = S.session->Statistics();
          out.push_back(x.outcome.completed ? 0 : -2);
          out.push_back(x.r.stop ? 1 : 0);
          out.push_back(static_cast<int32_t>(x.r.tokens.size()));
          out.insert(out.end(), x.r.tokens.begin(), x.r.tokens.end());
          out.push_back(static_cast<int32_t>(st.cycles - x.before.cycles));
          out.push_back(static_cast<int32_t>(st.drafted - x.before.drafted));
          out.push_back(static_cast<int32_t>(st.accepted - x.before.accepted));
          if (!x.outcome.completed) fprintf(stderr, "tobu-fnx: STEPB Slot %d: %s\n", x.slot, x.outcome.error.c_str());
        }
        out.push_back(us_since(t0));
        reply(0, out.data(), static_cast<int32_t>(out.size()));
      } break;
      case 17: {
        const bool stage_b = n > 0 && in[0] < 0;
        const int32_t ns = n > 0 ? (stage_b ? -in[0] : in[0]) : -1;
        if (ns < 0 || n < 1 + 2 * ns + 2) {
          reply(-1, nullptr, 0);
          break;
        }
        tobu::TeachSink sink;
        sink.stage_b = stage_b;
        for (int32_t i = 0; i < ns; ++i)
          sink.spans.emplace_back(static_cast<uint32_t>(in[1 + 2 * i]), static_cast<uint32_t>(in[2 + 2 * i]));
        std::span<const int32_t> prompt(in.data() + 1 + 2 * ns, in.size() - 1 - 2 * ns);
        session->Reset();
        fni::Model::SetTeach(&sink);
        const bool ok = session->Sync(prompt, &err);
        fni::Model::SetTeach(nullptr);
        session->Reset();
        sampler = make_sampler({});
        if (!ok) {
          fprintf(stderr, "tobu-fnx: TEACH: %s\n", err.c_str());
          reply(-2, nullptr, 0);
          break;
        }
        const size_t rows = sink.pos.size();
        const int32_t hc = rows ? static_cast<int32_t>(sink.r.size() / rows) : 0;
        const int32_t hid = rows ? static_cast<int32_t>(sink.y.size() / rows) : 0;
        if (stage_b && (sink.a.size() != rows * hc || sink.m.size() != rows * hid || sink.o.size() != rows * hid)) {
          fprintf(stderr, "tobu-fnx: TEACH Stufe B: Zeilenzahl passt nicht (%zu/%zu/%zu für %zu)\n", sink.a.size(),
                  sink.m.size(), sink.o.size(), rows);
          reply(-2, nullptr, 0);
          break;
        }
        std::vector<uint8_t> buf(12 + 4 * rows);
        const int32_t head[3] = {static_cast<int32_t>(rows), hc, hid};
        std::memcpy(buf.data(), head, 12);
        std::memcpy(buf.data() + 12, sink.pos.data(), 4 * rows);
        for (const auto* v : {&sink.r, &sink.y, &sink.a, &sink.m, &sink.o}) {
          const auto* p = reinterpret_cast<const uint8_t*>(v->data());
          buf.insert(buf.end(), p, p + 2 * v->size());
        }
        buf.resize((buf.size() + 3) / 4 * 4);
        reply_bytes(0, buf.data(), buf.size());
      } break;
      case 18: {
        if (n < 2) {
          reply(-1, nullptr, 0);
          break;
        }
        session->Reset();
        std::vector<int32_t> lp_bits, hit;
        bool ok = session->Sync(std::span<const int32_t>(in.data(), 1), &err);
        for (int32_t i = 1; ok && i < n; ++i) {
          const auto lg = session->Logits();
          float mx = -INFINITY;
          std::size_t am = 0;
          for (std::size_t v = 0; v < lg.size(); ++v)
            if (lg[v] > mx) { mx = lg[v]; am = v; }
          double z = 0.0;
          for (float v : lg) z += std::exp(static_cast<double>(v - mx));
          const float lp = static_cast<float>(static_cast<double>(lg[static_cast<std::size_t>(in[i])] - mx) - std::log(z));
          int32_t b;
          std::memcpy(&b, &lp, 4);
          lp_bits.push_back(b);
          hit.push_back(static_cast<int32_t>(am) == in[i] ? 1 : 0);
          if (i + 1 < n) ok = session->Evaluate(in[i], &err);
        }
        session->Reset();
        sampler = make_sampler({});
        if (!ok) {
          fprintf(stderr, "tobu-fnx: SCORE: %s\n", err.c_str());
          reply(-2, nullptr, 0);
          break;
        }
        out = lp_bits;
        out.insert(out.end(), hit.begin(), hit.end());
        reply(0, out.data(), static_cast<int32_t>(out.size()));
      } break;
      case 16: {
        std::vector<uint32_t> u;
        uint32_t layers = 0;
        if (!model->ExpertUsage(&u, &layers, n >= 1 && in[0] != 0, &err)) {
          fprintf(stderr, "tobu-fnx: USAGE: %s\n", err.c_str());
          reply(-2, nullptr, 0);
          break;
        }
        out.push_back(u.empty() ? 0 : static_cast<int32_t>(layers));
        out.push_back(u.empty() ? 0 : static_cast<int32_t>(u.size() / 3 / layers));
        for (auto v : u) out.push_back(static_cast<int32_t>(v));
        reply(0, out.data(), static_cast<int32_t>(out.size()));
      } break;
      case 19: {
        // IMAGE id, detail, w, h, max_pixels, rgb... -> gh, gw, tokens, sha256[8]
        if (n < 5 || in[2] <= 0 || in[3] <= 0 ||
            static_cast<uint64_t>(n - 5) * 4 < static_cast<uint64_t>(in[2]) * static_cast<uint64_t>(in[3]) * 3) {
          reply(-1, nullptr, 0);
          break;
        }
        if (!encoder) {
          fprintf(stderr, "tobu-fnx: IMAGE ohne Vision-Turm (--mmproj fehlt)\n");
          reply(-3, nullptr, 0);
          break;
        }
        try {
          gufo::core::Image raw;
          raw.width = static_cast<uint32_t>(in[2]);
          raw.height = static_cast<uint32_t>(in[3]);
          const auto* px = reinterpret_cast<const uint8_t*>(in.data() + 5);
          raw.pixels.assign(px, px + static_cast<size_t>(raw.width) * raw.height * 3);
          auto img = std::make_shared<const gufo::core::Image>(qv::ResizeImage(
              raw, in[1] == 1 ? gufo::tokenization_iq::ChatMessage::ImageDetail::kLow
                              : gufo::tokenization_iq::ChatMessage::ImageDetail::kAuto,
              in[4] > 0 ? static_cast<uint32_t>(in[4]) : 16777216u));
          gufo::crypto::Sha256Hasher hs;
          hs.Update(img->pixels);
          const auto dg = hs.Finish();
          if (auto it = images.find(in[0]); it != images.end()) {
            image_bytes -= it->second.img->pixels.size();
            images.erase(it);
          }
          image_bytes += img->pixels.size();
          images[in[0]] = {img, ++image_tick};
          while (image_bytes > image_limit && images.size() > 1) {   // LRU, the new image stays
            auto old = images.end();
            for (auto it = images.begin(); it != images.end(); ++it)
              if (it->first != in[0] && (old == images.end() || it->second.used < old->second.used)) old = it;
            if (old == images.end()) break;
            image_bytes -= old->second.img->pixels.size();
            images.erase(old);
          }
          const int32_t gh = static_cast<int32_t>(img->height / 32), gw = static_cast<int32_t>(img->width / 32);
          out = {gh, gw, gh * gw};
          for (int i = 0; i < 8; ++i) {
            int32_t w;
            memcpy(&w, dg.data() + 4 * i, 4);
            out.push_back(w);
          }
          reply(0, out.data(), static_cast<int32_t>(out.size()));
        } catch (const std::exception& e) {
          fprintf(stderr, "tobu-fnx: IMAGE: %s\n", e.what());
          reply(-2, nullptr, 0);
        }
      } break;
      case 20: {
        // VPROMPT k, (id, offset)[k] -> prefix_length, delta
        if (n < 1 || in[0] < 0 || n < 1 + 2 * in[0]) {
          reply(-1, nullptr, 0);
          break;
        }
        try {
          if (in[0] == 0) {
            session->ConfigureVision(nullptr);
            int32_t d[2] = {0, 0};
            reply(0, d, 2);
            break;
          }
          if (!encoder) {
            reply(-3, nullptr, 0);
            break;
          }
          std::vector<qv::PlacedImage> placed;
          bool missing = false;
          for (int32_t i = 0; i < in[0]; ++i) {
            auto it = images.find(in[1 + 2 * i]);
            if (it == images.end() || in[2 + 2 * i] < 0) {
              missing = true;
              break;
            }
            it->second.used = ++image_tick;
            placed.push_back({it->second.img, static_cast<uint32_t>(in[2 + 2 * i])});
          }
          if (missing) {
            fprintf(stderr, "tobu-fnx: VPROMPT: Bild fehlt im Bildspeicher\n");
            reply(-4, nullptr, 0);
            break;
          }
          auto prompt = std::make_shared<const qv::Prompt>(qv::BuildPrompt(placed, encoder->identity(), a.context));
          const int32_t d[2] = {static_cast<int32_t>(prompt->rope.PrefixLength()), prompt->rope.Delta()};
          session->ConfigureVision(std::move(prompt));
          reply(0, d, 2);
        } catch (const std::exception& e) {
          fprintf(stderr, "tobu-fnx: VPROMPT: %s\n", e.what());
          reply(-2, nullptr, 0);
        }
      } break;
      case 21: {
        // IMAGE_DROP id (-1 all) -> remaining images
        if (n >= 1 && in[0] == -1) {
          images.clear();
          image_bytes = 0;
        } else if (n >= 1) {
          if (auto it = images.find(in[0]); it != images.end()) {
            image_bytes -= it->second.img->pixels.size();
            images.erase(it);
          }
        }
        const int32_t left = static_cast<int32_t>(images.size());
        reply(0, &left, 1);
      } break;
      case 22: {
        // PERSIST slot, len, path -> MiB
        const std::string path = n >= 2 ? path_of(in.data() + 1, n - 1) : std::string();
        auto it = n >= 2 ? slots.find(in[0]) : slots.end();
        if (path.empty() || it == slots.end()) {
          reply(-1, nullptr, 0);
          break;
        }
        std::shared_ptr<fni::SessionSnapshot> snap = it->second;
        std::span<const uint8_t> bytes;
        try {
          bytes = snap->bytes();                    // materialize here (GPU transfer on the command thread)
        } catch (const std::exception& e) {
          fprintf(stderr, "tobu-fnx: PERSIST: %s\n", e.what());
          reply(-2, nullptr, 0);
          break;
        }
        {
          std::lock_guard<std::mutex> g(wmu);
          ++writers;
        }
        std::thread([snap, bytes, path, &wmu, &wcv, &writers] {
          const std::string tmp = path + ".tmp";
          bool ok = false;
          if (FILE* f = fopen(tmp.c_str(), "wb")) {
            (void)fchmod(fileno(f), 0600);           // states from prompts: only the owner
            ok = fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
            ok = (fflush(f) == 0) && ok;
            ok = (fsync(fileno(f)) == 0) && ok;
            ok = (fclose(f) == 0) && ok;
          }
          if (ok) ok = rename(tmp.c_str(), path.c_str()) == 0;
          if (!ok) {
            fprintf(stderr, "tobu-fnx: PERSIST %s: Schreiben fehlgeschlagen\n", path.c_str());
            remove(tmp.c_str());
          }
          std::lock_guard<std::mutex> g(wmu);
          --writers;
          wcv.notify_all();
        }).detach();
        const int32_t mb = static_cast<int32_t>((bytes.size() + (1u << 20) - 1) >> 20);
        reply(0, &mb, 1);
      } break;
      case 23: {
        // LOADFILE len, path -> position
        const std::string path = path_of(in.data(), n);
        std::vector<uint8_t> buf;
        if (FILE* f = path.empty() ? nullptr : fopen(path.c_str(), "rb")) {
          fseek(f, 0, SEEK_END);
          const long sz = ftell(f);
          fseek(f, 0, SEEK_SET);
          if (sz > 0) {
            buf.resize(static_cast<size_t>(sz));
            if (fread(buf.data(), 1, buf.size(), f) != buf.size()) buf.clear();
          }
          fclose(f);
        }
        if (buf.empty() || !session->RestoreSnapshot(std::span<const uint8_t>(buf), &err)) {
          fprintf(stderr, "tobu-fnx: LOADFILE %s: %s\n", path.c_str(), buf.empty() ? "nicht lesbar" : err.c_str());
          reply(-2, nullptr, 0);
          break;
        }
        const int32_t p = static_cast<int32_t>(session->Position());
        reply(0, &p, 1);
      } break;
      case 25: {                          // STATS: draft counters of the selected slot (u64 -> lo, hi)
        const auto& st = session->Statistics();
        const uint64_t v[13] = {st.cycles, st.drafted, st.accepted, st.lookup, st.lookup_accepted, st.lookup_cycles,
                                st.corpus, st.corpus_accepted, st.corpus_cycles, st.store, st.store_accepted,
                                st.store_cycles, 0};
        int32_t d[26];
        for (int i = 0; i < 13; ++i) {
          d[2 * i] = static_cast<int32_t>(static_cast<uint32_t>(v[i] & 0xffffffffu));
          d[2 * i + 1] = static_cast<int32_t>(static_cast<uint32_t>(v[i] >> 32));
        }
        reply(0, d, 26);
      } break;
      case 24: {
        std::unique_lock<std::mutex> g(wmu);
        wcv.wait(g, [&] { return writers == 0; });
        const int32_t z = 0;
        reply(0, &z, 1);
      } break;
      default:
        reply(-9, nullptr, 0);
    }
    if (cmd == 1 || cmd == 7 || cmd == 23) sl[cur].vision = false;           // RESET/LOAD: layout without images
    if (cmd == 20) sl[cur].vision = n >= 1 && in[0] > 0;
    if (encoder && sl[cur].vision && (cmd == 2 || cmd == 9 || cmd == 20))
      vision_used = std::chrono::steady_clock::now();   // only prefill with images keeps the tower warm
  }
  {  // End: wait for open writers (otherwise .tmp files remain)
    std::unique_lock<std::mutex> g(wmu);
    wcv.wait(g, [&] { return writers == 0; });
  }
  return 0;
}
