// tobu-fnx: external corpus for copy drafts (E4). Copyright (c) 2026 hakobu, Apache-2.0 (tobu).
//
// gufo's prompt lookup copies only from the session's own history. tobu additionally knows the files of the
// workspace (--workspace, server/agent/drafts.py WorkspaceCache). When the history has no hit, this index
// supplies a continuation from these files: the same rule as gufo (at least `minimum`
// matching context tokens, key from the last three tokens); the target model verifies it like any draft,
// so the output never depends on the corpus, only the speed.
#ifndef TOBU_FNX_CORPUS_HPP_
#define TOBU_FNX_CORPUS_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <memory>
#include <new>
#include <span>
#include <unordered_map>
#include <vector>

namespace tobu {

class Corpus {
public:
  static constexpr std::size_t kKey = 3;           // as speculative::PromptLookup::kKeyTokens
  static constexpr std::size_t kMaxCandidates = 48;  // per key the most recent live occurrences
  static constexpr std::size_t kMaxMatch = 64;        // as speculative::PromptLookup::kMaxMatch
  static constexpr std::int32_t kStoreBase = 1 << 24;  // ids from here on: answer store (server: XSTORE_BASE)
  /// Compaction in slices (TOB-57): per Set/Drop at most this many candidate lists (at least), plus per
  /// set/removed token kSweepPerToken more; compaction is thus done before dead > live holds
  /// again, and no single call carries the whole index.
  static constexpr std::size_t kSweepSlice = 8192;
  static constexpr std::size_t kSweepPerToken = 4;

  struct Match {
    const std::vector<std::int32_t>* doc{nullptr};
    std::size_t pos{0};     // index of the first token to copy in the document
    std::size_t length{0};  // matching context tokens before pos
    std::int32_t id{-1};    // document id (>= kStoreBase: store of earlier answers, otherwise file)
  };

  /// Set or replace a document. Replace = Drop + re-index, without a rebuild: the old occurrences stay
  /// in the index but belong to an old generation and are skipped by Find.
  void Set(std::int32_t id, std::vector<std::int32_t> tokens) {
    const std::size_t n = tokens.size();
    const std::size_t old = Drop(id, false);
    auto& d = docs_[id];
    d.tokens = std::move(tokens);
    d.gen = ++gen_;
    total_ += n;
    Reserve(used_ + n);  // at most one new key per token: no growth in the middle of indexing
    Migrate(kMigrateSlice + kMigratePerToken * n);
    Index(id, d);
    Sweep(n + old);
  }

  /// Remove a document (lazy): its occurrences count as dead and are skipped by Find. Compaction starts
  /// as soon as the index holds more dead than live tokens, in slices over the following Set/Drop
  /// (amortized O(1) per token, no single call O(index)).
  void Drop(std::int32_t id) { (void)Drop(id, true); }

  void Clear() {
    docs_.clear();
    ClearIndex();
    total_ = 0;
  }

  /// Reserve room for `keys` keys. If the table grows, it migrates incrementally (one slice per Set/Drop,
  /// Find searches both tables meanwhile); no call rehashes the whole index at once.
  void Reserve(std::size_t keys) {
    if (cur_.cap != 0 && (keys + tombs_) * 2 <= cur_.cap) return;  // load <= 1/2 incl. tombstones
    Migrate(SIZE_MAX);  // a migration still pending (only with very large documents): finish it first
    if ((keys + tombs_) * 2 <= cur_.cap) return;
    std::size_t cap = cur_.cap == 0 ? kMinSlots : cur_.cap;
    while (keys * 2 > cap) cap *= 2;
    old_ = std::move(cur_);  // grow or (same size) clear tombstones
    cur_ = Table(cap);
    tombs_ = 0;
    moved_ = 0;
    if (old_.cap <= kMigrateSlice) Migrate(SIZE_MAX);
  }

  [[nodiscard]] bool empty() const noexcept { return docs_.empty(); }
  [[nodiscard]] std::size_t tokens() const noexcept { return total_; }
  [[nodiscard]] std::size_t documents() const noexcept { return docs_.size(); }
  /// Tokens of removed/replaced documents whose occurrences may still be in the index (until compaction is done).
  [[nodiscard]] std::size_t dead_tokens() const noexcept { return dead_; }
  /// Is a compaction (in slices) currently running?
  [[nodiscard]] bool compacting() const noexcept { return sweeping_; }
  /// Keys in the index (candidate lists with at least one occurrence or not yet compacted).
  [[nodiscard]] std::size_t keys() const noexcept { return used_; }

  /// Longest match for the context `history` + `tail` (end = most recent token), at least `minimum` tokens.
  [[nodiscard]] Match Find(std::span<const std::int32_t> history, std::span<const std::int32_t> tail,
                           std::size_t minimum) const {
    Match best;
    const std::size_t n = history.size() + tail.size();
    if (n < std::max(minimum, kKey) || used_ == 0) return best;
    const auto at = [&](std::size_t i) {  // context token i (0 = oldest)
      return i < history.size() ? history[i] : tail[i - history.size()];
    };
    const std::uint32_t l = Lookup(Key(at(n - 3), at(n - 2), at(n - 1)));
    if (l == kNone) return best;
    for (const auto& [id, gen, pos] : lists_[l].v) {
      const auto d = docs_.find(id);
      // dead occurrence: document removed or replaced since (different generation)
      if (d == docs_.end() || d->second.gen != gen || pos >= d->second.tokens.size()) continue;
      const auto& doc = d->second.tokens;
      std::size_t len = 0;
      // at most kMaxMatch backwards (like gufo's PromptLookup): otherwise long equal runs cost O(length) per step
      while (len < kMaxMatch && len < n && len < pos && doc[pos - 1 - len] == at(n - 1 - len)) ++len;
      if (len >= minimum && len > best.length) best = {&doc, pos, len, id};
    }
    return best;
  }

private:
  struct Doc {
    std::vector<std::int32_t> tokens;
    std::uint32_t gen{0};  // generation at Set; occurrences of older generations are dead
  };
  struct Pos {
    std::int32_t id;    // document
    std::uint32_t gen;  // generation of the document at indexing
    std::uint32_t pos;  // index of the token after the key
  };
  struct List {
    std::uint64_t key{0};
    std::uint32_t seen{0};  // drops_ at the last removal of dead occurrences: no new dead ones since, if equal
    bool live{false};       // in the table (otherwise free, index in free_)
    std::vector<Pos> v;     // oldest first
  };
  struct Slot {
    std::uint64_t key;
    std::uint32_t tag;  // 0 empty, 1 tombstone, otherwise index in lists_ + 2 (empty = 0: table via calloc, without zeroing)
  };
  struct Table {
    struct Free {
      void operator()(Slot* p) const noexcept { std::free(p); }
    };
    std::unique_ptr<Slot[], Free> s;
    std::size_t cap{0};  // power of two or 0
    Table() = default;
    explicit Table(std::size_t n) : s(static_cast<Slot*>(std::calloc(n, sizeof(Slot)))), cap(n) {
      if (!s) throw std::bad_alloc();
    }
  };
  static constexpr std::uint32_t kEmpty = 0;
  static constexpr std::uint32_t kTomb = 1;
  static constexpr std::uint32_t kNone = 0xffffffffU;
  static constexpr std::size_t kMinSlots = 1024;
  static constexpr std::size_t kMigrateSlice = 16384;  // slots of the old table per Set/Drop, plus 4 per token
  static constexpr std::size_t kMigratePerToken = 4;

  static std::uint64_t Key(std::int32_t a, std::int32_t b, std::int32_t c) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) * 0x9E3779B97F4A7C15ULL) ^
           (static_cast<std::uint64_t>(static_cast<std::uint32_t>(b)) << 21) ^ static_cast<std::uint32_t>(c);
  }
  static std::size_t Home(const Table& t, std::uint64_t key) noexcept {
    return static_cast<std::size_t>((key * 0xD6E8FEB86659FD93ULL) >> 32) & (t.cap - 1);
  }

  // slot of the key in t or SIZE_MAX
  static std::size_t Probe(const Table& t, std::uint64_t key) noexcept {
    if (t.cap == 0) return SIZE_MAX;
    const std::size_t mask = t.cap - 1;
    for (std::size_t i = Home(t, key);; i = (i + 1) & mask) {
      const Slot& s = t.s[i];
      if (s.tag == kEmpty) return SIZE_MAX;
      if (s.tag != kTomb && s.key == key) return i;
    }
  }

  std::uint32_t Lookup(std::uint64_t key) const noexcept {
    std::size_t i = Probe(cur_, key);
    if (i != SIZE_MAX) return cur_.s[i].tag - 2;
    i = Probe(old_, key);  // migration running: keys not yet migrated
    return i != SIZE_MAX ? old_.s[i].tag - 2 : kNone;
  }

  // list for the key, newly created if needed (in the current table). Space is reserved via Reserve.
  List& Upsert(std::uint64_t key) {
    const std::size_t mask = cur_.cap - 1;
    std::size_t tomb = SIZE_MAX, i = Home(cur_, key);
    for (;; i = (i + 1) & mask) {
      const Slot& s = cur_.s[i];
      if (s.tag == kEmpty) break;
      if (s.tag == kTomb) {
        if (tomb == SIZE_MAX) tomb = i;
      } else if (s.key == key) {
        return lists_[s.tag - 2];
      }
    }
    if (const std::size_t o = Probe(old_, key); o != SIZE_MAX) return lists_[old_.s[o].tag - 2];
    if (tomb != SIZE_MAX) {
      i = tomb;
      --tombs_;
    }
    std::uint32_t l;
    if (!free_.empty()) {
      l = free_.back();
      free_.pop_back();
    } else {
      l = static_cast<std::uint32_t>(lists_.size());
      lists_.emplace_back();  // deque: growing moves no existing lists
    }
    cur_.s[i] = {key, l + 2};
    ++used_;
    List& L = lists_[l];
    L.key = key;
    L.seen = drops_;
    L.live = true;
    return L;
  }

  void Erase(std::uint32_t l) {
    List& L = lists_[l];
    if (const std::size_t i = Probe(cur_, L.key); i != SIZE_MAX) {
      cur_.s[i].tag = kTomb;
      ++tombs_;
    } else if (const std::size_t o = Probe(old_, L.key); o != SIZE_MAX) {
      old_.s[o].tag = kTomb;  // skipped during migration
    }
    --used_;
    L.live = false;
    std::vector<Pos>().swap(L.v);
    free_.push_back(l);
  }

  // Migrate up to `budget` slots of the old table into the current one; when done: free the old table.
  void Migrate(std::size_t budget) {
    if (old_.cap == 0) return;
    const std::size_t mask = cur_.cap - 1;
    for (; moved_ < old_.cap && budget > 0; ++moved_, --budget) {
      const Slot& s = old_.s[moved_];
      if (s.tag <= kTomb) continue;
      std::size_t i = Home(cur_, s.key);
      while (cur_.s[i].tag != kEmpty) i = (i + 1) & mask;
      cur_.s[i] = s;
    }
    if (moved_ == old_.cap) old_ = Table();
  }

  bool Dead(const Pos& p) const {
    const auto d = docs_.find(p.id);
    return d == docs_.end() || d->second.gen != p.gen;
  }

  void Index(std::int32_t id, const Doc& d) {
    const auto& t = d.tokens;
    for (std::size_t i = kKey; i < t.size(); ++i) {
      List& L = Upsert(Key(t[i - 3], t[i - 2], t[i - 1]));
      auto& v = L.v;
      if (v.size() >= kMaxCandidates && L.seen != drops_) {
        // full list: evict dead occurrences first, not live ones (TOB-57: otherwise the hit rate drops after
        // repeated replacing). Once per drop epoch; after that no dead ones are added until the next Drop.
        std::erase_if(v, [&](const Pos& p) { return Dead(p); });
        L.seen = drops_;
      }
      if (v.size() >= kMaxCandidates) v.erase(v.begin());
      v.push_back({id, d.gen, static_cast<std::uint32_t>(i)});
    }
  }

  // -> tokens of the removed document (0: did not exist)
  std::size_t Drop(std::int32_t id, bool sweep) {
    auto it = docs_.find(id);
    if (it == docs_.end()) return 0;
    const std::size_t n = it->second.tokens.size();
    total_ -= n;
    dead_ += n;
    ++drops_;
    docs_.erase(it);
    if (docs_.empty()) {
      ClearIndex();
      return n;
    }
    if (!sweeping_ && dead_ > total_) {
      sweeping_ = true;
      cursor_ = 0;
      sweep_dead_ = dead_;
    }
    if (sweep) {
      Migrate(kMigrateSlice + kMigratePerToken * n);
      Sweep(n);
    }
    return n;
  }

  // One slice of compaction: remove dead occurrences in place, free empty lists (tombstone in the
  // table). Live occurrences keep their order (most recent last). Whatever dies during compaction
  // stays counted for the next one.
  void Sweep(std::size_t tokens) {
    if (!sweeping_) return;
    std::size_t budget = kSweepSlice + kSweepPerToken * tokens;
    const std::size_t end = lists_.size();
    for (; cursor_ < end && budget > 0; ++cursor_, --budget) {
      List& L = lists_[cursor_];
      if (!L.live) continue;
      std::erase_if(L.v, [&](const Pos& p) { return Dead(p); });
      if (L.v.empty()) Erase(static_cast<std::uint32_t>(cursor_));
    }
    if (cursor_ < end) return;
    sweeping_ = false;
    dead_ -= sweep_dead_;
  }

  void ClearIndex() {
    cur_ = Table();
    old_ = Table();
    moved_ = 0;
    lists_.clear();
    free_.clear();
    used_ = tombs_ = 0;
    dead_ = 0;
    sweeping_ = false;
    cursor_ = sweep_dead_ = 0;
  }

  std::unordered_map<std::int32_t, Doc> docs_;
  // Index (TOB-57): flat table with linear probing (key -> list), lists in a deque. Growing migrates
  // only the 16-byte slots incrementally, lists stay in place; reservation is done per Set for all tokens.
  Table cur_;
  Table old_;                        // while growing: slots not yet migrated from moved_ on
  std::size_t moved_{0};
  std::deque<List> lists_;
  std::vector<std::uint32_t> free_;  // freed lists (compaction)
  std::size_t used_{0};              // keys (both tables)
  std::size_t tombs_{0};             // tombstones of the current table
  std::size_t total_{0};
  std::size_t dead_{0};      // tokens of dead documents that may still be in the index
  std::uint32_t gen_{0};     // running generation per Set (overflow only after 2^32 Sets, practically impossible)
  std::uint32_t drops_{0};   // drop epoch (per removed/replaced document)
  bool sweeping_{false};
  std::size_t cursor_{0};      // next list of the running compaction
  std::size_t sweep_dead_{0};  // dead_ at the start of compaction
};

}  // namespace tobu

#endif  // TOBU_FNX_CORPUS_HPP_
