// Test of the external corpus (tobu/corpus.hpp) without GPU, without model; runs with the system compiler:
//
//   clang++ -std=c++20 -O2 -Wall -Wextra -I engine \
//     engine/tobu/test_corpus.cpp -o /tmp/test_corpus && /tmp/test_corpus
//
// Checks lazy removal (TOB-56): after Drop/Set on an existing id, Find never returns occurrences of removed or
// replaced documents, hits match a freshly built corpus, rebuild only when dead > live.
// TOB-57: flat index (no rehash spike during build), compaction in slices over several Set/Drop, full
// candidate lists evict dead occurrences first (Find like a corpus freshly built in the same Set order,
// also with a small vocabulary). Finally a micro-bench (600 documents × 2000 tokens, single Drops + Set
// on an existing id, Find latency, longest single call).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "tobu/corpus.hpp"

namespace {

int fails = 0;

void check(bool ok, const char* name) {
  std::printf("%s %s\n", ok ? "ok    " : "FEHLER", name);
  if (!ok) ++fails;
}

using Tokens = std::vector<std::int32_t>;

Tokens Random(std::mt19937& rng, std::size_t n, std::int32_t vocab) {
  std::uniform_int_distribution<std::int32_t> d(0, vocab - 1);
  Tokens t(n);
  for (auto& x : t) x = d(rng);
  return t;
}

tobu::Corpus::Match FindIn(const tobu::Corpus& c, const Tokens& ctx, std::size_t minimum = 4) {
  return c.Find(std::span<const std::int32_t>(ctx), {}, minimum);
}

double Seconds(std::chrono::steady_clock::time_point a) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
}

}  // namespace

int main() {
  std::mt19937 rng(56);

  {  // Drop: hits disappear, others stay, no rebuild as long as dead <= live
    tobu::Corpus c;
    const Tokens a = Random(rng, 200, 1000), b = Random(rng, 200, 1000), z = Random(rng, 200, 1000);
    c.Set(1, a);
    c.Set(2, b);
    c.Set(3, z);
    const Tokens ctx_a(a.begin() + 50, a.begin() + 70), ctx_b(b.begin() + 50, b.begin() + 70);
    check(FindIn(c, ctx_a).id == 1 && FindIn(c, ctx_a).pos == 70, "Treffer in Dokument 1 vor Drop");
    c.Drop(1);
    check(FindIn(c, ctx_a).length == 0, "nach Drop kein Treffer aus Dokument 1");
    check(FindIn(c, ctx_b).id == 2 && FindIn(c, ctx_b).length >= 4, "Dokument 2 bleibt auffindbar");
    check(c.dead_tokens() == 200 && c.tokens() == 400 && c.documents() == 2, "Drop zählt tot, kein Neuaufbau");
    c.Drop(2);  // dead 400 > live 200 -> rebuild
    check(c.dead_tokens() == 0 && c.tokens() == 200, "Neuaufbau erst bei tot > lebend");
    check(FindIn(c, ctx_b).length == 0, "nach Neuaufbau kein Treffer aus Dokument 2");
    c.Drop(3);
    check(c.empty() && c.dead_tokens() == 0, "letztes Dokument entfernt: Index leer");
  }

  {  // Set on an existing id: old occurrences (old generation) do not count, even if positions would still fit
    tobu::Corpus c;
    const Tokens old = Random(rng, 300, 1000), filler = Random(rng, 1000, 1000);
    c.Set(7, old);
    c.Set(8, filler);
    Tokens neu = old;  // same length, only one section changed
    for (std::size_t i = 100; i < 140; ++i) neu[i] = (neu[i] + 1) % 1000;
    c.Set(7, neu);
    const Tokens ctx_old(old.begin() + 110, old.begin() + 130), ctx_new(neu.begin() + 110, neu.begin() + 130);
    check(FindIn(c, ctx_old).length == 0, "Set: alte Fassung nicht mehr auffindbar");
    const auto m = FindIn(c, ctx_new);
    check(m.id == 7 && m.pos == 130 && m.length == 20, "Set: neue Fassung auffindbar");
    check(m.doc != nullptr && (*m.doc)[m.pos] == neu[130], "Set: Match zeigt in die neue Fassung");
    check(c.documents() == 2 && c.tokens() == 1300 && c.dead_tokens() == 300, "Set: Zähler (lebend/tot)");
  }

  // Random sequence of Set/Drop: Find like a freshly built corpus in the same Set order. Large vocabulary:
  // candidate limit not reached; small vocabulary: lists fill up, dead occurrences must not evict live
  // ones (TOB-57).
  for (const std::int32_t vocab : {60000, 12}) {
    tobu::Corpus c;
    std::vector<Tokens> live(40);
    std::vector<bool> has(40, false);
    std::vector<int> order;  // ids in the order of their last Set
    std::uniform_int_distribution<int> pick(0, 39), op(0, 2);
    bool same = true, never_dead = true;
    for (int step = 0; step < 3000; ++step) {
      const int id = pick(rng);
      std::erase(order, id);
      if (op(rng) == 0) {
        c.Drop(id);
        has[id] = false;
      } else {
        live[id] = Random(rng, 30 + id, vocab);
        has[id] = true;
        order.push_back(id);
        c.Set(id, live[id]);
      }
      tobu::Corpus fresh;
      for (const int i : order) fresh.Set(i, live[i]);
      for (int q = 0; q < 4; ++q) {
        const int j = pick(rng);
        if (live[j].size() < 20) continue;
        const Tokens ctx(live[j].begin() + 5, live[j].begin() + 20);
        const auto a = FindIn(c, ctx), b = FindIn(fresh, ctx);
        if (a.id != b.id || a.pos != b.pos || a.length != b.length) same = false;
        if (a.length > 0 && (!has[a.id] || (*a.doc) != live[a.id])) never_dead = false;
      }
      if (c.tokens() != fresh.tokens() || c.documents() != fresh.documents()) same = false;
      if (c.dead_tokens() > c.tokens() && !c.empty()) same = false;
    }
    check(same, vocab > 1000 ? "Zufallsfolge: Find/Zähler wie frisch aufgebaut, tot <= lebend"
                             : "Zufallsfolge, volle Listen: Find wie frisch aufgebaut (tote verdrängen keine lebenden)");
    check(never_dead, "Zufallsfolge: nie ein Treffer aus entferntem/ersetztem Dokument");
  }

  {  // Compaction in slices: large index, compaction over several calls, Find correct meanwhile
    tobu::Corpus c;
    std::vector<Tokens> docs;
    for (int i = 0; i < 64; ++i) docs.push_back(Random(rng, 4000, 1 << 20));
    for (int i = 0; i < 64; ++i) c.Set(i, docs[static_cast<std::size_t>(i)]);
    const std::size_t keys0 = c.keys();
    int i = 0;
    while (!c.compacting() && i < 64) c.Drop(i++);
    const int started = i;
    bool ok = c.compacting(), finds = true;
    int ops = 0;
    for (; c.compacting() && i < 63; ++i, ++ops) {
      const Tokens ctx(docs[63].begin() + 10, docs[63].begin() + 30), gone(docs[0].begin() + 10, docs[0].begin() + 30);
      if (FindIn(c, ctx).id != 63 || FindIn(c, gone).length != 0) finds = false;
      c.Drop(i);
    }
    check(ok && ops >= 2, "Verdichtung läuft über mehrere Drops (kein einzelner Aufruf über den ganzen Index)");
    check(finds, "während der Verdichtung: lebende auffindbar, entfernte nicht");
    check(!c.compacting() && c.keys() < keys0 * (64 - started + ops) / 64 + 1000,
          "nach der Verdichtung: Schlüssel entfernter Dokumente freigegeben");
  }

  {  // Micro-bench: 600 × 2000 tokens, 600 single Drops or Set on an existing id; Find latency
    constexpr int kDocs = 600;
    constexpr std::size_t kLen = 2000;
    std::vector<Tokens> docs;
    for (int i = 0; i < kDocs; ++i) docs.push_back(Random(rng, kLen, 150000));
    tobu::Corpus c;
    double build_max = 0;
    auto t = std::chrono::steady_clock::now();
    for (int i = 0; i < kDocs; ++i) {
      const auto t1 = std::chrono::steady_clock::now();
      c.Set(i, docs[i]);
      build_max = std::max(build_max, Seconds(t1));
    }
    const double build = Seconds(t);
    std::vector<Tokens> queries;
    for (int i = 0; i < 2000; ++i) {
      const auto& d = docs[static_cast<std::size_t>(i) % kDocs];
      queries.emplace_back(d.begin() + 100 + i % 500, d.begin() + 140 + i % 500);
    }
    t = std::chrono::steady_clock::now();
    std::size_t hits = 0;
    for (const auto& q : queries) hits += FindIn(c, q).length > 0;
    const double find0 = Seconds(t) / queries.size();

    t = std::chrono::steady_clock::now();
    double op_max = 0;
    for (int r = 1; r <= 3; ++r)  // every file changed three times (compaction kicks in meanwhile)
      for (int i = 0; i < kDocs; ++i) {
        const auto t1 = std::chrono::steady_clock::now();
        c.Set(i, docs[(i + r) % kDocs]);
        op_max = std::max(op_max, Seconds(t1));
      }
    const double sets = Seconds(t);
    t = std::chrono::steady_clock::now();
    for (const auto& q : queries) hits += FindIn(c, q).length > 0;
    const double find1 = Seconds(t) / queries.size();

    t = std::chrono::steady_clock::now();
    for (int i = 0; i < kDocs; ++i) {
      const auto t1 = std::chrono::steady_clock::now();
      c.Drop(i);
      op_max = std::max(op_max, Seconds(t1));
    }
    const double drops = Seconds(t);
    std::printf("bench  600×2000: Aufbau %.3f s (längstes Set %.1f ms), %d Set auf bestehende Id %.3f s, "
                "%d Einzel-Drops %.3f s, längster Set/Drop %.1f ms, Find %.2f µs -> %.2f µs (%zu/%zu Treffer)\n",
                build, build_max * 1e3, 3 * kDocs, sets, kDocs, drops, op_max * 1e3, find0 * 1e6, find1 * 1e6, hits,
                2 * queries.size());
    check(hits == 2 * queries.size(), "Bench: alle Anfragen treffen, auch nach wiederholtem Ersetzen");
    check(build_max < 0.05, "Bench: kein Set im Aufbau >= 50 ms (alt: Rehash-Spitze ~250 ms)");
    check(op_max < 0.05, "Bench: kein Set/Drop >= 50 ms (alt: Verdichtung in einem Aufruf ~290 ms)");
    check(drops < 3.0, "Bench: 600 Einzel-Drops < 3 s (alt: je Drop voller Neuaufbau, ~100 s)");
    check(c.empty(), "Bench: alles entfernt");
  }

  std::printf(fails == 0 ? "alle Prüfungen bestanden\n" : "%d Prüfung(en) fehlgeschlagen\n", fails);
  return fails == 0 ? 0 : 1;
}
