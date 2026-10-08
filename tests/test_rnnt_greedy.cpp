// tests/test_rnnt_greedy.cpp — Catch2 unit tests for the greedy RNNT loop seam
// (core/rnnt_greedy.h) with synthetic joint / predictor callbacks, no model.

#include <catch2/catch_test_macros.hpp>
#include "core/rnnt_greedy.h"

#include <cmath>
#include <cstdint>
#include <vector>

using core_context_bias::Trie;
using core_rnnt::emission;
using core_rnnt::rnnt_greedy_loop;

namespace {

const Trie kNoHints;

constexpr int kVocab = 6;
constexpr int kBlank = kVocab - 1;

// Synthetic predictor: its state is a hash of the emitted tokens; the joint is
// a deterministic pseudo-random function of (frame, state).
struct synthetic_model {
    uint32_t seed;
    uint32_t state = 0;

    void joint(int t, std::vector<float>& logits) const {
        logits.resize(kVocab);
        uint32_t x = seed ^ (uint32_t)t * 2654435761u ^ state * 40503u;
        for (int v = 0; v < kVocab; v++) {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            logits[(size_t)v] = (float)(x % 1000) / 100.0f;
        }
        logits[kBlank] += 1.5f; // blank wins often enough to end frames
    }
    void emit(int tok) { state = state * 31u + (uint32_t)tok + 1u; }
};

// The greedy loop as it was inlined in gigaam_rnnt_decode before the seam.
std::vector<emission> reference_loop(int T, int max_symbols, synthetic_model m) {
    std::vector<emission> emitted;
    std::vector<float> logits;
    for (int t = 0; t < T; t++) {
        for (int sym = 0; sym < max_symbols; sym++) {
            m.joint(t, logits);
            int tok = 0;
            float best = logits[0];
            for (int v = 1; v < kVocab; v++) {
                if (logits[(size_t)v] > best) {
                    best = logits[(size_t)v];
                    tok = v;
                }
            }
            if (tok == kBlank)
                break;
            float maxl = best, sum = 0.0f;
            for (int v = 0; v < kVocab; v++)
                sum += expf(logits[(size_t)v] - maxl);
            emitted.push_back({tok, t, 1.0f / sum});
            m.emit(tok);
        }
    }
    return emitted;
}

// Scripted joint: frame `t` emits `script[t]` in order, then blank.
struct scripted_model {
    std::vector<std::vector<int>> script;
    int cur_t = -1;
    size_t k = 0;
    std::vector<int> emitted_to_predictor;

    void joint(int t, std::vector<float>& logits) {
        if (t != cur_t) {
            cur_t = t;
            k = 0;
        }
        logits.assign(kVocab, 0.0f);
        const int want = k < script[(size_t)t].size() ? script[(size_t)t][k] : kBlank;
        logits[(size_t)want] = 1.0f;
    }
    void emit(int tok) {
        emitted_to_predictor.push_back(tok);
        k++;
    }
};

std::vector<emission> run(scripted_model& m, int max_symbols) {
    return rnnt_greedy_loop(
        (int)m.script.size(), max_symbols, kBlank, [&](int t, std::vector<float>& l) { m.joint(t, l); },
        [&](int tok) { m.emit(tok); }, kNoHints, 0.0f);
}

// Table joint: frame `t`, step `k` (emissions so far in the frame) gives
// `rows[t][k]`; past the listed steps blank wins even over biased logits.
struct table_model {
    std::vector<std::vector<std::vector<float>>> rows;
    int cur_t = -1;
    size_t k = 0;

    void joint(int t, std::vector<float>& logits) {
        if (t != cur_t) {
            cur_t = t;
            k = 0;
        }
        if (k < rows[(size_t)t].size()) {
            logits = rows[(size_t)t][k];
        } else {
            logits.assign(kVocab, 0.0f);
            logits[kBlank] = 1000.0f; // above any bias
        }
    }
};

std::vector<emission> run(table_model& m, const Trie& trie, float boost) {
    return rnnt_greedy_loop(
        (int)m.rows.size(), 10, kBlank, [&](int t, std::vector<float>& l) { m.joint(t, l); },
        [&](int) { m.k++; }, trie, boost);
}

// One row: `hot` gets `value`, blank gets `blank`, the rest 0.
std::vector<float> row(int hot, float value, float blank) {
    std::vector<float> l(kVocab, 0.0f);
    l[(size_t)hot] = value;
    l[kBlank] = blank;
    return l;
}

Trie trie_of(const std::vector<std::vector<int32_t>>& forms, float boost) {
    Trie trie;
    for (const auto& f : forms)
        trie.insert(f, boost);
    trie.build_failure_links();
    return trie;
}

std::vector<int> ids(const std::vector<emission>& out) {
    std::vector<int> r;
    for (const auto& e : out)
        r.push_back(e.id);
    return r;
}

} // namespace

TEST_CASE("rnnt_greedy_loop matches the reference loop bit for bit", "[rnnt_greedy]") {
    size_t total = 0;
    for (uint32_t seed = 1; seed <= 20; seed++) {
        const auto ref = reference_loop(64, 10, synthetic_model{seed});
        synthetic_model m{seed};
        const auto got = rnnt_greedy_loop(
            64, 10, kBlank, [&](int t, std::vector<float>& l) { m.joint(t, l); }, [&](int tok) { m.emit(tok); },
            kNoHints, 4.0f);
        REQUIRE(got.size() == ref.size());
        for (size_t i = 0; i < ref.size(); i++) {
            CHECK(got[i].id == ref[i].id);
            CHECK(got[i].t == ref[i].t);
            CHECK(got[i].p == ref[i].p); // exact: same arithmetic, same order
        }
        total += ref.size();
    }
    CHECK(total > 0); // the synthetic joint does emit tokens
}

TEST_CASE("rnnt_greedy_loop: blank ends the frame", "[rnnt_greedy]") {
    scripted_model m{{{1}, {}, {2, 3}}};
    const auto out = run(m, 10);
    REQUIRE(out.size() == 3);
    CHECK(out[0].id == 1);
    CHECK(out[0].t == 0);
    CHECK(out[1].id == 2);
    CHECK(out[1].t == 2);
    CHECK(out[2].id == 3);
    CHECK(out[2].t == 2);
    CHECK(m.emitted_to_predictor == std::vector<int>{1, 2, 3});
}

TEST_CASE("rnnt_greedy_loop: max_symbols caps emissions per frame", "[rnnt_greedy]") {
    scripted_model m{{{1, 2, 3, 4, 1, 2}, {4}}};
    const auto out = run(m, 3);
    REQUIRE(out.size() == 4);
    CHECK(out[0].id == 1);
    CHECK(out[1].id == 2);
    CHECK(out[2].id == 3);
    CHECK(out[2].t == 0);
    CHECK(out[3].id == 4);
    CHECK(out[3].t == 1); // the next frame starts afresh
}

TEST_CASE("rnnt_greedy_loop: two equal tokens in a row are both emitted", "[rnnt_greedy]") {
    scripted_model m{{{3, 3}, {3}}};
    const auto out = run(m, 10);
    REQUIRE(out.size() == 3);
    for (const auto& e : out)
        CHECK(e.id == 3);
    CHECK(out[1].t == 0);
    CHECK(out[2].t == 1);
    // the predictor advances on each emission, not once per run of equal tokens
    CHECK(m.emitted_to_predictor == std::vector<int>{3, 3, 3});
}

// ---- Term hints (S6 § «Подсказки в greedy GigaAM») ----

TEST_CASE("rnnt_greedy_loop hints: the trie moves argmax onto the form", "[rnnt_greedy]") {
    // Without hints: 3, then 4. Form [1 2] at boost 4: 1 (0+4 > 1), then the
    // continuation 2 (0 + 4 + end boost 4 > 1).
    std::vector<std::vector<float>> frame = {row(3, 1.0f, 0.5f), row(4, 1.0f, 0.5f)};
    frame[0][1] = 0.0f;
    table_model plain{{frame}};
    CHECK(ids(run(plain, kNoHints, 4.0f)) == std::vector<int>{3, 4});
    table_model biased{{frame}};
    CHECK(ids(run(biased, trie_of({{1, 2}}, 4.0f), 4.0f)) == std::vector<int>{1, 2});
}

TEST_CASE("rnnt_greedy_loop hints: blank does not move the match", "[rnnt_greedy]") {
    // Frame 0 emits 1, then blank. Frame 1: 4 at 5.0 loses to the continuation
    // 2 (0 + 4 + 4) only if the match is still after 1.
    table_model m{{{row(1, 100.0f, 0.0f)}, {row(4, 5.0f, 0.0f)}}};
    CHECK(ids(run(m, trie_of({{1, 2}}, 4.0f), 4.0f)) == std::vector<int>{1, 2});
}

TEST_CASE("rnnt_greedy_loop hints: blank above every biased logit emits nothing", "[rnnt_greedy]") {
    // Forms [1 2 3] and [2 3]. After 1 2 the bias on 3 sums the fail chain:
    // node(1 2) → 3: 4 + end 8 (own 4 + suffix [2 3] 4); node(2) → 3: 4 + 4.
    // Total 20 > 12 of the current node alone.
    const Trie trie = trie_of({{1, 2, 3}, {2, 3}}, 4.0f);
    auto frames = [](float blank) {
        return std::vector<std::vector<std::vector<float>>>{
            {row(1, 100.0f, 0.0f), row(2, 100.0f, 0.0f), row(0, 0.0f, 1000.0f)}, {row(0, 0.0f, blank)}};
    };
    table_model above{frames(20.5f)};
    CHECK(ids(run(above, trie, 4.0f)) == std::vector<int>{1, 2});
    table_model below{frames(19.5f)};
    const auto out = run(below, trie, 4.0f);
    REQUIRE(out.size() >= 3);
    CHECK(out[2].id == 3);
    CHECK(out[2].t == 1);
}

TEST_CASE("rnnt_greedy_loop hints: an empty trie is bit for bit the unbiased loop", "[rnnt_greedy]") {
    for (uint32_t seed = 1; seed <= 20; seed++) {
        const auto ref = reference_loop(64, 10, synthetic_model{seed});
        synthetic_model m{seed};
        const Trie empty = trie_of({}, 4.0f);
        const auto got = rnnt_greedy_loop(
            64, 10, kBlank, [&](int t, std::vector<float>& l) { m.joint(t, l); }, [&](int tok) { m.emit(tok); },
            empty, 4.0f);
        REQUIRE(got.size() == ref.size());
        for (size_t i = 0; i < ref.size(); i++) {
            CHECK(got[i].id == ref[i].id);
            CHECK(got[i].p == ref[i].p);
        }
    }
}

TEST_CASE("rnnt_greedy_loop hints: the match resets between calls", "[rnnt_greedy]") {
    // Call 1 ends right after 1. In call 2, 4 at 5.0 beats 2 at 0: the form
    // [1 2] started in call 1 is not continued (2 is not a root child).
    const Trie trie = trie_of({{1, 2}}, 4.0f);
    table_model first{{{row(1, 100.0f, 0.0f)}}};
    CHECK(ids(run(first, trie, 4.0f)) == std::vector<int>{1});
    table_model second{{{row(4, 5.0f, 0.0f)}}};
    CHECK(ids(run(second, trie, 4.0f)) == std::vector<int>{4});
}
