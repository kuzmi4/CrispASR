// tests/test_context_bias.cpp — Catch2 unit tests for CTC-WS phrase-boost trie.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "core/asr_context_bias.h"

#include <algorithm>
#include <numeric>
#include <vector>
#include <string>

using namespace core_context_bias;

// ── Helpers ──────────────────────────────────────────────────────────

// Trivial "tokenizer": each character maps to its ASCII value.
static std::vector<int32_t> char_tokenize(const std::string& s) {
    std::vector<int32_t> ids;
    for (char c : s) ids.push_back((int32_t)(unsigned char)c);
    return ids;
}

// ── Trie construction ────────────────────────────────────────────────

TEST_CASE("empty trie", "[context_bias]") {
    Trie trie;
    REQUIRE(trie.empty());

    MatchState st;
    std::vector<float> logits(256, 0.0f);
    apply_bias(trie, st, logits.data(), 256, 2.0f, -1);
    // All logits should remain zero — no bias applied
    float sum = std::accumulate(logits.begin(), logits.end(), 0.0f);
    REQUIRE(sum == 0.0f);
}

TEST_CASE("single hotword insert + match", "[context_bias]") {
    Trie trie;
    // Insert "cat" as token IDs [99, 97, 116]
    trie.insert({99, 97, 116}, 5.0f);
    trie.build_failure_links();
    REQUIRE_FALSE(trie.empty());

    MatchState st;
    const int V = 256;
    const float boost = 2.0f;

    // At root nothing is boosted: a hint does not start a hotword
    std::vector<float> logits(V, 0.0f);
    apply_bias(trie, st, logits.data(), V, boost, -1);
    REQUIRE(logits[99] == 0.0f); // 'c' not boosted
    REQUIRE(logits[97] == 0.0f);

    // Advance with 'c' (99)
    advance(trie, st, 99);
    std::fill(logits.begin(), logits.end(), 0.0f);
    apply_bias(trie, st, logits.data(), V, boost, -1);
    REQUIRE(logits[97] > 0.0f); // 'a' boosted (continues "ca")
    // a new "cat" start via the failure link to the root is not boosted
    REQUIRE(logits[99] == 0.0f);

    // Advance with 'a' (97)
    advance(trie, st, 97);
    std::fill(logits.begin(), logits.end(), 0.0f);
    apply_bias(trie, st, logits.data(), V, boost, -1);
    // 't' (116) should get boost + endpoint boost (5.0)
    REQUIRE(logits[116] >= boost + 5.0f);

    // Advance with 't' — hotword complete, back to root-ish
    advance(trie, st, 116);
}

TEST_CASE("multiple hotwords with overlap", "[context_bias]") {
    Trie trie;
    // "ab" and "abc"
    trie.insert({1, 2}, 3.0f);
    trie.insert({1, 2, 3}, 5.0f);
    trie.build_failure_links();

    MatchState st;
    const int V = 10;
    const float boost = 1.0f;

    // Root: nothing is boosted
    std::vector<float> logits(V, 0.0f);
    apply_bias(trie, st, logits.data(), V, boost, -1);
    REQUIRE(logits[1] == 0.0f);

    advance(trie, st, 1);
    std::fill(logits.begin(), logits.end(), 0.0f);
    apply_bias(trie, st, logits.data(), V, boost, -1);
    // Token 2 ('b') should be boosted
    REQUIRE(logits[2] > 0.0f);

    advance(trie, st, 2);
    std::fill(logits.begin(), logits.end(), 0.0f);
    apply_bias(trie, st, logits.data(), V, boost, -1);
    // Token 3 ('c') should be boosted (continues "abc")
    REQUIRE(logits[3] > 0.0f);
}

TEST_CASE("mismatch resets via failure links", "[context_bias]") {
    Trie trie;
    trie.insert({1, 2, 3}, 5.0f);
    trie.build_failure_links();

    MatchState st;

    // Match prefix "1, 2"
    advance(trie, st, 1);
    advance(trie, st, 2);

    // Now mismatch with token 9 — should reset
    advance(trie, st, 9);
    REQUIRE(st.node == 0); // back at root

    // Back at the root nothing is boosted until the decoder starts the form
    std::vector<float> logits(10, 0.0f);
    apply_bias(trie, st, logits.data(), 10, 1.0f, -1);
    REQUIRE(logits[1] == 0.0f);
}

TEST_CASE("failure link suffix match", "[context_bias]") {
    Trie trie;
    // "abc" and "bc"
    trie.insert({1, 2, 3}, 5.0f);
    trie.insert({2, 3}, 3.0f);
    trie.build_failure_links();

    MatchState st;
    // Start matching "a, b"
    advance(trie, st, 1); // in "a" prefix
    advance(trie, st, 2); // in "ab" prefix

    // Now mismatch with 9 — should fall back.
    // But "b" is a prefix of "bc", so via failure links we might land
    // at the "b" node of the "bc" pattern.
    advance(trie, st, 9);
    // After mismatch, we should be back at root
    // (9 is not a child of any node in the "bc" path either)
}

// ── parse_hotwords ───────────────────────────────────────────────────

TEST_CASE("parse_hotwords basic", "[context_bias]") {
    auto hw = parse_hotwords("Acme Corp, Sandra Berenz, GPU-PB");
    REQUIRE(hw.size() == 3);
    REQUIRE(hw[0] == "Acme Corp");
    REQUIRE(hw[1] == "Sandra Berenz");
    REQUIRE(hw[2] == "GPU-PB");
}

TEST_CASE("parse_hotwords with boost suffix", "[context_bias]") {
    auto hw = parse_hotwords("Berenz^5.0, NVIDIA^3.0, plain");
    REQUIRE(hw.size() == 3);
    REQUIRE(hw[0] == "Berenz^5.0");
    REQUIRE(hw[1] == "NVIDIA^3.0");
    REQUIRE(hw[2] == "plain");
}

TEST_CASE("parse_hotwords empty and whitespace", "[context_bias]") {
    REQUIRE(parse_hotwords("").empty());
    REQUIRE(parse_hotwords("  ,  ,  ").empty());
    auto hw = parse_hotwords("  foo  ,  bar  ");
    REQUIRE(hw.size() == 2);
    REQUIRE(hw[0] == "foo");
    REQUIRE(hw[1] == "bar");
}

// ── build_trie convenience ───────────────────────────────────────────

TEST_CASE("build_trie with char tokenizer", "[context_bias]") {
    auto trie = build_trie({"cat", "car"}, char_tokenize, 2.0f);
    REQUIRE_FALSE(trie.empty());
    // Should have nodes for: root → c → a → t, and c → a → r
    // At least 5 nodes: root, c, a, t, r
    REQUIRE(trie.nodes.size() >= 5);
}

TEST_CASE("build_trie with boost suffix", "[context_bias]") {
    auto trie = build_trie({"cat^10.0"}, char_tokenize, 2.0f);
    REQUIRE_FALSE(trie.empty());

    MatchState st;
    advance(trie, st, 'c');
    advance(trie, st, 'a');

    // At "ca" node, 't' should get boost + 10.0 endpoint
    std::vector<float> logits(256, 0.0f);
    apply_bias(trie, st, logits.data(), 256, 1.5f, -1);
    REQUIRE(logits['t'] >= 11.5f); // 1.5 prefix + 10.0 endpoint
}

// ── edge cases ───────────────────────────────────────────────────────

TEST_CASE("single-token hotword", "[context_bias]") {
    Trie trie;
    trie.insert({42}, 5.0f);
    trie.build_failure_links();

    MatchState st;
    std::vector<float> logits(100, 0.0f);
    apply_bias(trie, st, logits.data(), 100, 1.0f, -1);
    // A one-token hotword starts at the root, so it is never boosted
    REQUIRE(logits[42] == 0.0f);
}

TEST_CASE("token ID out of vocab range ignored", "[context_bias]") {
    Trie trie;
    trie.insert({999}, 5.0f); // token 999 > vocab size 100
    trie.build_failure_links();

    MatchState st;
    std::vector<float> logits(100, 0.0f);
    apply_bias(trie, st, logits.data(), 100, 1.0f, -1);
    // Should not crash, and no logit should be modified
    float sum = std::accumulate(logits.begin(), logits.end(), 0.0f);
    REQUIRE(sum == 0.0f);
}

TEST_CASE("repeated advance and reset", "[context_bias]") {
    Trie trie;
    trie.insert({1, 2, 3}, 5.0f);
    trie.build_failure_links();

    MatchState st;

    // Match full sequence twice
    for (int rep = 0; rep < 2; rep++) {
        advance(trie, st, 1);
        advance(trie, st, 2);
        advance(trie, st, 3);
    }

    // Manual reset
    st.reset();
    REQUIRE(st.node == 0);
}

// ── tokenize_spm_form (F8) ───────────────────────────────────────────

namespace {
const std::string U = "\xE2\x96\x81"; // ▁

// Synthetic SentencePiece vocab; the index is the token id.
const std::vector<std::string> kVocab = {
    "<unk>",  // 0
    U,        // 1
    U + "к",  // 2
    "ли",     // 3
    "к",      // 4
    "ха",     // 5
    "у",      // 6
    "с",      // 7
    U + "ка", // 8
    "ф",      // 9
    "а",      // 10
    U + "Ka", // 11
    "fka",    // 12
    U + "ka", // 13
    "f",      // 14
    U + "x",  // 15 — blank_id in these tests
    "x",      // 16
    "<",      // 17
    "unk>",   // 18
};
const int32_t kBlank = 15;

std::vector<int32_t> tok(const std::string& form) {
    return tokenize_spm_form(kVocab, kBlank, form);
}
} // namespace

TEST_CASE("tokenize_spm_form: leading ▁, greedy longest piece, UTF-8", "[context_bias]") {
    // ▁ка beats ▁к; Cyrillic pieces are multi-byte.
    REQUIRE(tok("кафка") == std::vector<int32_t>{8, 9, 4, 10});
}

TEST_CASE("tokenize_spm_form: space, tab and repeated spaces become one ▁", "[context_bias]") {
    // "клик хаус": no ▁х piece, so the inner separator is a lone ▁ (open decision 1
    // of the T3.1 plan) — the form stays.
    const std::vector<int32_t> want = {2, 3, 4, 1, 5, 6, 7};
    REQUIRE(tok("клик хаус") == want);
    REQUIRE(tok("клик\tхаус") == want);
    REQUIRE(tok("клик  \t хаус") == want);
}

TEST_CASE("tokenize_spm_form: uncovered character drops the whole form", "[context_bias]") {
    REQUIRE(tok("кафкаz").empty());
    REQUIRE(tok("кz").empty());
}

TEST_CASE("tokenize_spm_form: <…> pieces and blank_id never match", "[context_bias]") {
    // "x" after ▁ would match ▁x (blank_id) — must fall to the lone ▁ + "x",
    // and a lone ▁ as the first piece drops the form.
    REQUIRE(tok("x").empty());
    // "<unk>" in the form is not the <unk> piece: it is covered by "<" + "unk>"
    // only after a lone ▁ first piece — dropped as well.
    REQUIRE(tok("<unk>").empty());
    // Inside a form: ▁к + "<" + "unk>" (never id 0).
    REQUIRE(tok("к<unk>") == std::vector<int32_t>{2, 17, 18});
}

TEST_CASE("tokenize_spm_form: lone ▁ as the first piece drops the form", "[context_bias]") {
    // No ▁л piece: the form would start with a lone ▁.
    REQUIRE(tok("ли").empty());
}

TEST_CASE("tokenize_spm_form: case is kept", "[context_bias]") {
    REQUIRE(tok("Kafka") == std::vector<int32_t>{11, 12});
    REQUIRE(tok("kafka") == std::vector<int32_t>{13, 12});
}

TEST_CASE("blank gate: no bias when blank wins unbiased", "[context_bias]") {
    Trie trie;
    trie.insert({1, 2}, 5.0f);
    trie.build_failure_links();
    MatchState st;
    advance(trie, st, 1); // the decoder started the form on its own
    const int V = 10, blank = 9;

    // Pause: blank is the unbiased argmax — logits stay untouched.
    std::vector<float> pause(V, 0.0f);
    pause[blank] = 0.5f;
    apply_bias(trie, st, pause.data(), V, 4.0f, blank);
    REQUIRE(pause[2] == 0.0f);

    // Speech: another token wins — the continuation is boosted.
    std::vector<float> speech(V, 0.0f);
    speech[blank] = 0.5f;
    speech[3] = 1.0f;
    apply_bias(trie, st, speech.data(), V, 4.0f, blank);
    REQUIRE(speech[2] > speech[3]);

    // -1 disables the gate.
    std::vector<float> off(V, 0.0f);
    off[blank] = 0.5f;
    apply_bias(trie, st, off.data(), V, 4.0f, -1);
    REQUIRE(off[2] > off[blank]);
}

// ── Text mode (fold_case) ────────────────────────────────────────────

namespace {
// A character-level vocab like nemotron's: the hint `Сре` is ▁С ре, the
// decoder's own spelling is ▁С Р Е.
const std::string S = "\xE2\x96\x81"; // ▁
const std::vector<std::string> kText = {
    "<blk>",   // 0 — blank
    S + "С",   // 1
    "ре",      // 2
    "Р",       // 3
    "Е",       // 4
    S + "и",   // 5
    "н",       // 6
    "р",       // 7
    "е",       // 8
    S,         // 9
    "э",       // 10
    S + "с",   // 11
    "<0xD0>",  // 12
};
const int32_t kTextBlank = 0;

Trie text_trie(const std::vector<std::string>& forms, float boost) {
    Trie trie;
    for (const auto& f : forms)
        trie.insert(tokenize_spm_form(kText, kTextBlank, f), boost);
    trie.build_failure_links();
    trie.fold_case(kText, kTextBlank);
    return trie;
}

std::vector<float> biased(const Trie& trie, const MatchState& st, float boost) {
    std::vector<float> logits(kText.size(), 0.0f);
    logits[5] = 0.1f; // a non-blank wins: the blank gate stays open
    apply_bias(trie, st, logits.data(), (int)logits.size(), boost, kTextBlank);
    logits[5] -= 0.1f;
    return logits;
}
} // namespace

TEST_CASE("fold_case: the decoder's own spelling of a form is boosted", "[context_bias]") {
    // Recheo dictation: the hint `Сре` pulled `▁С Р Е` onto `ре` → "Сренженер".
    const Trie trie = text_trie({"Сре"}, 5.0f);
    MatchState st;
    REQUIRE(biased(trie, st, 2.0f)[1] == 0.0f); // the start is never boosted

    advance(trie, st, 1); // ▁С
    auto lg = biased(trie, st, 2.0f);
    // ре (the hint's spelling), Р and р alike: the decoder keeps its own
    // spelling; ре ending the form at once gets no endpoint bonus over Р.
    REQUIRE(lg[2] == 2.0f);
    REQUIRE(lg[3] == 2.0f);
    REQUIRE(lg[7] == 2.0f);
    REQUIRE(lg[5] == 0.0f); // ▁и does not continue the form

    advance(trie, st, 3); // Р
    lg = biased(trie, st, 2.0f);
    REQUIRE(lg[4] == 2.0f); // Е ends the form
    REQUIRE(lg[8] == 2.0f);
    REQUIRE(lg[2] == 0.0f); // ре would read "срре"

    advance(trie, st, 4); // Е — the form is complete, nothing continues it
    lg = biased(trie, st, 2.0f);
    REQUIRE(std::all_of(lg.begin(), lg.end(), [](float x) { return x == 0.0f; }));
}

TEST_CASE("fold_case: case variants are one form", "[context_bias]") {
    const Trie trie = text_trie({"Сре", "сре"}, 5.0f);
    MatchState st;
    advance(trie, st, 11); // ▁с starts the same folded form
    REQUIRE(biased(trie, st, 2.0f)[2] == 2.0f); // boosted once, not per variant
}

TEST_CASE("fold_case: a broken match restarts on the same token", "[context_bias]") {
    const Trie trie = text_trie({"Сре"}, 5.0f);
    MatchState st;
    advance(trie, st, 1); // ▁С
    advance(trie, st, 5); // ▁и — off the form
    REQUIRE(st.node == 0);
    advance(trie, st, 1); // ▁С again
    REQUIRE(biased(trie, st, 2.0f)[2] == 2.0f);
    advance(trie, st, 12); // a byte piece never matches
    REQUIRE(st.node == 0);
}

TEST_CASE("fold_case: a lone ▁ does not start a form", "[context_bias]") {
    // ▁С is a piece, so `Сэ` is covered; a decoder at a lone ▁ must not get
    // the second letter of every form boosted.
    const Trie trie = text_trie({"Сре"}, 5.0f);
    MatchState st;
    advance(trie, st, 9); // ▁
    REQUIRE(st.node != 0);
    const auto lg = biased(trie, st, 2.0f);
    REQUIRE(std::all_of(lg.begin(), lg.end(), [](float x) { return x == 0.0f; }));
}

TEST_CASE("fold_case: the blank gate holds in text mode", "[context_bias]") {
    const Trie trie = text_trie({"Сре"}, 5.0f);
    MatchState st;
    advance(trie, st, 1);
    std::vector<float> pause(kText.size(), 0.0f);
    pause[kTextBlank] = 0.5f;
    apply_bias(trie, st, pause.data(), (int)pause.size(), 2.0f, kTextBlank);
    REQUIRE(pause[2] == 0.0f);
}

TEST_CASE("fold_utf8: ASCII and Cyrillic fold, ё stays apart from е", "[context_bias]") {
    REQUIRE(fold_utf8("SRE") == U"sre");
    REQUIRE(fold_utf8("СРЕ") == U"сре");
    REQUIRE(fold_utf8("Ёж") == U"ёж");
    REQUIRE(fold_utf8("ё") != fold_utf8("е"));
    REQUIRE(fold_utf8("\xD0").empty());
}
