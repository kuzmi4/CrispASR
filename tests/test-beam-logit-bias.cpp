// tests/test-beam-logit-bias.cpp — Config::logit_bias seam of the branched beam
// helper: the callback must see each live beam's own suffix, and a bias applied
// before the top-K must be able to change the winning sequence (that is how
// Canary term hints reach beam search).

#include <catch2/catch_test_macros.hpp>

#include "core/beam_decode.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int V = 6;
constexpr int EOS = 5;

// Toy decoder: flat logits, token 0 best, EOS last so no beam finishes early.
float* toy_logits() {
    float* lg = (float*)std::malloc(sizeof(float) * V);
    for (int v = 0; v < V; v++)
        lg[v] = 5.0f - (float)v;
    lg[EOS] = -2.0f;
    return lg;
}

struct Ctx {
    std::vector<int32_t> seq;
};

core_beam_decode::Config make_cfg() {
    core_beam_decode::Config cfg;
    cfg.semantics = core_beam_decode::Semantics::Legacy;
    cfg.vocab_size = V;
    cfg.beam_size = 3;
    cfg.max_new_tokens = 2;
    cfg.eos_id = EOS;
    cfg.prompt_len = 1;
    return cfg;
}

std::vector<int32_t> run(const core_beam_decode::Config& cfg) {
    Ctx ctx{{7}}; // prompt token
    auto save = [](Ctx* x) { return new std::vector<int32_t>(x->seq); };
    auto restore = [](Ctx* x, std::vector<int32_t>* s) { x->seq = *s; };
    auto snap_free = [](std::vector<int32_t>* s) { delete s; };
    auto step = [](Ctx* x, int32_t tok, int n_past) {
        REQUIRE((int)x->seq.size() == n_past);
        x->seq.push_back(tok);
        return toy_logits();
    };
    float* prefill = toy_logits();
    auto r = core_beam_decode::run_with_probs_branched(&ctx, prefill, save, restore, snap_free, step, cfg);
    std::free(prefill);
    return r.tokens;
}

} // namespace

TEST_CASE("beam without logit_bias keeps the unbiased best beam", "[beam]") {
    CHECK(run(make_cfg()) == std::vector<int32_t>{0, 0});
}

TEST_CASE("beam logit_bias sees the beam suffix and steers the top-K", "[beam]") {
    auto cfg = make_cfg();
    // Boosting token 3 costs the same as the best two tokens; it only survives
    // the top-K pruning if the callback ran on that beam with the suffix {0}.
    cfg.logit_bias = [](const int32_t* toks, int n, float* lg) {
        REQUIRE(toks != nullptr);
        REQUIRE(n >= 1);
        if (n >= 1 && toks[0] == 0)
            lg[3] = 100.0f; // only the beam that started on token 0 is boosted
    };
    CHECK(run(cfg) == std::vector<int32_t>{0, 3});
}

TEST_CASE("beam logit_bias is called per beam with that beam's own suffix", "[beam]") {
    auto cfg = make_cfg();
    std::vector<int> lengths;
    cfg.logit_bias = [&lengths](const int32_t*, int n, float*) { lengths.push_back(n); };
    run(cfg);
    // Three seeded beams, each expanded once per round.
    REQUIRE(!lengths.empty());
    for (int n : lengths)
        CHECK(n == 1);
}
