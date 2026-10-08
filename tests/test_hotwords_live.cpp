// Context-Assist (T3.1, F8, F9, F1): hotword setters on real models.
//
// Requires:
//   CRISPASR_MODEL_PARAKEET        — Parakeet TDT 0.6B v3 q8_0 GGUF of T0.3 (sha256 checked)
//   CRISPASR_MODEL_GIGAAM          — GigaAM v3 e2e_rnnt q8_0 GGUF of T0.3 (sha256 checked)
//   CRISPASR_MODEL_GIGAAM_RNNT     — GigaAM v3 `rnnt` (charwise), models/convert-gigaam-to-gguf.py
//   CRISPASR_MODEL_GIGAAM_E2E_CTC  — GigaAM v3 `e2e_ctc`, models/convert-gigaam-to-gguf.py
//   CA_HOTWORDS_FILE        — comma-separated forms (`crispasr-spike stand --dump-hints`
//                             of Context-Assist: the 25 forms of enrollment.json)
// SKIPs (exit code 4) when either is missing; with CA_REQUIRE_MODELS=1 a missing
// input FAILs instead.

#include <catch2/catch_test_macros.hpp>

#include "core/asr_context_bias.h"
#include "crispasr_session.h"
#include "gigaam.h"
#include "parakeet.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <CommonCrypto/CommonDigest.h>
#endif

namespace {

const char* kParakeetV3Sha256 = "300de963db10e991a8c3c1674000245546f2e99d396f860aecbde2dd0534e43f";
const char* kGigaamV3E2eRnntSha256 = "3aa25ce8a3e8ea9ebc66fc1f1a650a8d515f85c1846c76aaba1af84c1ded56b0";

bool require_models() {
    const char* v = std::getenv("CA_REQUIRE_MODELS");
    return v && std::string(v) == "1";
}

// Path from the environment if readable; SKIP or FAIL (CA_REQUIRE_MODELS=1) otherwise.
std::string input_path(const char* var) {
    const char* p = std::getenv(var);
    if (p && *p) {
        if (FILE* f = fopen(p, "rb")) {
            fclose(f);
            return p;
        }
    }
    if (require_models())
        FAIL(var << " not set or not readable (CA_REQUIRE_MODELS=1)");
    SKIP(var << " not set or not readable");
    return {};
}

std::string sha256_file(const std::string& path) {
#ifdef __APPLE__
    std::ifstream in(path, std::ios::binary);
    CC_SHA256_CTX c;
    CC_SHA256_Init(&c);
    std::vector<char> buf(1 << 20);
    while (in.read(buf.data(), (std::streamsize)buf.size()) || in.gcount() > 0)
        CC_SHA256_Update(&c, buf.data(), (CC_LONG)in.gcount());
    unsigned char d[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(d, &c);
    std::string hex;
    char h[3];
    for (unsigned char b : d) {
        snprintf(h, sizeof h, "%02x", b);
        hex += h;
    }
    return hex;
#else
    return {}; // the reference model is checked on macOS only
#endif
}

std::string read_text(const std::string& path) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::string> read_forms(const std::string& path) {
    return core_context_bias::parse_hotwords(read_text(path));
}

} // namespace

TEST_CASE("parakeet_set_hotwords: enrollment forms on Parakeet v3", "[hotwords-live][parakeet]") {
    const std::string model = input_path("CRISPASR_MODEL_PARAKEET");
    const std::string hints = input_path("CA_HOTWORDS_FILE");
#ifdef __APPLE__
    REQUIRE(sha256_file(model) == kParakeetV3Sha256);
#endif
    const auto forms = read_forms(hints);
    REQUIRE(forms.size() == 25);
    std::vector<const char*> ptrs;
    for (const auto& f : forms)
        ptrs.push_back(f.c_str());

    auto params = parakeet_context_default_params();
    params.use_gpu = false; // only the vocab is used
    parakeet_context* ctx = parakeet_init_from_file(model.c_str(), params);
    REQUIRE(ctx != nullptr);

    // Reference count computed outside the fork (plan T3.1, «Данные»): all 25 covered.
    CHECK(parakeet_set_hotwords(ctx, ptrs.data(), (int)ptrs.size(), 4.0f) == 25);
    CHECK(parakeet_set_hotwords(ctx, nullptr, 0, 0.0f) == 0);
    CHECK(parakeet_set_hotwords(nullptr, ptrs.data(), (int)ptrs.size(), 4.0f) == -1);
    parakeet_free(ctx);
}

TEST_CASE("gigaam_set_hotwords: enrollment forms on GigaAM v3 e2e_rnnt", "[hotwords-live][gigaam]") {
    const std::string model = input_path("CRISPASR_MODEL_GIGAAM");
    const std::string hints = input_path("CA_HOTWORDS_FILE");
#ifdef __APPLE__
    REQUIRE(sha256_file(model) == kGigaamV3E2eRnntSha256);
#endif
    const auto forms = read_forms(hints);
    REQUIRE(forms.size() == 25);
    std::vector<const char*> ptrs;
    for (const auto& f : forms)
        ptrs.push_back(f.c_str());

    auto params = gigaam_context_default_params();
    params.use_gpu = false; // only the vocab is used
    gigaam_context* ctx = gigaam_init_from_file(model.c_str(), params);
    REQUIRE(ctx != nullptr);
    REQUIRE(gigaam_is_rnnt(ctx) == 1);
    REQUIRE(gigaam_is_spm(ctx) == 1);

    // Reference count computed outside the fork (plan T3.1, «Данные»): 24 of 25,
    // «ранбук» is dropped (its first piece is not covered).
    CHECK(gigaam_set_hotwords(ctx, ptrs.data(), (int)ptrs.size(), 4.0f) == 24);
    CHECK(gigaam_set_hotwords(ctx, nullptr, 0, 0.0f) == 0);
    CHECK(gigaam_set_hotwords(ctx, ptrs.data(), 0, 4.0f) == 0);
    CHECK(gigaam_set_hotwords(nullptr, ptrs.data(), (int)ptrs.size(), 4.0f) == -1);
    gigaam_free(ctx);
}

// ─── Session (F1): codes, inserted count, trie cache — S6 § «Сессия и коды возврата»

extern "C" int crispasr_session_hotwords_builds(crispasr_session* s); // test build only

namespace {

// Forms the model vocab cannot take whole: `ыы` starts with a lone "▁" (neither
// vocab has a piece "▁ы…"), `☃` is in neither vocab.
const char* kLoneSepForm = "ыы";
const char* kUncoveredForm = "☃";

// A hint-capable session: enrollment forms in, `expected` of them inserted.
void check_hint_session(const std::string& model, int expected) {
    const std::string hints = read_text(input_path("CA_HOTWORDS_FILE"));
    crispasr_session* s = crispasr_session_open(model.c_str(), 4);
    REQUIRE(s != nullptr);

    CHECK(crispasr_session_set_hotwords(s, hints.c_str(), 4.0f) == 0);
    CHECK(crispasr_session_hotwords_inserted(s) == expected);
    const int builds = crispasr_session_hotwords_builds(s);
    CHECK(builds == 1);

    // Same pair: cached, no rebuild. Other boost: rebuild.
    CHECK(crispasr_session_set_hotwords(s, hints.c_str(), 4.0f) == 0);
    CHECK(crispasr_session_hotwords_builds(s) == builds);
    CHECK(crispasr_session_set_hotwords(s, hints.c_str(), 5.0f) == 0);
    CHECK(crispasr_session_hotwords_builds(s) == builds + 1);
    CHECK(crispasr_session_hotwords_inserted(s) == expected);

    // A form with a lone "▁" first piece or an uncovered char is dropped whole.
    for (const char* bad : {kLoneSepForm, kUncoveredForm}) {
        const std::string more = hints + "," + bad;
        CHECK(crispasr_session_set_hotwords(s, more.c_str(), 4.0f) == 0);
        CHECK(crispasr_session_hotwords_inserted(s) == expected);
    }

    // Empty string clears and resets the cache: the same pair builds again.
    CHECK(crispasr_session_set_hotwords(s, "", 4.0f) == 0);
    CHECK(crispasr_session_hotwords_inserted(s) == 0);
    const int before = crispasr_session_hotwords_builds(s);
    CHECK(crispasr_session_set_hotwords(s, hints.c_str(), 4.0f) == 0);
    CHECK(crispasr_session_hotwords_builds(s) == before + 1);
    CHECK(crispasr_session_hotwords_inserted(s) == expected);

    CHECK(crispasr_session_set_hotwords(nullptr, hints.c_str(), 4.0f) == -1);
    CHECK(crispasr_session_hotwords_inserted(nullptr) == -1);
    crispasr_session_close(s);
}

// GigaAM without hint support: a non-empty string is refused, the empty one is not.
void check_no_hint_gigaam(const std::string& model) {
    const std::string hints = read_text(input_path("CA_HOTWORDS_FILE"));
    crispasr_session* s = crispasr_session_open(model.c_str(), 4);
    REQUIRE(s != nullptr);
    REQUIRE(std::string(crispasr_session_backend(s)) == "gigaam");

    CHECK(crispasr_session_set_hotwords(s, hints.c_str(), 4.0f) == -2);
    CHECK(crispasr_session_hotwords_inserted(s) == 0);
    CHECK(crispasr_session_hotwords_builds(s) == 0);
    CHECK(crispasr_session_set_hotwords(s, "", 4.0f) == 0);
    CHECK(crispasr_session_set_beam_size(s, 4) == -2);
    CHECK(crispasr_session_set_beam_size(s, 1) == 0);
    crispasr_session_close(s);
}

} // namespace

TEST_CASE("session hotwords: Parakeet v3", "[hotwords-live][parakeet]") {
    const std::string model = input_path("CRISPASR_MODEL_PARAKEET");
    check_hint_session(model, 25);
    crispasr_session* s = crispasr_session_open(model.c_str(), 4);
    REQUIRE(s != nullptr);
    CHECK(crispasr_session_set_beam_size(s, 4) == 0); // as before F1
    crispasr_session_close(s);
}

TEST_CASE("session hotwords and beam: GigaAM v3 e2e_rnnt", "[hotwords-live][gigaam]") {
    const std::string model = input_path("CRISPASR_MODEL_GIGAAM");
    check_hint_session(model, 24);
    crispasr_session* s = crispasr_session_open(model.c_str(), 4);
    REQUIRE(s != nullptr);
    CHECK(crispasr_session_set_beam_size(s, 4) == -2);
    CHECK(crispasr_session_set_beam_size(s, 1) == 0);
    CHECK(crispasr_session_set_beam_size(s, 0) == 0);
    crispasr_session_close(s);
}

TEST_CASE("session hotwords: GigaAM v3 rnnt (charwise) refuses", "[hotwords-live][gigaam-rnnt]") {
    const std::string model = input_path("CRISPASR_MODEL_GIGAAM_RNNT");
    gigaam_context* ctx = gigaam_init_from_file(model.c_str(), [] {
        auto p = gigaam_context_default_params();
        p.use_gpu = false;
        return p;
    }());
    REQUIRE(ctx != nullptr);
    CHECK(gigaam_is_rnnt(ctx) == 1);
    CHECK(gigaam_is_spm(ctx) == 0);
    gigaam_free(ctx);
    check_no_hint_gigaam(model);
}

TEST_CASE("session hotwords: GigaAM v3 e2e_ctc refuses", "[hotwords-live][gigaam-e2e-ctc]") {
    const std::string model = input_path("CRISPASR_MODEL_GIGAAM_E2E_CTC");
    gigaam_context* ctx = gigaam_init_from_file(model.c_str(), [] {
        auto p = gigaam_context_default_params();
        p.use_gpu = false;
        return p;
    }());
    REQUIRE(ctx != nullptr);
    CHECK(gigaam_is_rnnt(ctx) == 0);
    CHECK(gigaam_is_spm(ctx) == 1);
    gigaam_free(ctx);
    check_no_hint_gigaam(model);
}
