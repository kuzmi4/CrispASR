// Context-Assist (T3.1, F8): hotword setters on real models.
//
// Requires:
//   CRISPASR_MODEL_PARAKEET — Parakeet TDT 0.6B v3 q8_0 GGUF of T0.3 (sha256 checked)
//   CA_HOTWORDS_FILE        — comma-separated forms (`crispasr-spike stand --dump-hints`
//                             of Context-Assist: the 25 forms of enrollment.json)
// SKIPs (exit code 4) when either is missing; with CA_REQUIRE_MODELS=1 a missing
// input FAILs instead.

#include <catch2/catch_test_macros.hpp>

#include "core/asr_context_bias.h"
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

std::vector<std::string> read_forms(const std::string& path) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return core_context_bias::parse_hotwords(ss.str());
}

} // namespace

TEST_CASE("parakeet_set_hotwords: enrollment forms on Parakeet v3", "[hotwords-live]") {
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
