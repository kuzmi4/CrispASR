// tests/test_vad_failure.cpp — Context-Assist (T3.1, F6): a failed Silero VAD
// window is an error, not silence. Model: models/for-tests-silero-v6.2.0-ggml.bin,
// audio: samples/jfk.wav (both in the repo; WORKING_DIRECTORY is the source root).
// Compute failure goes through the real status check via
// CRISPASR_VAD_SIMULATE_COMPUTE_FAILURE_AT_WINDOW; NaN/Inf and the graph
// allocation failure use the test-build seam crispasr_test_vad_inject.

#include <catch2/catch_test_macros.hpp>
#include "crispasr.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

extern "C" void crispasr_test_vad_inject(bool fail_alloc, int nonfinite_window, float value);

namespace {

const char* kModel = "models/for-tests-silero-v6.2.0-ggml.bin";
const char* kAudio = "samples/jfk.wav";
const char* kSimVar = "CRISPASR_VAD_SIMULATE_COMPUTE_FAILURE_AT_WINDOW";

struct Pcm {
    float* data = nullptr;
    int n = 0;
    Pcm() {
        int sr = 0;
        REQUIRE(crispasr_audio_load(kAudio, &data, &n, &sr) == 0);
        REQUIRE(sr == 16000);
    }
    ~Pcm() { crispasr_audio_free(data); }
};

whisper_vad_context* open_vad() {
    whisper_vad_context_params cp = whisper_vad_default_context_params();
    cp.n_threads = 1;
    cp.use_gpu = false;
    cp.gpu_device = 0;
    whisper_vad_context* v = whisper_vad_init_from_file_with_params(kModel, cp);
    REQUIRE(v != nullptr);
    return v;
}

// S2 parameters (speech_pad_ms=0, samples_overlap=0).
whisper_vad_params params() {
    whisper_vad_params p = whisper_vad_default_params();
    p.threshold = 0.35f;
    p.min_speech_duration_ms = 80;
    p.min_silence_duration_ms = 500;
    p.max_speech_duration_s = 20.0f;
    p.speech_pad_ms = 0;
    p.samples_overlap = 0.0f;
    return p;
}

// Probabilities and segment bounds of one successful pass.
struct Pass {
    std::vector<float> probs;
    std::vector<float> bounds;
    bool operator==(const Pass& o) const { return probs == o.probs && bounds == o.bounds; }
};

Pass run_ok(whisper_vad_context* v, const Pcm& pcm) {
    whisper_vad_segments* s = whisper_vad_segments_from_samples(v, params(), pcm.data, pcm.n);
    REQUIRE(s != nullptr);
    Pass p;
    const float* pr = whisper_vad_probs(v);
    p.probs.assign(pr, pr + whisper_vad_n_probs(v));
    for (int i = 0; i < whisper_vad_segments_n_segments(s); i++) {
        p.bounds.push_back(whisper_vad_segments_get_segment_t0(s, i));
        p.bounds.push_back(whisper_vad_segments_get_segment_t1(s, i));
    }
    whisper_vad_free_segments(s);
    return p;
}

// A failed pass: detect=false, no leftover probabilities, segments NULL.
void require_failed(whisper_vad_context* v, const Pcm& pcm) {
    CHECK_FALSE(whisper_vad_detect_speech(v, pcm.data, pcm.n));
    CHECK(whisper_vad_n_probs(v) == 0);
    CHECK(whisper_vad_segments_from_samples(v, params(), pcm.data, pcm.n) == nullptr);
}

struct ScopedEnv {
    const char* name;
    ScopedEnv(const char* n, const char* v) : name(n) { setenv(n, v, 1); }
    ~ScopedEnv() { unsetenv(name); }
};

struct ScopedInject {
    ScopedInject(bool fail_alloc, int window, float value) { crispasr_test_vad_inject(fail_alloc, window, value); }
    ~ScopedInject() { crispasr_test_vad_inject(false, -1, 0.0f); }
};

} // namespace

TEST_CASE("VAD failure is an error, not silence", "[vad_failure]") {
    unsetenv(kSimVar);
    Pcm pcm;
    const int n_windows = (pcm.n + 511) / 512;
    REQUIRE(n_windows > 2);
    const std::string mid = std::to_string(n_windows / 2);

    whisper_vad_context* clean_ctx = open_vad();
    const Pass clean = run_ok(clean_ctx, pcm);
    whisper_vad_free(clean_ctx);
    REQUIRE(clean.probs.size() == (size_t)n_windows);
    REQUIRE_FALSE(clean.bounds.empty()); // speech found: a failure cannot look like it

    whisper_vad_context* v = open_vad();

    SECTION("compute failure at the first and an intermediate window") {
        for (const std::string& at : {std::string("0"), mid}) {
            CAPTURE(at);
            REQUIRE(run_ok(v, pcm) == clean); // a successful pass of the same length first
            {
                ScopedEnv sim(kSimVar, at.c_str());
                require_failed(v, pcm);
            }
            CHECK(run_ok(v, pcm) == clean);
        }
    }

    SECTION("non-finite probability at the first and an intermediate window") {
        const float bad[] = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()};
        for (int at : {0, n_windows / 2}) {
            for (float value : bad) {
                CAPTURE(at, value);
                REQUIRE(run_ok(v, pcm) == clean);
                {
                    ScopedInject inject(false, at, value);
                    require_failed(v, pcm);
                }
                CHECK(run_ok(v, pcm) == clean);
            }
        }
    }

    SECTION("graph allocation failure stays false") {
        REQUIRE(run_ok(v, pcm) == clean);
        {
            ScopedInject inject(true, -1, 0.0f);
            require_failed(v, pcm);
        }
        CHECK(run_ok(v, pcm) == clean);
    }

    whisper_vad_free(v);
}
