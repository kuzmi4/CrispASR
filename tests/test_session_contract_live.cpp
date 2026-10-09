// Context-Assist (T3.1, F5, F2): CRISPASR_SIMULATE_* failure injection and the
// actual session device on real models.
//
// Requires:
//   CRISPASR_MODEL_PARAKEET — Parakeet TDT 0.6B v3 GGUF
//   CRISPASR_MODEL_GIGAAM   — GigaAM v3 e2e_rnnt GGUF
// Audio: samples/jfk.wav (WORKING_DIRECTORY is the source root). Metal is assumed
// available (macOS arm64). SKIPs (exit code 4) when the model is missing; with
// CA_REQUIRE_MODELS=1 a missing model FAILs instead.
//
// What each variable must do (S6 § «Сводные требования к форку», F5), only with
// use_gpu=1: GPU_OPEN_FAILURE — open returns NULL; GPU_COMPUTE_FAILURE — the
// first transcribe* computes nothing (the session outcome and the error
// category are F3), the next call is normal; GPU_INIT_FALLBACK — the session
// opens and transcribes as a CPU session does, crispasr_session_device = 0 (F2:
// 1 — Metal, 0 — CPU, -1 — NULL); HANG_OPEN /
// HANG_CALL — the process does not finish in 5 s (helper, posix_spawn, SIGKILL).

#include <catch2/catch_test_macros.hpp>

#include "crispasr.h"
#include "crispasr_session.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <spawn.h>
#include <signal.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern char** environ;

struct crispasr_open_params_v1 {
    int abi_version;
    int n_threads;
    int use_gpu;
    int verbosity;
    int flash_attn;
    int n_gpu_layers;
    int reserved[6];
};

namespace {

const char* kAudio = "samples/jfk.wav";
const char* kVars[] = {"CRISPASR_SIMULATE_GPU_INIT_FALLBACK", "CRISPASR_SIMULATE_GPU_OPEN_FAILURE",
                       "CRISPASR_SIMULATE_GPU_COMPUTE_FAILURE", "CRISPASR_SIMULATE_HANG_OPEN",
                       "CRISPASR_SIMULATE_HANG_CALL"};

std::string model_path(const char* var) {
    const char* p = std::getenv(var);
    if (p && *p) {
        if (FILE* f = fopen(p, "rb")) {
            fclose(f);
            return p;
        }
    }
    const char* req = std::getenv("CA_REQUIRE_MODELS");
    if (req && std::string(req) == "1")
        FAIL(var << " not set or not readable (CA_REQUIRE_MODELS=1)");
    SKIP(var << " not set or not readable");
    return {};
}

// Sets one CRISPASR_SIMULATE_* (nullptr — none) and clears the rest for the scope.
struct SimulateEnv {
    explicit SimulateEnv(const char* var) {
        for (const char* v : kVars)
            unsetenv(v);
        if (var)
            setenv(var, "1", 1);
    }
    ~SimulateEnv() {
        for (const char* v : kVars)
            unsetenv(v);
    }
};

crispasr_session* open_session(const std::string& model, int use_gpu) {
    crispasr_open_params_v1 p = {2, 4, use_gpu, 0, 1, -1, {0}};
    return crispasr_session_open_with_params(model.c_str(), nullptr, &p);
}

struct Pcm {
    float* data = nullptr;
    int n = 0;
    Pcm() {
        int sr = 0;
        REQUIRE(crispasr_audio_load(kAudio, &data, &n, &sr) == 0);
    }
    ~Pcm() { crispasr_audio_free(data); }
};

// Joined segment text; "" for NULL or a result without segments.
std::string transcribe(crispasr_session* s, const Pcm& pcm) {
    crispasr_session_result* r = crispasr_session_transcribe(s, pcm.data, pcm.n);
    std::string text;
    if (r) {
        for (int i = 0; i < crispasr_session_result_n_segments(r); i++)
            text += crispasr_session_result_segment_text(r, i);
        crispasr_session_result_free(r);
    }
    return text;
}

std::string transcribe_once(const std::string& model, int use_gpu, const Pcm& pcm) {
    crispasr_session* s = open_session(model, use_gpu);
    REQUIRE(s != nullptr);
    std::string text = transcribe(s, pcm);
    crispasr_session_close(s);
    return text;
}

void run_injection_cases(const char* model_var) {
    const std::string model = model_path(model_var);
    Pcm pcm;

    // Reference texts without variables, once per model (Catch2 reruns the case
    // for every SECTION; the Debug CPU pass is slow).
    static std::map<std::string, std::pair<std::string, std::string>> reference;
    if (!reference.count(model)) {
        SimulateEnv env(nullptr);
        reference[model] = {transcribe_once(model, 1, pcm), transcribe_once(model, 0, pcm)};
    }
    const std::string& metal = reference[model].first;
    const std::string& cpu = reference[model].second;
    REQUIRE_FALSE(metal.empty());
    REQUIRE_FALSE(cpu.empty());

    SECTION("GPU_OPEN_FAILURE: GPU open returns NULL, CPU open is normal") {
        SimulateEnv env("CRISPASR_SIMULATE_GPU_OPEN_FAILURE");
        CHECK(open_session(model, 1) == nullptr);
        CHECK(transcribe_once(model, 0, pcm) == cpu);
    }
    SECTION("GPU_COMPUTE_FAILURE: the first GPU call computes nothing, the next is normal") {
        SimulateEnv env("CRISPASR_SIMULATE_GPU_COMPUTE_FAILURE");
        crispasr_session* s = open_session(model, 1);
        REQUIRE(s != nullptr);
        CHECK(transcribe(s, pcm).empty());
        CHECK(transcribe(s, pcm) == metal);
        crispasr_session_close(s);
        CHECK(transcribe_once(model, 0, pcm) == cpu);
    }
    SECTION("device without variables: use_gpu=1 is Metal, use_gpu=0 is CPU, NULL is -1") {
        SimulateEnv env(nullptr);
        for (int use_gpu : {1, 0}) {
            crispasr_session* s = open_session(model, use_gpu);
            REQUIRE(s != nullptr);
            CHECK(crispasr_session_device(s) == use_gpu);
            crispasr_session_close(s);
        }
        CHECK(crispasr_session_device(nullptr) == -1);
    }
    SECTION("GPU_INIT_FALLBACK: the GPU open lands on CPU, device 0") {
        SimulateEnv env("CRISPASR_SIMULATE_GPU_INIT_FALLBACK");
        crispasr_session* s = open_session(model, 1);
        REQUIRE(s != nullptr);
        CHECK(crispasr_session_device(s) == 0);
        CHECK(transcribe(s, pcm) == cpu);
        crispasr_session_close(s);
        CHECK(transcribe_once(model, 0, pcm) == cpu);
    }
}

#ifndef CA_SESSION_CONTRACT_HELPER
#error "CA_SESSION_CONTRACT_HELPER (path of session-contract-helper) is not defined"
#endif

enum class Outcome { exited_ok, exited_fail, hung };

// Runs the helper with `var`=1 (nullptr — none); "hung" if it has not exited
// `timeout` after start, or — with `after_open` — after it wrote the marker.
Outcome run_helper(const std::string& model, int use_gpu, const char* var, bool after_open, int timeout_s,
                   bool* opened) {
    char marker[] = "/tmp/ca-session-contract-XXXXXX";
    const int fd = mkstemp(marker);
    REQUIRE(fd >= 0);
    close(fd);
    unlink(marker);

    std::vector<std::string> env_s;
    for (char** e = environ; *e; e++) {
        bool simulate = false;
        for (const char* v : kVars)
            simulate |= std::strncmp(*e, v, std::strlen(v)) == 0 && (*e)[std::strlen(v)] == '=';
        if (!simulate)
            env_s.push_back(*e);
    }
    if (var)
        env_s.push_back(std::string(var) + "=1");
    std::vector<char*> envp;
    for (auto& e : env_s)
        envp.push_back(e.data());
    envp.push_back(nullptr);

    std::string gpu = std::to_string(use_gpu);
    std::vector<char*> argv = {const_cast<char*>(CA_SESSION_CONTRACT_HELPER), const_cast<char*>(model.c_str()),
                               gpu.data(), const_cast<char*>(kAudio), marker, nullptr};
    pid_t pid = 0;
    REQUIRE(posix_spawn(&pid, CA_SESSION_CONTRACT_HELPER, nullptr, nullptr, argv.data(), envp.data()) == 0);

    auto marker_exists = [&] { return access(marker, F_OK) == 0; };
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    bool deadline_from_open = false;
    int status = 0;
    for (;;) {
        if (waitpid(pid, &status, WNOHANG) == pid) {
            *opened = marker_exists();
            unlink(marker);
            return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? Outcome::exited_ok : Outcome::exited_fail;
        }
        if (after_open && !deadline_from_open && marker_exists()) {
            deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
            deadline_from_open = true;
        }
        // The open itself gets a minute to load the model before the hang clock starts.
        const auto limit = after_open && !deadline_from_open ? deadline + std::chrono::seconds(60) : deadline;
        if (std::chrono::steady_clock::now() > limit)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    *opened = marker_exists();
    unlink(marker);
    return Outcome::hung;
}

void run_hang_cases(const char* model_var) {
    const std::string model = model_path(model_var);
    bool opened = false;

    SECTION("no variable: the helper finishes") {
        CHECK(run_helper(model, 1, nullptr, false, 120, &opened) == Outcome::exited_ok);
        CHECK(opened);
    }
    SECTION("HANG_OPEN: the GPU open does not return in 5 s") {
        CHECK(run_helper(model, 1, "CRISPASR_SIMULATE_HANG_OPEN", false, 5, &opened) == Outcome::hung);
        CHECK_FALSE(opened);
    }
    SECTION("HANG_CALL: the first GPU call does not return in 5 s") {
        CHECK(run_helper(model, 1, "CRISPASR_SIMULATE_HANG_CALL", true, 5, &opened) == Outcome::hung);
        CHECK(opened);
    }
    SECTION("HANG_OPEN and HANG_CALL with use_gpu=0: the helper finishes") {
        CHECK(run_helper(model, 0, "CRISPASR_SIMULATE_HANG_OPEN", false, 120, &opened) == Outcome::exited_ok);
        CHECK(run_helper(model, 0, "CRISPASR_SIMULATE_HANG_CALL", false, 120, &opened) == Outcome::exited_ok);
    }
}

} // namespace

TEST_CASE("CRISPASR_SIMULATE_* on Parakeet v3", "[session-contract][parakeet]") {
    run_injection_cases("CRISPASR_MODEL_PARAKEET");
}

TEST_CASE("CRISPASR_SIMULATE_* on GigaAM e2e_rnnt", "[session-contract][gigaam]") {
    run_injection_cases("CRISPASR_MODEL_GIGAAM");
}

TEST_CASE("CRISPASR_SIMULATE_HANG_* on Parakeet v3", "[session-contract][parakeet]") {
    run_hang_cases("CRISPASR_MODEL_PARAKEET");
}

TEST_CASE("CRISPASR_SIMULATE_HANG_* on GigaAM e2e_rnnt", "[session-contract][gigaam]") {
    run_hang_cases("CRISPASR_MODEL_GIGAAM");
}
