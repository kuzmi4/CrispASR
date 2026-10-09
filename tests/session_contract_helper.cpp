// Context-Assist (T3.1, F5): child process for the hang cases of
// test_session_contract_live.cpp. The parent sets CRISPASR_SIMULATE_* in the
// environment and kills this process when it does not exit in time.
//
//   session-contract-helper <model> <use_gpu 0|1> <wav> <opened-marker>
//
// Opens a session, writes <opened-marker> once open returned, transcribes <wav>.
// Exit 0 — a result came back; 1 — open or transcribe failed; 2 — bad args.

#include "crispasr.h"
#include "crispasr_session.h"

#include <cstdio>
#include <cstdlib>

struct crispasr_open_params_v1 {
    int abi_version;
    int n_threads;
    int use_gpu;
    int verbosity;
    int flash_attn;
    int n_gpu_layers;
    int reserved[6];
};

int main(int argc, char** argv) {
    if (argc != 5)
        return 2;
    crispasr_open_params_v1 p = {2, 4, std::atoi(argv[2]), 0, 1, -1, {0}};
    crispasr_session* s = crispasr_session_open_with_params(argv[1], nullptr, &p);
    if (!s)
        return 1;
    if (FILE* f = std::fopen(argv[4], "w"))
        std::fclose(f);

    float* pcm = nullptr;
    int n = 0, sr = 0;
    if (crispasr_audio_load(argv[3], &pcm, &n, &sr) != 0) // 16 kHz: Parakeet and GigaAM
        return 1;
    crispasr_session_result* r = crispasr_session_transcribe(s, pcm, n);
    const int rc = r ? 0 : 1;
    crispasr_session_result_free(r);
    crispasr_audio_free(pcm);
    crispasr_session_close(s);
    return rc;
}
