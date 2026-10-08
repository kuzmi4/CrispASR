// src/core/rnnt_greedy.h — greedy RNNT decode loop with the model behind callbacks.
//
// The loop is the test seam of the GigaAM greedy RNNT decode: the backend passes
// its joint and predictor steps, a test passes synthetic ones. Per frame `t` the
// loop emits up to `max_symbols` tokens; `blank` ends the frame.
//
//   joint(t, logits)  — fills `logits` (vocab size) for frame `t` at the current
//                       predictor state;
//   on_emit(tok)      — advances the predictor after a non-blank emission.
//
// Header-only, no ggml dependency.

#pragma once

#include <cmath>
#include <vector>

namespace core_rnnt {

struct emission {
    int id;
    int t;
    float p; // softmax probability of the emitted token (confidence fields only)
};

template <class Joint, class OnEmit>
std::vector<emission> rnnt_greedy_loop(int T, int max_symbols, int blank_id, Joint&& joint, OnEmit&& on_emit) {
    std::vector<emission> emitted;
    std::vector<float> logits;
    for (int t = 0; t < T; t++) {
        for (int sym = 0; sym < max_symbols; sym++) {
            joint(t, logits);
            const int C = (int)logits.size();

            int tok = 0;
            float best = logits[0];
            for (int v = 1; v < C; v++) {
                if (logits[(size_t)v] > best) {
                    best = logits[(size_t)v];
                    tok = v;
                }
            }
            if (tok == blank_id)
                break;

            float sum = 0.0f;
            for (int v = 0; v < C; v++)
                sum += expf(logits[(size_t)v] - best);

            emitted.push_back({tok, t, 1.0f / sum});
            on_emit(tok);
        }
    }
    return emitted;
}

} // namespace core_rnnt
