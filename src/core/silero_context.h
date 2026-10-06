#pragma once
#include "crispasr.h"

#ifdef __cplusplus
extern "C" {
#endif
// Internal recipe switch: Silero's 16 kHz TorchScript wrapper prepends the
// previous 64 samples (zeros at file start), then reflects only the right edge.
// Enable on a fresh VAD context; other backends keep their existing recipe.
CRISPASR_API bool crispasr_silero_enable_context(struct whisper_vad_context* ctx);
// Like whisper_vad_detect_speech, but keeps the LSTM state and the source
// context from the previous call (a stateful per-chunk run). Read the result
// with whisper_vad_probs / whisper_vad_n_probs. A fresh context, or one just
// run through whisper_vad_detect_speech, starts from the reset state.
CRISPASR_API bool crispasr_silero_detect_continue(struct whisper_vad_context* ctx, const float* samples, int n_samples);
#ifdef __cplusplus
}
#endif
