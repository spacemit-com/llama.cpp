#pragma once

#include "ggml.h"
#include "clip-model.h"

#include <cstdint>
#include <vector>
#include <string>

#define MTMD_INTERNAL_HEADER

struct mtmd_audio_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct mtmd_audio_mel_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

// cache for audio processing, each processor instance owns its own cache
struct mtmd_audio_cache {
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;

    std::vector<float> hann_window;

    mtmd_audio_mel_filters filters;

    void fill_sin_cos_table(uint32_t n);

    void fill_hann_window(uint32_t length, bool periodic);

    // Build mel filterbank matrix [n_mel × n_fft_bins] at runtime.
    // n_fft_bins must be (N_fft / 2 + 1). Example: if N_fft=512 -> n_fft_bins=257.
    void fill_mel_filterbank_matrix(int   n_mel,
                                    int   n_fft,
                                    int   sample_rate,               // e.g. 16000
                                    float fmin             = 0.0f,   // e.g. 0.0
                                    float fmax             = -1.0f,  // e.g. sr/2; pass -1 for auto
                                    bool  slaney_area_norm = true,
                                    float scale            = 1.0f,
                                    bool  use_htk          = false
    );
};

struct mtmd_audio_preprocessor {
    const clip_hparams & hparams;

    mtmd_audio_preprocessor(const clip_ctx * ctx): hparams(*clip_get_hparams(ctx)) {}

    virtual ~mtmd_audio_preprocessor() = default;
    virtual void initialize() = 0; // NOT thread-safe
    virtual bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) = 0;
};

struct mtmd_audio_preprocessor_whisper : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_whisper(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_conformer : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_conformer(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_granite_speech : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_granite_speech(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_gemma4a : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_gemma4a(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    mtmd_audio_cache cache;
};

// Generic helper to compute a log-mel spectrogram with the same implementation used by mtmd.
// The output layout is [n_mel, n_len] flattened in row-major order per mel bin.
bool mtmd_audio_compute_log_mel_spectrogram(const float * samples,
                                            size_t        n_samples,
                                            int           n_threads,
                                            int           n_mel,
                                            int           n_fft,
                                            int           window_len,
                                            int           hop_len,
                                            int           sample_rate,
                                            bool          center_padding,
                                            float         preemph,
                                            bool          use_natural_log,
                                            bool          norm_per_feature,
                                            mtmd_audio_mel & out);

// Gemma4 audio frontend features. Output layout: [n_frames, n_mel].
bool mtmd_audio_compute_gemma4_features(const float * samples,
                                        size_t        n_samples,
                                        int           sample_rate,
                                        int           n_mel,
                                        int           n_fft,
                                        int           window_len,
                                        int           hop_len,
                                        std::vector<float> & features,
                                        int &         n_frames_out);
struct mtmd_audio_preprocessor_qwen3a : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_qwen3a(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    mtmd_audio_cache cache;
};

// Kaldi-compatible fbank features (used by FunASR/SenseVoice models).
// Output layout: [n_frames, n_mel] flattened in row-major order per time frame.
bool mtmd_audio_compute_kaldi_fbank(const float * samples,
                                    size_t        n_samples,
                                    int           sample_rate,
                                    int           n_mel,
                                    int           frame_len,
                                    int           frame_shift,
                                    float         preemph_coeff,
                                    std::vector<float> & features,
                                    int &         n_frames_out);

// Low Frame Rate (LFR) frame stacking for FunASR models.
// Stacks lfr_m consecutive frames with stride lfr_n, using centered window
// and boundary clamping.  Output layout: [n_lfr_frames, n_mel * lfr_m].
bool mtmd_audio_compute_lfr(const std::vector<float> & features,
                            int n_frames,
                            int n_mel,
                            int lfr_m,
                            int lfr_n,
                            std::vector<float> & lfr_features,
                            int & n_lfr_frames_out);

//
// streaming ISTFT - converts spectrogram frames back to audio one frame at a time
//
struct mtmd_audio_streaming_istft {
    mtmd_audio_streaming_istft(int n_fft, int hop_length);

    // reset streaming state
    void reset();

    // process a single STFT frame (streaming)
    // frame_spectrum: [n_fft_bins x 2] interleaved real/imag
    // returns: up to hop_length samples
    std::vector<float> process_frame(const float * frame_spectrum);

    // flush remaining samples at end of stream
    std::vector<float> flush();

  private:
    int n_fft;
    int hop_length;
    int n_fft_bins;

    // Own cache for output processing
    mtmd_audio_cache cache;

    // Streaming state
    std::vector<float> overlap_buffer;
    std::vector<float> window_sum_buffer;
    int                padding_to_remove;

    // Working buffers for IFFT
    std::vector<float> ifft_in;
    std::vector<float> ifft_out;
};
