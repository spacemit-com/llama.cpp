#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <cerrno>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef Q3TTS_ENABLE_SDK_AUDIO
#include <alsa/asoundlib.h>
#endif

namespace q3tts_audio {

#ifdef Q3TTS_ENABLE_SDK_AUDIO

using Clock = std::chrono::steady_clock;

inline std::vector<int16_t> resample_pcm16_linear(const std::vector<int16_t> &input,
                                                  int input_rate,
                                                  int output_rate,
                                                  int channels) {
    if (input.empty() || input_rate == output_rate) {
        return input;
    }
    if (input_rate <= 0 || output_rate <= 0 || channels <= 0) {
        throw std::runtime_error("invalid audio resample config");
    }

    const size_t in_frames = input.size() / static_cast<size_t>(channels);
    if (in_frames == 0) {
        return {};
    }
    const size_t out_frames = std::max<size_t>(
        1, static_cast<size_t>(
               (static_cast<uint64_t>(in_frames) * static_cast<uint64_t>(output_rate) +
                static_cast<uint64_t>(input_rate) / 2) /
               static_cast<uint64_t>(input_rate)));
    std::vector<int16_t> output(out_frames * static_cast<size_t>(channels));
    const double scale = static_cast<double>(input_rate) / static_cast<double>(output_rate);
    for (size_t out_frame = 0; out_frame < out_frames; ++out_frame) {
        const double src = static_cast<double>(out_frame) * scale;
        const size_t i0 = std::min(static_cast<size_t>(src), in_frames - 1);
        const size_t i1 = std::min(i0 + 1, in_frames - 1);
        const double frac = src - static_cast<double>(i0);
        for (int ch = 0; ch < channels; ++ch) {
            const int16_t a = input[i0 * static_cast<size_t>(channels) + static_cast<size_t>(ch)];
            const int16_t b = input[i1 * static_cast<size_t>(channels) + static_cast<size_t>(ch)];
            const double v = static_cast<double>(a) + (static_cast<double>(b) - static_cast<double>(a)) * frac;
            output[out_frame * static_cast<size_t>(channels) + static_cast<size_t>(ch)] =
                static_cast<int16_t>(std::max(-32768.0, std::min(32767.0, v)));
        }
    }
    return output;
}

inline std::vector<int16_t> convert_channels_pcm16(const std::vector<int16_t> &input,
                                                   int input_channels,
                                                   int output_channels) {
    if (input.empty() || input_channels == output_channels) {
        return input;
    }

    const size_t frames = input.size() / static_cast<size_t>(input_channels);
    std::vector<int16_t> output(frames * static_cast<size_t>(output_channels));
    for (size_t frame = 0; frame < frames; ++frame) {
        const size_t in_base = frame * static_cast<size_t>(input_channels);
        const size_t out_base = frame * static_cast<size_t>(output_channels);
        if (output_channels == 1) {
            int total = 0;
            for (int ch = 0; ch < input_channels; ++ch) {
                total += input[in_base + static_cast<size_t>(ch)];
            }
            output[out_base] = static_cast<int16_t>(total / input_channels);
        } else if (input_channels == 1) {
            for (int ch = 0; ch < output_channels; ++ch) {
                output[out_base + static_cast<size_t>(ch)] = input[in_base];
            }
        } else {
            for (int ch = 0; ch < output_channels; ++ch) {
                const int in_ch = ch < input_channels ? ch : input_channels - 1;
                output[out_base + static_cast<size_t>(ch)] =
                    input[in_base + static_cast<size_t>(in_ch)];
            }
        }
    }
    return output;
}

class SdkSegmentPlayer {
public:
    SdkSegmentPlayer(int sample_rate, int channels, int device, int frames_per_buffer,
                     int tail_ms, int drain_ms, int segment_pause_ms)
        : sample_rate_(sample_rate), channels_(channels),
          tail_ms_(std::max(0, tail_ms)), drain_ms_(std::max(0, drain_ms)),
          segment_pause_ms_(std::max(0, segment_pause_ms)) {
        if (sample_rate_ <= 0 || channels_ <= 0) {
            throw std::runtime_error("invalid SDK audio playback config");
        }
        frames_per_buffer_ = frames_per_buffer > 0 ? frames_per_buffer : 1024;
        const char *env_device = std::getenv("Q3TTS_ALSA_DEVICE");
        device_name_ = env_device && *env_device
            ? env_device
            : (device >= 0 ? "plughw:" + std::to_string(device) + ",0" : "default");

        int err = snd_pcm_open(&pcm_, device_name_.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
        if (err < 0) {
            throw std::runtime_error("ALSA open failed for " + device_name_ + ": " + snd_strerror(err));
        }
        err = snd_pcm_set_params(
            pcm_, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED,
            static_cast<unsigned int>(channels_), static_cast<unsigned int>(sample_rate_),
            1, 200000);
        if (err < 0) {
            snd_pcm_close(pcm_);
            pcm_ = nullptr;
            throw std::runtime_error("ALSA set params failed for " + device_name_ + ": " + snd_strerror(err));
        }
        std::cerr << "[Q3TTSAudio] Opened ALSA " << device_name_
                  << " " << sample_rate_ << "Hz ch " << channels_
                  << " buffer " << frames_per_buffer_ << "\n";
        expected_end_ = Clock::now();
        worker_ = std::thread([this]() { run(); });
    }

    ~SdkSegmentPlayer() {
        finish_no_throw();
    }

    SdkSegmentPlayer(const SdkSegmentPlayer &) = delete;
    SdkSegmentPlayer &operator=(const SdkSegmentPlayer &) = delete;

    double enqueue_mono24k(std::vector<int16_t> samples) {
        if (enqueued_ > 0 && segment_pause_ms_ > 0) {
            const size_t pause_frames =
                static_cast<size_t>(24000) * static_cast<size_t>(segment_pause_ms_) / 1000;
            samples.insert(samples.end(), pause_frames, 0);
        }
        samples = resample_pcm16_linear(samples, 24000, sample_rate_, 1);
        samples = convert_channels_pcm16(samples, 1, channels_);
        const double audio_s =
            static_cast<double>(samples.size()) / static_cast<double>(sample_rate_ * channels_);

        const auto now = Clock::now();
        double gap_s = 0.0;
        if (enqueued_ > 0 && now > expected_end_) {
            gap_s = std::chrono::duration<double>(now - expected_end_).count();
        }
        const auto start = now > expected_end_ ? now : expected_end_;
        expected_end_ = start +
            std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(audio_s));
        ++enqueued_;

        {
            std::lock_guard<std::mutex> lock(mu_);
            queue_.push_back(std::move(samples));
        }
        cv_.notify_one();
        return gap_s;
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (finished_) {
                return;
            }
            finished_ = true;
            if (enqueued_ > 0 && tail_ms_ > 0) {
                const size_t tail_frames =
                    static_cast<size_t>(sample_rate_) * static_cast<size_t>(tail_ms_) / 1000;
                if (tail_frames > 0) {
                    queue_.emplace_back(tail_frames * static_cast<size_t>(channels_), 0);
                    const auto now = Clock::now();
                    const auto start = now > expected_end_ ? now : expected_end_;
                    expected_end_ = start + std::chrono::duration_cast<Clock::duration>(
                        std::chrono::duration<double>(static_cast<double>(tail_ms_) / 1000.0));
                }
            }
            done_ = true;
        }
        cv_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
        if (enqueued_ > 0 && drain_ms_ > 0) {
            const auto drain_until = expected_end_ + std::chrono::milliseconds(drain_ms_);
            const auto now = Clock::now();
            if (now < drain_until) {
                std::this_thread::sleep_until(drain_until);
            }
        }
        if (pcm_) {
            snd_pcm_drain(pcm_);
            snd_pcm_close(pcm_);
            pcm_ = nullptr;
        }
        if (!ok_) {
            throw std::runtime_error("ALSA write failed");
        }
    }

private:
    void finish_no_throw() {
        try {
            finish();
        } catch (...) {
        }
    }

    void run() {
        while (true) {
            std::vector<int16_t> samples;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&]() { return done_ || !queue_.empty(); });
                if (queue_.empty()) {
                    if (done_) {
                        break;
                    }
                    continue;
                }
                samples = std::move(queue_.front());
                queue_.pop_front();
            }
            snd_pcm_sframes_t offset = 0;
            snd_pcm_sframes_t remaining =
                static_cast<snd_pcm_sframes_t>(samples.size() / static_cast<size_t>(channels_));
            while (remaining > 0) {
                const snd_pcm_sframes_t chunk =
                    std::min<snd_pcm_sframes_t>(remaining, static_cast<snd_pcm_sframes_t>(frames_per_buffer_));
                snd_pcm_sframes_t wrote = snd_pcm_writei(
                    pcm_,
                    samples.data() + static_cast<size_t>(offset) * static_cast<size_t>(channels_),
                    chunk);
                if (wrote == -EPIPE) {
                    snd_pcm_prepare(pcm_);
                    continue;
                }
                if (wrote < 0) {
                    std::cerr << "[Q3TTSAudio] Write failed: " << snd_strerror(static_cast<int>(wrote)) << "\n";
                    ok_ = false;
                    break;
                }
                offset += wrote;
                remaining -= wrote;
            }
            if (!ok_) {
                break;
            }
        }
    }

    int sample_rate_;
    int channels_;
    int tail_ms_;
    int drain_ms_;
    int segment_pause_ms_;
    int frames_per_buffer_ = 1024;
    std::string device_name_;
    snd_pcm_t *pcm_ = nullptr;
    std::thread worker_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::vector<int16_t>> queue_;
    bool done_ = false;
    bool finished_ = false;
    bool ok_ = true;
    size_t enqueued_ = 0;
    Clock::time_point expected_end_;
};

#endif

}  // namespace q3tts_audio
