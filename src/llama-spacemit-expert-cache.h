// SPDX-License-Identifier: MIT
#pragma once

#include "ggml-cpp.h"
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

struct llama_model;

// One model-owned buffer shared by both policies in llama-spacemit-prefetch.cpp.
// Prefill replaces only the first layer-sized region; decode uses every slot
// with LRU replacement. Readers pin complete experts at router READY and release
// at EXPERTS END. A writing reservation is never evicted, including cancellation:
// the transfer owner discards it only after terminal acknowledgement.
class llama_spacemit_expert_cache {
public:
    struct range {
        const unsigned char * source;
        unsigned char * destination;
        size_t bytes;
        size_t slot;
    };
    struct reservation {
        size_t slot = std::numeric_limits<size_t>::max();
        uint64_t generation = 0;
        explicit operator bool() const { return slot != std::numeric_limits<size_t>::max(); }
    };
    using banks = std::vector<std::vector<const ggml_tensor *>>;
    llama_spacemit_expert_cache(const banks & weights, double ratio);
    ~llama_spacemit_expert_cache();
    llama_spacemit_expert_cache(const llama_spacemit_expert_cache &) = delete;
    llama_spacemit_expert_cache & operator=(const llama_spacemit_expert_cache &) = delete;

    static size_t count_slots(size_t total, double ratio);
    size_t total_experts() const { return total; }
    size_t capacity() const { return n_slots; }
    size_t prefill_capacity() const { return n_slots ? prefill_slots : 0; }
    size_t slot_bytes() const { return stride; }
    size_t buffer_bytes() const { return n_slots * stride; }
    bool contains(size_t layer, size_t expert) const;
    reservation reserve(size_t layer, size_t expert, bool prefill);
    std::vector<range> ranges(reservation r) const;
    void publish(reservation r);
    void discard(reservation r);
    void protect(size_t layer, const std::vector<uint8_t> & experts);
    void begin_experts(size_t layer, const std::vector<uint8_t> & experts);
    void end_experts(size_t layer);

private:
    enum class slot_state { empty, writing, ready };
    struct entry {
        size_t layer = 0, expert = 0;
        uint64_t generation = 0, used = 0;
        slot_state state = slot_state::empty;
        bool protected_hit = false, pinned = false;
    };
    struct bank {
        size_t layer, offset, bytes;
    };
    bool valid(reservation r) const; // mutex held
    static const void * resolve(const ggml_tensor * tensor, int64_t expert, void * user_data);
    size_t total = 0, n_slots = 0, stride = 0, prefill_slots = 0;
    uint64_t clock = 0;
    ggml_backend_buffer_ptr buffer;
    unsigned char * data = nullptr;
    mutable std::mutex mutex;
    banks tensors;
    std::unordered_map<const ggml_tensor *, bank> lookup;
    std::vector<std::vector<size_t>> slot_of;
    std::vector<std::vector<uint8_t>> visible;
    std::vector<bool> reading;
    std::vector<entry> entries;
    std::vector<ggml_backend_buffer_t> source_buffers;
};

std::unique_ptr<llama_spacemit_expert_cache> llama_spacemit_expert_cache_create(
        const llama_model & model, double ratio);
