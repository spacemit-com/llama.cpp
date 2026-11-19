// SPDX-License-Identifier: MIT
#pragma once

#include "ggml-cpp.h"
#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>

struct llama_model;

// Fixed-capacity host slots. One slot holds ALL routed matrices of one layer's
// expert. Initial admission is round-robin across layers, lowest expert IDs first.
// Slots are immutable after release publication. Misses always read source mmap;
// no eviction/replacement or claim about OS file-cache residency is made here.
class llama_spacemit_expert_cache {
public:
    struct range {
        const unsigned char * source;
        unsigned char * destination;
        size_t bytes;
        size_t slot;
    };
    using banks = std::vector<std::vector<const ggml_tensor *>>;
    llama_spacemit_expert_cache(const banks & weights, double ratio);
    ~llama_spacemit_expert_cache();
    llama_spacemit_expert_cache(const llama_spacemit_expert_cache &) = delete;
    llama_spacemit_expert_cache & operator=(const llama_spacemit_expert_cache &) = delete;

    static size_t count_slots(size_t total, double ratio);
    size_t total_experts() const { return total; }
    size_t capacity() const { return n_slots; }
    size_t slot_bytes() const { return stride; }
    size_t buffer_bytes() const { return n_slots * stride; }
    const std::vector<range> & ranges(size_t layer) const { return plans.at(layer); }
    bool ready(size_t slot) const { return published[slot].load(std::memory_order_acquire); }
    void publish(size_t slot) { published[slot].store(true, std::memory_order_release); }

private:
    struct bank {
        size_t layer;
        size_t offset;
        size_t bytes;
    };
    static const void * resolve(const ggml_tensor * tensor, int64_t expert, void * user_data);
    size_t total = 0, n_slots = 0, stride = 0;
    ggml_backend_buffer_ptr buffer;
    unsigned char * data = nullptr;
    std::unique_ptr<std::atomic<bool>[]> published;
    std::unordered_map<const ggml_tensor *, bank> lookup;
    std::vector<std::vector<size_t>> slot_of;
    std::vector<std::vector<range>> plans;
    std::vector<ggml_backend_buffer_t> source_buffers;
};

std::unique_ptr<llama_spacemit_expert_cache> llama_spacemit_expert_cache_create(
        const llama_model & model, double ratio);
