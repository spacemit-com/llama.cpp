// SPDX-License-Identifier: MIT
#include "llama-spacemit-expert-cache.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace {
constexpr size_t alignment = 64;
constexpr size_t no_slot = std::numeric_limits<size_t>::max();
size_t add_size(size_t a, size_t b) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        throw std::overflow_error("expert cache size overflow");
    }
    return a + b;
}
size_t align_size(size_t n) {
    return add_size(n, alignment - 1) & ~(alignment - 1);
}
}

size_t llama_spacemit_expert_cache::count_slots(size_t total, double ratio) {
    if (!std::isfinite(ratio) || ratio < 0.0 || ratio > 1.0) {
        throw std::invalid_argument("expert slot ratio must be finite and in [0, 1]");
    }
    if (ratio == 1.0) { return total; }
    const double count = std::floor(ratio * double(total));
    return std::min(total, static_cast<size_t>(count));
}

llama_spacemit_expert_cache::llama_spacemit_expert_cache(const banks & weights, double ratio) {
    plans.resize(weights.size());
    slot_of.resize(weights.size());
    std::unordered_set<ggml_backend_buffer_t> seen_buffers;
    for (size_t il = 0; il < weights.size(); ++il) {
        size_t count = 0, offset = 0;
        for (const auto * tensor : weights[il]) {
            if (!tensor || !tensor->data || !tensor->buffer || !ggml_backend_buffer_is_host(tensor->buffer) ||
                    !ggml_is_contiguous(tensor) || tensor->ne[2] <= 0 || tensor->ne[3] != 1 ||
                    (count && count != size_t(tensor->ne[2]))) {
                throw std::runtime_error("expert slots require contiguous host-mapped expert banks with matching expert counts");
            }
            auto * device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const size_t bytes = ggml_nbytes(tensor);
            const size_t capacity = ggml_backend_buffer_get_size(tensor->buffer);
            const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(tensor->buffer));
            const auto address = reinterpret_cast<uintptr_t>(tensor->data);
            if ((device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) || !base ||
                    !bytes || bytes > capacity || address < base || address - base > capacity - bytes) {
                throw std::runtime_error("expert slots require valid CPU expert bank spans");
            }
            count = size_t(tensor->ne[2]);
            offset = align_size(offset);
            if (!lookup.emplace(tensor, bank{il, offset, tensor->nb[2]}).second) {
                throw std::runtime_error("expert slots do not support aliased expert banks across layers");
            }
            offset = add_size(offset, tensor->nb[2]);
            if (seen_buffers.insert(tensor->buffer).second) { source_buffers.push_back(tensor->buffer); }
        }
        slot_of[il].assign(count, no_slot);
        total = add_size(total, count);
        stride = std::max(stride, align_size(offset));
    }
    n_slots = count_slots(total, ratio);
    if (stride && n_slots > std::numeric_limits<size_t>::max() / stride) {
        throw std::overflow_error("expert cache allocation overflow");
    }
    if (n_slots) {
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) { throw std::runtime_error("expert slots require the CPU backend"); }
        buffer.reset(ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(cpu), buffer_bytes()));
        if (!buffer || !ggml_backend_buffer_get_base(buffer.get())) { throw std::bad_alloc(); }
        data = static_cast<unsigned char *>(ggml_backend_buffer_get_base(buffer.get()));
        published = std::make_unique<std::atomic<bool>[]>(n_slots);
        for (size_t i = 0; i < n_slots; ++i) { published[i].store(false, std::memory_order_relaxed); }
    }
    size_t assigned = 0;
    for (size_t expert = 0; assigned < n_slots; ++expert) {
        for (size_t il = 0; il < weights.size() && assigned < n_slots; ++il) {
            if (expert >= slot_of[il].size()) { continue; }
            const size_t slot = assigned++;
            slot_of[il][expert] = slot;
            for (const auto * tensor : weights[il]) {
                const auto & b = lookup.at(tensor);
                plans[il].push_back({static_cast<const unsigned char *>(tensor->data) + expert * tensor->nb[2],
                        data + slot * stride + b.offset, b.bytes, slot});
            }
        }
    }
    // Publish callbacks only after every fallible allocation. Model lifetime
    // guarantees all source buffers and this resolver outlive graph computation.
    for (auto * source : source_buffers) {
        ggml_backend_buffer_set_expert_resolver(source, &resolve, this);
    }
    LLAMA_LOG_INFO("spacemit expert slots: ratio=%.6f, total=%zu, slots=%zu, slot_bytes=%zu, buffer_bytes=%zu (%.2f MiB); fixed admission, mmap fallback\n",
            ratio, total, n_slots, stride, buffer_bytes(), buffer_bytes() / 1048576.0);
}

llama_spacemit_expert_cache::~llama_spacemit_expert_cache() {
    for (auto * source : source_buffers) { ggml_backend_buffer_set_expert_resolver(source, nullptr, nullptr); }
}

const void * llama_spacemit_expert_cache::resolve(const ggml_tensor * tensor, int64_t expert, void * user_data) {
    auto & self = *static_cast<llama_spacemit_expert_cache *>(user_data);
    const auto it = self.lookup.find(tensor);
    if (it == self.lookup.end()) { return nullptr; }
    const auto & b = it->second;
    const size_t slot = self.slot_of[b.layer][size_t(expert)];
    if (slot == no_slot || !self.ready(slot)) { return nullptr; }
    return self.data + slot * self.stride + b.offset;
}

std::unique_ptr<llama_spacemit_expert_cache> llama_spacemit_expert_cache_create(
        const llama_model & model, double ratio) {
    llama_spacemit_expert_cache::banks banks(model.layers.size());
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & layer = model.layers[il];
        const ggml_tensor * tensors[] = {
            layer.ffn_gate_up_exps ? nullptr : layer.ffn_gate_exps,
            layer.ffn_gate_up_exps ? nullptr : layer.ffn_up_exps,
            layer.ffn_down_exps, layer.ffn_gate_up_exps,
            layer.ffn_gate_chexps, layer.ffn_up_chexps, layer.ffn_down_chexps,
        };
        for (const auto * tensor : tensors) {
            if (!tensor) { continue; }
            if (!model.can_prefetch.count(tensor)) {
                throw std::runtime_error("expert slot ratio requires all routed experts to use lazy CPU mmap; use --cpu-moe");
            }
            banks[il].push_back(tensor);
        }
    }
    return std::make_unique<llama_spacemit_expert_cache>(banks, ratio);
}
