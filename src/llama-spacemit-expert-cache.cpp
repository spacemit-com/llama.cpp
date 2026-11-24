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

llama_spacemit_expert_cache::llama_spacemit_expert_cache(
        const banks & weights, double ratio) {
    tensors = weights;
    reading.assign(weights.size(), false);
    visible.resize(weights.size());
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
            const bank b{il, offset, tensor->nb[2]};
            if (!lookup.emplace(tensor, b).second) {
                throw std::runtime_error("expert slots do not support aliased expert banks across layers");
            }
            offset = add_size(b.offset, b.bytes);
            if (seen_buffers.insert(tensor->buffer).second) { source_buffers.push_back(tensor->buffer); }
        }
        slot_of[il].assign(count, no_slot);
        visible[il].assign(count, 0);
        prefill_slots = std::max(prefill_slots, count);
        total = add_size(total, count);
        stride = std::max(stride, align_size(offset));
    }
    n_slots = count_slots(total, ratio);
    if (ratio > 0 && n_slots < prefill_slots) {
        throw std::invalid_argument("expert slot ratio reserves fewer than one layer of experts; increase ratio or use 0 for mmap only");
    }
    if (stride && n_slots > std::numeric_limits<size_t>::max() / stride) {
        throw std::overflow_error("expert cache allocation overflow");
    }
    if (n_slots) {
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) { throw std::runtime_error("expert slots require the CPU backend"); }
        buffer.reset(ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(cpu), add_size(buffer_bytes(), alignment - 1)));
        if (!buffer || !ggml_backend_buffer_get_base(buffer.get())) { throw std::bad_alloc(); }
        data = static_cast<unsigned char *>(ggml_backend_buffer_get_base(buffer.get()));
        data = reinterpret_cast<unsigned char *>((reinterpret_cast<uintptr_t>(data) + alignment - 1) & ~(uintptr_t(alignment) - 1));
        entries.resize(n_slots);
    }
    for (auto * source : source_buffers) {
        ggml_backend_buffer_set_expert_resolver(source, &resolve, this);
    }
    LLAMA_LOG_INFO("spacemit expert slots: ratio=%.6f, total=%zu, slots=%zu, prefill_region=%zu, slot_bytes=%zu, buffer_bytes=%zu\n",
            ratio, total, n_slots, n_slots ? prefill_slots : 0, stride, buffer_bytes());
}

llama_spacemit_expert_cache::~llama_spacemit_expert_cache() {
    for (auto * source : source_buffers) {
        ggml_backend_buffer_set_expert_resolver(source, nullptr, nullptr);
    }
}

bool llama_spacemit_expert_cache::contains(size_t layer, size_t expert) const {
    std::lock_guard<std::mutex> lock(mutex);
    const size_t slot = slot_of.at(layer).at(expert);
    return slot != no_slot && entries[slot].state == slot_state::ready;
}

llama_spacemit_expert_cache::reservation llama_spacemit_expert_cache::reserve(
        size_t layer, size_t expert, bool prefill) {
    std::lock_guard<std::mutex> lock(mutex);
    const size_t old = slot_of.at(layer).at(expert);
    if (!n_slots || (old != no_slot && entries[old].state == slot_state::writing)) { return {}; }
    size_t slot = no_slot;
    if (prefill) {
        // Always reuse [0, experts_per_layer). Nothing beyond this region is a
        // prefill destination, even if the ratio allocated a larger buffer.
        slot = expert;
        if (old == slot && entries[old].state == slot_state::ready) { return {}; }
        if (entries[slot].state == slot_state::writing || entries[slot].pinned) { return {}; }
    } else {
        if (old != no_slot && entries[old].state == slot_state::ready) { return {}; }
        for (size_t i = 0; i < n_slots; ++i) {
            const auto & e = entries[i];
            if (e.state == slot_state::writing || e.pinned || e.protected_hit) { continue; }
            if (slot == no_slot || e.state == slot_state::empty || e.used < entries[slot].used) { slot = i; }
            if (e.state == slot_state::empty) { break; }
        }
        if (slot == no_slot) { return {}; }
    }
    if (old != no_slot && old != slot) {
        if (entries[old].pinned) { return {}; }
        entries[old].state = slot_state::empty;
        entries[old].protected_hit = false;
    }
    auto & e = entries[slot];
    if (e.state != slot_state::empty) { slot_of[e.layer][e.expert] = no_slot; }
    e.layer = layer;
    e.expert = expert;
    e.state = slot_state::writing;
    e.protected_hit = false;
    e.pinned = false;
    ++e.generation;
    slot_of[layer][expert] = slot;
    return {slot, e.generation};
}

std::vector<llama_spacemit_expert_cache::range> llama_spacemit_expert_cache::ranges(reservation r) const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<range> result;
    if (!valid(r)) { return result; }
    const auto & e = entries[r.slot];
    for (const auto * tensor : tensors[e.layer]) {
        const auto & b = lookup.at(tensor);
        result.push_back({static_cast<const unsigned char *>(tensor->data) + e.expert * tensor->nb[2],
                data + r.slot * stride + b.offset, b.bytes, r.slot});
    }
    return result;
}

bool llama_spacemit_expert_cache::valid(reservation r) const {
    return r.slot < entries.size() && entries[r.slot].state == slot_state::writing &&
            entries[r.slot].generation == r.generation;
}

void llama_spacemit_expert_cache::publish(reservation r) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!valid(r)) { return; }
    auto & e = entries[r.slot];
    e.state = slot_state::ready;
    e.used = ++clock;
    e.pinned = reading[e.layer] && visible[e.layer][e.expert];
}

void llama_spacemit_expert_cache::discard(reservation r) {
    // The transfer owner must acknowledge ALL writes before releasing a ticket.
    std::lock_guard<std::mutex> lock(mutex);
    if (!valid(r)) { return; }
    auto & e = entries[r.slot];
    slot_of[e.layer][e.expert] = no_slot;
    e.state = slot_state::empty;
}

void llama_spacemit_expert_cache::protect(size_t layer, const std::vector<uint8_t> & experts) {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto & e : entries) {
        e.protected_hit = e.state == slot_state::ready && e.layer == layer && experts.at(e.expert);
    }
}

void llama_spacemit_expert_cache::begin_experts(size_t layer, const std::vector<uint8_t> & experts) {
    std::lock_guard<std::mutex> lock(mutex);
    visible.at(layer) = experts;
    reading.at(layer) = true;
    for (auto & e : entries) {
        e.protected_hit = false;
        if (e.layer == layer && e.state == slot_state::ready && experts.at(e.expert)) {
            e.pinned = true;
            e.used = ++clock;
        }
    }
}

void llama_spacemit_expert_cache::end_experts(size_t layer) {
    std::lock_guard<std::mutex> lock(mutex);
    reading.at(layer) = false;
    for (auto & e : entries) { if (e.layer == layer) { e.pinned = false; } }
}

const void * llama_spacemit_expert_cache::resolve(const ggml_tensor * tensor, int64_t expert, void * user_data) {
    auto & self = *static_cast<llama_spacemit_expert_cache *>(user_data);
    const auto it = self.lookup.find(tensor);
    if (it == self.lookup.end()) { return nullptr; }
    const auto & b = it->second;
    std::lock_guard<std::mutex> lock(self.mutex);
    const size_t slot = self.slot_of[b.layer][size_t(expert)];
    if (slot == no_slot || self.entries[slot].state != slot_state::ready || !self.entries[slot].pinned) { return nullptr; }
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
    auto cache = std::make_unique<llama_spacemit_expert_cache>(banks, ratio);
    if (cache->capacity()) { LLAMA_LOG_INFO("spacemit expert slots: asynchronous mmap-to-slot copy enabled\n"); }
    return cache;
}
