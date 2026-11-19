// SPDX-License-Identifier: MIT
#include "llama-spacemit-prefetch.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-spacemit-expert-cache.h"

#include "ggml-spacemit.h"
#include "ggml-spacemit-transfer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <utility>

namespace {
// There is no callback getter or ownership token. One conservative owner also
// covers duplicate handles and device wrappers sharing a session. Direct API
// users still have to coordinate; their existing callbacks cannot be queried.
std::atomic_flag spacemit_prefetch_owner = ATOMIC_FLAG_INIT;

template<typename T>
T spacemit_prefetch_proc(ggml_backend_reg_t reg, const char * name) {
    return reinterpret_cast<T>(ggml_backend_reg_get_proc_address(reg, name));
}

struct spacemit_prefetch_api {
    decltype(&ggml_backend_spacemit_set_prefetch_callback) set_callback = nullptr;
    decltype(&ggml_spacemit_transfer_create) create = nullptr;
    decltype(&ggml_spacemit_transfer_free) free = nullptr;
    decltype(&ggml_spacemit_transfer_event_create) event_create = nullptr;
    decltype(&ggml_spacemit_transfer_event_free) event_free = nullptr;
    decltype(&ggml_spacemit_transfer_prefetch) prefetch = nullptr;
    decltype(&ggml_spacemit_transfer_submit) submit = nullptr;
    decltype(&ggml_spacemit_transfer_cancel) cancel = nullptr;
    decltype(&ggml_spacemit_transfer_query) query = nullptr;
    decltype(&ggml_spacemit_transfer_drain) drain = nullptr;

    bool load(ggml_backend_reg_t reg) {
        set_callback = spacemit_prefetch_proc<decltype(set_callback)>(reg,
                "ggml_backend_spacemit_set_prefetch_callback");
        if (!set_callback) {
            return false;
        }
        create       = spacemit_prefetch_proc<decltype(create)>(reg,       "ggml_spacemit_transfer_create");
        free         = spacemit_prefetch_proc<decltype(free)>(reg,         "ggml_spacemit_transfer_free");
        event_create = spacemit_prefetch_proc<decltype(event_create)>(reg, "ggml_spacemit_transfer_event_create");
        event_free   = spacemit_prefetch_proc<decltype(event_free)>(reg,   "ggml_spacemit_transfer_event_free");
        submit       = spacemit_prefetch_proc<decltype(submit)>(reg, "ggml_spacemit_transfer_submit");
        prefetch     = spacemit_prefetch_proc<decltype(prefetch)>(reg,     "ggml_spacemit_transfer_prefetch");
        cancel       = spacemit_prefetch_proc<decltype(cancel)>(reg,       "ggml_spacemit_transfer_cancel");
        query        = spacemit_prefetch_proc<decltype(query)>(reg,        "ggml_spacemit_transfer_query");
        drain        = spacemit_prefetch_proc<decltype(drain)>(reg,        "ggml_spacemit_transfer_drain");
        return create && free && event_create && event_free && prefetch && submit && cancel && query && drain;
    }
};
} // namespace

struct llama_spacemit_prefetch::impl {
    static constexpr size_t chunk_bytes = size_t(16) * 1024 * 1024;
    static constexpr size_t inflight_limit = size_t(64) * 1024 * 1024;
    static constexpr size_t slot_count = 4;

    struct projection {
        const unsigned char * source = nullptr;
        size_t bytes = 0;
        size_t cursor = 0;
        unsigned char * destination = nullptr;
        size_t expert_slot = 0;
    };

    struct layer_plan {
        std::vector<projection> projections;
        size_t next_projection = 0;
    };

    struct event_slot {
        ggml_spacemit_transfer_event_t event = nullptr;
        size_t bytes = 0;
        bool pending = false;
        size_t expert_slot = 0;
    };

    llama_spacemit_expert_cache * expert_cache = nullptr; // model-owned
    std::vector<size_t> remaining_chunks; // dispatcher-only, per expert slot
    spacemit_prefetch_api api;
    ggml_backend_t backend = nullptr; // borrowed through stop()
    ggml_spacemit_transfer_t transfer = nullptr;
    std::vector<layer_plan> layers;
    // One permanently set activation bit per layer, allocated before binding.
    // The callback writes only these bits; cursors/events belong to dispatcher.
    std::unique_ptr<std::atomic<bool>[]> activated;
    size_t n_layers = 0;
    std::array<event_slot, slot_count> slots{};
    std::atomic<bool> stopping{false};
    std::thread dispatcher;
    std::mutex dispatch_mutex;
    std::condition_variable dispatch_wakeup;
    size_t next_layer = 0;
    size_t inflight = 0;
    bool owns_callback = false;
    bool callback_bound = false;
    bool stopped = false;
    bool transfer_failed = false; // dispatcher-only until join()

    ~impl() {
        stop();
    }

    static bool eligible(const llama_model & model, const ggml_tensor * tensor) {
        // The loader's allowlist distinguishes lazy host mappings from eager
        // allocations and other merely host-accessible memory.
        if (!tensor || !model.can_prefetch.count(tensor) || !tensor->data || !tensor->buffer ||
                !ggml_backend_buffer_is_host(tensor->buffer)) {
            return false;
        }
        auto * buft = ggml_backend_buffer_get_type(tensor->buffer);
        auto * device = buft ? ggml_backend_buft_get_device(buft) : nullptr;
        // Core CPU/CPU_Mapped buffer types intentionally have no device.
        // The loader allowlist plus host/span checks identify these mappings;
        // reject an explicitly attached non-CPU device, not a missing device.
        if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            return false;
        }
        const size_t bytes = ggml_nbytes(tensor);
        const size_t capacity = ggml_backend_buffer_get_size(tensor->buffer);
        const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(tensor->buffer));
        const auto address = reinterpret_cast<uintptr_t>(tensor->data);
        // Validate the borrowed span without reading or faulting mapped pages.
        return base != 0 && bytes != 0 && bytes <= capacity && address >= base &&
                address - base <= capacity - bytes;
    }

    size_t make_prefill_plan(const llama_model & model) {
        n_layers = model.layers.size();
        layers.resize(n_layers);
        if (n_layers) {
            activated = std::make_unique<std::atomic<bool>[]>(n_layers);
            for (size_t il = 0; il < n_layers; ++il) {
                activated[il].store(false, std::memory_order_relaxed);
            }
        }
        expert_cache = model.expert_cache.get();
        if (expert_cache) {
            remaining_chunks.assign(expert_cache->capacity(), 0);
            size_t ranges = 0;
            for (size_t il = 0; il < n_layers; ++il) {
                for (const auto & r : expert_cache->ranges(il)) {
                    if (expert_cache->ready(r.slot)) { continue; }
                    layers[il].projections.push_back({r.source, r.bytes, 0, r.destination, r.slot});
                    remaining_chunks[r.slot] += (r.bytes + chunk_bytes - 1) / chunk_bytes;
                    ++ranges;
                }
            }
            return ranges;
        }
        std::unordered_set<const ggml_tensor *> seen;
        size_t ranges = 0;
        for (size_t il = 0; il < n_layers; ++il) {
            const auto & layer = model.layers[il];
            // Routed matrices only. Exclude routers, shared experts, biases,
            // and scales; include ordinary, fused gate/up, and CHEXPS banks.
            const ggml_tensor * tensors[] = {
                layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps,
                layer.ffn_gate_up_exps,
                layer.ffn_gate_chexps, layer.ffn_up_chexps, layer.ffn_down_chexps,
            };
            for (const auto * tensor : tensors) {
                if (!eligible(model, tensor) || !seen.insert(tensor).second) {
                    continue;
                }
                layers[il].projections.push_back({static_cast<const unsigned char *>(tensor->data),
                        ggml_nbytes(tensor), 0});
                ++ranges;
            }
        }
        // No permanent quotas: every byte of each eligible range can be
        // admitted once its layer has been activated, while this context lives.
        return ranges;
    }

    void start_prefill(const llama_model & model, const std::vector<ggml_backend_ptr> & backends) {
        // Bind only one supported handle: additional handles may share exactly
        // the same callback/session and would overwrite the original binding.
        for (const auto & candidate : backends) {
            if (!candidate) {
                continue;
            }
            auto * device = ggml_backend_get_device(candidate.get());
            auto * reg = device ? ggml_backend_dev_backend_reg(device) : nullptr;
            if (!reg) {
                continue;
            }
            spacemit_prefetch_api candidate_api;
            if (candidate_api.load(reg)) {
                api = candidate_api;
                backend = candidate.get();
                break;
            }
            if (candidate_api.set_callback) {
                LLAMA_LOG_WARN("spacemit prefetch: optional transfer API incomplete; skipping backend\n");
            }
        }
        if (!backend) {
            return;
        }
        const size_t ranges = make_prefill_plan(model);
        if (!ranges) {
            LLAMA_LOG_DEBUG("spacemit prefetch: no eligible lazy CPU-mapped routed expert weights\n");
            return;
        }
        if (spacemit_prefetch_owner.test_and_set(std::memory_order_acquire)) {
            LLAMA_LOG_WARN("spacemit prefetch: another context owns the shared callback; warmup disabled\n");
            return;
        }
        owns_callback = true;

        // The service discovers online, affinity-eligible X100 CPUs, excludes
        // CPUs with >10% busy time in a 100ms sample, and binds the single
        // transfer worker before accepting work. No caller-side CPU ranges,
        // affinity changes, A100 fallback, or payload work on a compute thread.
        // Selection remains an availability snapshot, not a CPU reservation.
        const ggml_spacemit_transfer_config config{1, slot_count};
        ggml_spacemit_transfer_status status = GGML_SPACEMIT_TRANSFER_UNAVAILABLE;
        transfer = api.create(&config, &status);
        if (!transfer) {
            LLAMA_LOG_WARN("spacemit prefetch: no available X100 transfer worker (status %d); warmup disabled\n",
                    static_cast<int>(status));
            stop();
            return;
        }
        for (auto & slot : slots) {
            slot.event = api.event_create();
            if (!slot.event) {
                LLAMA_LOG_WARN("spacemit prefetch: event allocation failed; warmup disabled\n");
                stop();
                return;
            }
        }
        // All state is preallocated. This unpinned thread only dispatches
        // descriptors and observes events; payload touches use the X100 worker.
        dispatcher = std::thread(&impl::dispatch_prefill_entry, this);
        if (!api.set_callback(backend, &impl::on_prefill_event, this)) {
            LLAMA_LOG_WARN("spacemit prefetch: callback registration failed; warmup disabled\n");
            stop();
            return;
        }
        callback_bound = true;
        if (expert_cache) {
            LLAMA_LOG_INFO("spacemit prefetch: bounded expert copy enabled, %zu slots, %zu bytes, %zu ranges; one X100 worker, mmap fallback until complete\n",
                    expert_cache->capacity(), expert_cache->buffer_bytes(), ranges);
            return;
        }
        LLAMA_LOG_INFO("spacemit prefetch: source warmup enabled for %zu routed weight ranges, "
                "16 MiB chunks, 64 MiB in flight, one X100 transfer worker plus dispatcher; "
                "one sweep per activated layer, no lifetime byte quota or residency bound\n", ranges);
    }

    static void on_prefill_event(const ggml_spacemit_prefetch_event * event, void * user_data) noexcept {
        if (!event || event->module != GGML_SPACEMIT_MODULE_QKV ||
                event->phase != GGML_SPACEMIT_PREFETCH_BEGIN || event->layer < 0) {
            return;
        }
        auto & self = *static_cast<impl *>(user_data);
        const size_t il = static_cast<size_t>(event->layer);
        if (!self.stopping.load(std::memory_order_acquire) && il < self.n_layers) {
            // Constant, preallocated activation only. No allocation, mutex,
            // notification, event query, transfer submission, or payload read.
            // Repeated callbacks cannot reset the dispatcher's persistent cursor.
            self.activated[il].store(true, std::memory_order_release);
        }
    }

    void reap() noexcept {
        // Only the dispatcher observes/reuses events. Control-thread cancel and
        // destruction start after join(), so no earlier observer/canceller can
        // race reuse. The provider additionally try-locks terminal events.
        for (auto & slot : slots) {
            if (!slot.pending) {
                continue;
            }
            const auto state = api.query(slot.event);
            if (state == GGML_SPACEMIT_TRANSFER_COMPLETE || state == GGML_SPACEMIT_TRANSFER_FAILED ||
                    state == GGML_SPACEMIT_TRANSFER_CANCELLED_STATE) {
                if (expert_cache && state == GGML_SPACEMIT_TRANSFER_COMPLETE) {
                    GGML_ASSERT(remaining_chunks[slot.expert_slot] > 0);
                    if (--remaining_chunks[slot.expert_slot] == 0) {
                        expert_cache->publish(slot.expert_slot);
                    }
                }
                inflight -= slot.bytes;
                slot.bytes = 0;
                slot.pending = false;
                if (state != GGML_SPACEMIT_TRANSFER_COMPLETE) {
                    transfer_failed = true;
                }
            }
        }
    }

    bool choose_projection(size_t & il, size_t & ip) noexcept {
        // Round robin across activated layers, then within each layer's
        // projections. A large layer cannot consume every refill indefinitely.
        for (size_t probe = 0; probe < n_layers; ++probe) {
            il = (next_layer + probe) % n_layers;
            if (!activated[il].load(std::memory_order_acquire)) {
                continue;
            }
            auto & layer = layers[il];
            const size_t count = layer.projections.size();
            for (size_t pp = 0; pp < count; ++pp) {
                ip = (layer.next_projection + pp) % count;
                const auto & p = layer.projections[ip];
                if (p.cursor < p.bytes) {
                    return true;
                }
            }
        }
        return false;
    }

    void refill_prefill() noexcept {
        for (size_t attempt = 0; attempt < slot_count; ++attempt) {
            if (stopping.load(std::memory_order_acquire) || inflight == inflight_limit) {
                return;
            }
            event_slot * available = nullptr;
            for (auto & slot : slots) {
                if (!slot.pending) {
                    available = &slot;
                    break;
                }
            }
            if (!available) {
                return;
            }
            size_t il = 0;
            size_t ip = 0;
            if (!choose_projection(il, ip)) {
                return;
            }
            auto & layer = layers[il];
            auto & p = layer.projections[ip];
            const size_t bytes = std::min({chunk_bytes, p.bytes - p.cursor, inflight_limit - inflight});
            const ggml_spacemit_transfer_request request{
                p.destination ? p.destination + p.cursor : nullptr, p.source + p.cursor, bytes};
            const auto status = p.destination
                ? api.submit(transfer, &request, available->event)
                : api.prefetch(transfer, p.source + p.cursor, bytes, available->event);
            if (status == GGML_SPACEMIT_TRANSFER_BUSY) {
                // Preserve every cursor and scheduling position. Sleep before
                // retrying; overloaded speculative work never blocks inference.
                return;
            }
            if (status != GGML_SPACEMIT_TRANSFER_OK) {
                transfer_failed = true;
                return;
            }
            available->expert_slot = p.expert_slot;
            available->bytes = bytes;
            available->pending = true;
            inflight += bytes;
            p.cursor += bytes;
            layer.next_projection = (ip + 1) % layer.projections.size();
            next_layer = (il + 1) % n_layers;
        }
    }

    static void dispatch_prefill_entry(impl * self) noexcept {
        try {
            self->dispatch_prefill_loop();
        } catch (...) {
            // Keep optional thread/synchronization failures from escaping a
            // thread entry point. Already accepted work stays alive until stop.
            self->transfer_failed = true;
        }
    }

    void dispatch_prefill_loop() {
        while (!stopping.load(std::memory_order_acquire)) {
            reap();
            if (!transfer_failed) {
                refill_prefill();
            }
            if (transfer_failed) {
                return;
            }
            // No wait-any/completion notification exists in the transfer API.
            // Timed sleeping observes new activations and terminal events even
            // when there are no more graph callbacks or decode calls. Always
            // sleep between refill rounds, including BUSY and idle rounds;
            // the compute callback need not acquire/notify any synchronization.
            std::unique_lock<std::mutex> lock(dispatch_mutex);
            dispatch_wakeup.wait_for(lock, std::chrono::milliseconds(1), [this] {
                return stopping.load(std::memory_order_acquire);
            });
        }
    }

    void stop() noexcept {
        if (stopped) {
            return;
        }
        stopped = true;
        // Prevent further refill rounds before detaching. An already executing
        // submission may finish; join() quiesces it before cancellation/free.
        stopping.store(true, std::memory_order_release);
        dispatch_wakeup.notify_one();
        if (callback_bound) {
            // Caller synchronized its context. The backend setter holds the
            // shared stream mutex, waiting out callback snapshots in any other
            // handle's graph before removing the borrowed user_data pointer.
            api.set_callback(backend, nullptr, nullptr);
            callback_bound = false;
        }
        if (dispatcher.joinable()) {
            dispatcher.join();
        }
        // The callback is detached and dispatcher is gone: only this thread
        // can now query/cancel/free events, with no producer left to race it.
        if (transfer) {
            for (auto & slot : slots) {
                if (!slot.event) {
                    continue;
                }
                while (api.query(slot.event) == GGML_SPACEMIT_TRANSFER_PENDING) {
                    if (api.cancel(slot.event) != GGML_SPACEMIT_TRANSFER_BUSY) {
                        break;
                    }
                    std::this_thread::yield();
                }
            }
            // Cancellation is cooperative. All mappings/events and the service
            // survive terminal acknowledgement, including outstanding faults.
            api.drain(transfer);
            for (auto & slot : slots) {
                if (slot.event) {
                    api.event_free(slot.event);
                    slot.event = nullptr;
                }
                slot.pending = false;
                slot.bytes = 0;
            }
            api.free(transfer);
            transfer = nullptr;
        }
        inflight = 0;
        layers.clear();
        activated.reset();
        n_layers = 0;
        if (owns_callback) {
            spacemit_prefetch_owner.clear(std::memory_order_release);
            owns_callback = false;
        }
        if (transfer_failed) {
            LLAMA_LOG_WARN("spacemit prefetch: source warmup stopped after a transfer/dispatcher failure\n");
        }
    }
};

llama_spacemit_prefetch::llama_spacemit_prefetch(
        const llama_model & model, const std::vector<ggml_backend_ptr> & backends) {
    try {
        auto state = std::make_unique<impl>();
        state->start_prefill(model, backends);
        pimpl = std::move(state);
    } catch (...) {
        // Partial setup belongs to the temporary state; its destructor stops
        // and joins it before this optional initialization failure is reported.
        LLAMA_LOG_WARN("spacemit prefetch: initialization failed; warmup disabled\n");
    }
}

llama_spacemit_prefetch::~llama_spacemit_prefetch() = default;

void llama_spacemit_prefetch::stop() noexcept {
    if (pimpl) {
        pimpl->stop();
    }
}
