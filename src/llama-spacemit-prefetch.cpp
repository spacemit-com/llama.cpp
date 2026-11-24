// SPDX-License-Identifier: MIT
#include "llama-spacemit-prefetch.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-spacemit-expert-cache.h"

#include "ggml-spacemit.h"
#include "ggml-spacemit-transfer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <list>
#include <cstring>
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
        submit       = spacemit_prefetch_proc<decltype(submit)>(reg,       "ggml_spacemit_transfer_submit");
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
    struct layer_plan {
        std::vector<const ggml_tensor *> weights;
        std::vector<uint8_t> predicted, actual;
        bool has_prediction = false, reading = false;
    };
    struct copy_job {
        size_t layer, expert;
        uint64_t window;
        llama_spacemit_expert_cache::reservation reservation;
        std::vector<llama_spacemit_expert_cache::range> ranges;
        size_t projection = 0, cursor = 0, pending = 0;
        bool aborted = false;
        bool pagewarm = false;
    };
    struct event_slot {
        ggml_spacemit_transfer_event_t event = nullptr;
        copy_job * job = nullptr;
        size_t bytes = 0;
        bool pending = false;
    };
    llama_spacemit_expert_cache * expert_cache = nullptr; // model-owned
    spacemit_prefetch_api api;
    ggml_backend_t backend = nullptr;
    ggml_spacemit_transfer_t transfer = nullptr;
    std::vector<layer_plan> layers;
    std::list<copy_job> jobs; // stable addresses until every accepted write is terminal
    std::vector<event_slot> slots;
    std::atomic<bool> stopping{false};
    std::thread dispatcher;
    std::mutex mutex;
    std::condition_variable wakeup;
    uint64_t window = 0;
    int window_layer = -1;
    bool window_open = false, decode = false;
    std::vector<uint8_t> warmed; // independent pagewarm progress in the current PP window
    std::vector<uint8_t> wanted, completed; // current window, preallocated
    bool owns_callback = false, callback_bound = false, stopped = false;
    bool transfer_failed = false; // mutex held, or control thread after join()

    ~impl() { stop(); }

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

    size_t collect_weights(const llama_model & model) {
        expert_cache = model.expert_cache.get();
        layers.resize(model.layers.size());
        size_t count = 0, max_experts = 0;
        std::unordered_set<const ggml_tensor *> seen;
        for (size_t il = 0; il < layers.size(); ++il) {
            const auto & src = model.layers[il];
            const ggml_tensor * tensors[] = {
                src.ffn_gate_up_exps ? nullptr : src.ffn_gate_exps,
                src.ffn_gate_up_exps ? nullptr : src.ffn_up_exps,
                src.ffn_down_exps, src.ffn_gate_up_exps,
                src.ffn_gate_chexps, src.ffn_up_chexps, src.ffn_down_chexps,
            };
            size_t experts = 0;
            for (const auto * t : tensors) {
                if (!eligible(model, t) || !ggml_is_contiguous(t) || t->ne[3] != 1 ||
                        (experts && experts != size_t(t->ne[2])) || !seen.insert(t).second) { continue; }
                experts = size_t(t->ne[2]);
                layers[il].weights.push_back(t);
                ++count;
            }
            layers[il].predicted.assign(experts, 0);
            layers[il].actual.assign(experts, 0);
            max_experts = std::max(max_experts, experts);
        }
        wanted.resize(max_experts);
        completed.resize(max_experts);
        warmed.resize(max_experts);
        return count;
    }

    void append_warm_range(copy_job & job, const ggml_tensor * tensor, size_t expert) {
        const auto * data = static_cast<const unsigned char *>(tensor->data) + expert*tensor->nb[2];
        job.ranges.push_back({data, nullptr, tensor->nb[2], 0});
    }

    void start(const llama_model & model, const std::vector<ggml_backend_ptr> & backends) {
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
        const size_t ranges = collect_weights(model);
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
        // CPUs with >10% busy time in a 100ms sample, and binds each
        // transfer worker to a distinct CPU before accepting work. No caller-side CPU ranges,
        // affinity changes, A100 fallback, or payload work on a compute thread.
        // Selection remains an availability snapshot, not a CPU reservation.
        if (model.prefetch_threads() < 1) {
            throw std::invalid_argument("prefetch_threads must be at least 1");
        }
        const size_t workers = size_t(model.prefetch_threads());
        const ggml_spacemit_transfer_config config{workers, workers};
        ggml_spacemit_transfer_status status = GGML_SPACEMIT_TRANSFER_UNAVAILABLE;
        transfer = api.create(&config, &status);
        if (!transfer) {
            LLAMA_LOG_WARN("spacemit prefetch: cannot start %zu X100 transfer workers (status %d); warmup disabled\n",
                    workers, static_cast<int>(status));
            stop();
            return;
        }
        slots.resize(workers);
        for (auto & slot : slots) {
            slot.event = api.event_create();
            if (!slot.event) {
                LLAMA_LOG_WARN("spacemit prefetch: event allocation failed; warmup disabled\n");
                stop();
                return;
            }
        }
        // This unpinned thread owns job storage and dispatches descriptors.
        // Payload touches use the X100 workers.
        dispatcher = std::thread(&impl::dispatch_entry, this);
        if (!api.set_callback(backend, &impl::on_event, this)) {
            LLAMA_LOG_WARN("spacemit prefetch: callback registration failed; warmup disabled\n");
            stop();
            return;
        }
        callback_bound = true;
        LLAMA_LOG_INFO("spacemit prefetch: %zu X100 transfer workers on distinct available CPUs\n", slots.size());
        if (expert_cache) {
            LLAMA_LOG_INFO("spacemit prefetch: shared prefill/decode copy enabled, %zu slots, %zu bytes, %zu banks; fixed prefill region, decode LRU, router cancellation\n",
                    expert_cache->capacity(), expert_cache->buffer_bytes(), ranges);
            if (slots.size() > 1 && expert_cache->capacity()) {
                LLAMA_LOG_INFO("spacemit prefetch: concurrent prefill pagewarm + mmap-to-slot copy, 1 pagewarm lane and %zu copy lanes; router cancellation\n", slots.size() - 1);
            }
            return;
        }
        LLAMA_LOG_INFO("spacemit prefetch: shared prefill/decode dispatcher, %zu expert banks, "
                "%zu X100 workers, one expert request per worker, 16 MiB chunks, "
                "router cancellation and mmap fallback\n", ranges, slots.size());
    }

    // Tensor contents are borrowed only inside the callback. Respect all four
    // strides, deduplicate across sequences, and never fault model weight pages.
    static bool read_ids(const ggml_tensor * ids, std::vector<uint8_t> & selected) {
        std::fill(selected.begin(), selected.end(), 0);
        if (!ids || ids->type != GGML_TYPE_I32 || !ids->data ||
                (ids->buffer && !ggml_backend_buffer_is_host(ids->buffer))) { return false; }
        for (int64_t i3 = 0; i3 < ids->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < ids->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
                        int32_t id;
                        const auto * p = static_cast<const unsigned char *>(ids->data) +
                                i0*ids->nb[0] + i1*ids->nb[1] + i2*ids->nb[2] + i3*ids->nb[3];
                        std::memcpy(&id, p, sizeof(id));
                        if (id < 0 || size_t(id) >= selected.size()) {
                            std::fill(selected.begin(), selected.end(), 0);
                            return false;
                        }
                        selected[size_t(id)] = 1;
                    }
                }
            }
        }
        return true;
    }

    static void on_event(const ggml_spacemit_prefetch_event * event, void * user_data) noexcept {
        auto & self = *static_cast<impl *>(user_data);
        if (!event || event->layer < 0 || size_t(event->layer) >= self.layers.size() ||
                self.stopping.load(std::memory_order_acquire)) { return; }
        try {
            std::lock_guard<std::mutex> lock(self.mutex);
            self.handle_event(*event);
            self.wakeup.notify_one();
        } catch (...) {
            // Optional prefetch must not throw through a compute worker. Stop
            // dispatching; accepted requests are kept alive through stop().
            self.stopping.store(true, std::memory_order_release);
        }
    }

    void handle_event(const ggml_spacemit_prefetch_event & event) {
        const size_t il = size_t(event.layer);
        auto & layer = layers[il];
        if (event.module == GGML_SPACEMIT_MODULE_PREDICTOR && event.phase == GGML_SPACEMIT_PREFETCH_READY) {
            read_ids(event.tensor, layer.predicted);
            layer.has_prediction = true; // invalid IDs mean no speculation, not prefill
        } else if (event.module == GGML_SPACEMIT_MODULE_QKV && event.phase == GGML_SPACEMIT_PREFETCH_BEGIN) {
            // Backend splits can emit several QKV fragments in one microbatch.
            // Keep its original transfers alive until that layer's router runs.
            // After router_ready(), reading is true, so even a single-layer
            // model can open a fresh window in the next microbatch.
            if (window_open && window_layer == event.layer && !layer.reading) { return; }
            cancel_jobs(); // an older window may still have cancelled writes
            // CPU-offloaded experts do not emit a SpacemiT EXPERTS END.
            // The next layer/token's QKV depends on the completed prior FFN,
            // so this boundary also safely releases any outstanding readers.
            for (size_t previous = 0; previous < layers.size(); ++previous) {
                if (expert_cache && layers[previous].reading) { expert_cache->end_experts(previous); }
                layers[previous].reading = false;
            }
            ++window;
            window_layer = event.layer;
            window_open = true;
            decode = layer.has_prediction;
            LLAMA_LOG_DEBUG("spacemit prefetch: begin %s layer=%d window=%llu\n",
                    decode ? "decode" : "prefill", event.layer, (unsigned long long) window);
            std::fill(wanted.begin(), wanted.end(), 0);
            std::fill(completed.begin(), completed.end(), 0);
            std::fill(warmed.begin(), warmed.end(), 0);
            for (size_t e = 0; e < layer.predicted.size(); ++e) {
                wanted[e] = decode ? layer.predicted[e] : 1;
            }
            layer.has_prediction = false;
            if (expert_cache) {
                // Protect decode hits before admitting misses, so a hit later
                // in the prediction list cannot be evicted by an earlier miss.
                if (decode) { expert_cache->protect(il, wanted); }
            }
        } else if (event.module == GGML_SPACEMIT_MODULE_ROUTER && event.phase == GGML_SPACEMIT_PREFETCH_READY) {
            router_ready(il, event.tensor);
        } else if (event.module == GGML_SPACEMIT_MODULE_EXPERTS && event.phase == GGML_SPACEMIT_PREFETCH_BEGIN) {
            // Also handles router/expert graph splits: the first expert matmul
            // already carries the actual IDs even if its producer was elsewhere.
            if (!layer.reading && event.tensor && event.tensor->op == GGML_OP_MUL_MAT_ID) {
                router_ready(il, event.tensor->src[2]);
            }
        } else if (event.module == GGML_SPACEMIT_MODULE_EXPERTS && event.phase == GGML_SPACEMIT_PREFETCH_END) {
            if (expert_cache && layer.reading) { expert_cache->end_experts(il); }
            layer.reading = false;
            if (window_layer == event.layer) {
                window_open = false;
                cancel_jobs();
            }
        }
    }

    void router_ready(size_t il, const ggml_tensor * ids) {
        // Querying worker descriptors only observes terminal state; no payload,
        // wait, drain, remapping or new demand copy runs on this compute worker.
        reap();
        // Both prefill and decode yield storage bandwidth to demand mmap reads.
        // A worker faulting a mapped page cannot acknowledge until it resumes.
        if (window_layer == int(il)) {
            window_open = false;
            cancel_jobs();
        }
        auto & layer = layers[il];
        read_ids(ids, layer.actual); // failure selects no slots, preserving mmap
        if (expert_cache) { expert_cache->begin_experts(il, layer.actual); }
        layer.reading = true;
    }

    void cancel_jobs() {
        for (auto & job : jobs) { job.aborted = true; }
        for (auto & slot : slots) {
            if (slot.pending && slot.job->aborted) { api.cancel(slot.event); }
        }
        // An aborted reservation remains writing until reap() sees ALL of its
        // events terminal. In particular, BUSY cancellation never frees a slot.
    }

    void reap() {
        for (auto & slot : slots) {
            if (!slot.pending) { continue; }
            const auto state = api.query(slot.event);
            if (state == GGML_SPACEMIT_TRANSFER_PENDING) { continue; }
            if (state != GGML_SPACEMIT_TRANSFER_COMPLETE) { slot.job->aborted = true; }
            if (state == GGML_SPACEMIT_TRANSFER_FAILED) { transfer_failed = true; }
            --slot.job->pending;
            slot.bytes = 0;
            slot.pending = false;
            slot.job = nullptr;
        }
        for (auto it = jobs.begin(); it != jobs.end();) {
            auto & job = *it;
            const bool complete = job.projection == job.ranges.size();
            if (job.pending || (!job.aborted && !complete)) { ++it; continue; }
            if (expert_cache && !job.pagewarm) {
                if (job.aborted) { expert_cache->discard(job.reservation); }
                else { expert_cache->publish(job.reservation); }
            }
            if (!job.aborted && job.window == window) {
                (job.pagewarm ? warmed : completed)[job.expert] = 1;
            }
            it = jobs.erase(it);
        }
    }

    void admit_jobs() {
        const size_t il = size_t(window_layer);
        for (size_t e = 0; e < layers[il].predicted.size(); ++e) {
            if (!wanted[e] || completed[e]) { continue; }
            if (expert_cache && expert_cache->contains(il, e)) {
                // Prefill needs the fixed region even if decode left a hit in
                // another part of the buffer. reserve() decides whether to move.
                if (decode) { completed[e] = 1; continue; }
            }
            const auto existing = std::find_if(jobs.begin(), jobs.end(), [=](const copy_job & job) {
                return !job.pagewarm && job.layer == il && job.expert == e;
            });
            if (existing != jobs.end()) { continue; }
            copy_job job{il, e, window, {}, {}};
            if (expert_cache) {
                job.reservation = expert_cache->reserve(il, e, !decode);
                if (!job.reservation) { continue; }
                try { job.ranges = expert_cache->ranges(job.reservation); }
                catch (...) { expert_cache->discard(job.reservation); throw; }
            } else {
                for (const auto * t : layers[il].weights) { append_warm_range(job, t, e); }
            }
            try { jobs.push_back(std::move(job)); }
            catch (...) {
                if (expert_cache && !job.pagewarm) { expert_cache->discard(job.reservation); }
                throw;
            }
        }
    }

    void refill() {
        if (!window_open || window_layer < 0) { return; }
        if (expert_cache && !expert_cache->capacity()) { return; } // ratio 0: mmap only
        admit_jobs();
        const bool x100_pagewarm_lane = expert_cache && !decode && slots.size() > 1;
        const bool concurrent_pagewarm = expert_cache && !decode && slots.size() > 1;
        if (concurrent_pagewarm) {
            const size_t il = size_t(window_layer);
            for (size_t e = 0; e < layers[il].predicted.size(); ++e) {
                if (!wanted[e] || warmed[e] || completed[e]) { continue; }
                const auto existing = std::find_if(jobs.begin(), jobs.end(), [=](const copy_job & job) {
                    return job.pagewarm && job.layer == il && job.expert == e;
                });
                if (existing != jobs.end()) { continue; }
                copy_job job{il, e, window, {}, {}};
                job.pagewarm = true;
                for (const auto * t : layers[il].weights) { append_warm_range(job, t, e); }
                jobs.push_back(std::move(job));
            }
        }
        for (auto & slot : slots) {
            if (slot.pending) { continue; }
            // One lane warms mapped pages independently while the others copy.
            const auto eligible_job = [](const copy_job & job) {
                if (job.projection == job.ranges.size()) { return false; }
                if (job.pending) { return false; }
                return !job.pagewarm || !job.ranges[job.projection].destination;
            };
            const bool warm_lane = x100_pagewarm_lane && &slot == &slots.front();
            auto it = std::find_if(jobs.begin(), jobs.end(), [&](const copy_job & job) {
                return eligible_job(job) && job.pagewarm == warm_lane;
            });
            if (it == jobs.end() && warm_lane) {
                it = std::find_if(jobs.begin(), jobs.end(), [&](const copy_job & job) {
                    return eligible_job(job) && !job.pagewarm;
                });
            }
            if (it == jobs.end()) { continue; }
            auto & job = *it;
            const auto & range = job.ranges[job.projection];
            const size_t bytes = std::min(chunk_bytes, range.bytes - job.cursor);
            ggml_spacemit_transfer_status status;
            if (range.destination) {
                const ggml_spacemit_transfer_request request{
                        range.destination + job.cursor, range.source + job.cursor, bytes};
                status = api.submit(transfer, &request, slot.event);
            } else {
                status = api.prefetch(transfer, range.source + job.cursor, bytes, slot.event);
            }
            if (status == GGML_SPACEMIT_TRANSFER_BUSY) { return; }
            if (status != GGML_SPACEMIT_TRANSFER_OK) { transfer_failed = true; return; }
            slot.job = &job;
            slot.pending = true;
            slot.bytes = bytes;
            ++job.pending;
            job.cursor += bytes;
            if (job.cursor == range.bytes) { ++job.projection; job.cursor = 0; }
        }
    }

    static void dispatch_entry(impl * self) noexcept {
        try { self->dispatch_loop(); }
        catch (...) { self->stopping.store(true, std::memory_order_release); }
    }

    void dispatch_loop() {
        std::unique_lock<std::mutex> lock(mutex);
        while (!stopping.load(std::memory_order_acquire)) {
            reap();
            // Retry BUSY cancellations without blocking the router callback.
            for (auto & slot : slots) {
                if (slot.pending && slot.job->aborted) { api.cancel(slot.event); }
            }
            if (transfer_failed) { window_open = false; cancel_jobs(); }
            else { refill(); }
            // wakeup on graph events; timed polling observes transfer completion
            // when no further compute callback is available to notify us.
            wakeup.wait_for(lock, std::chrono::milliseconds(1));
        }
    }

    void stop() noexcept {
        if (stopped) { return; }
        stopped = true;
        stopping.store(true, std::memory_order_release);
        wakeup.notify_one();
        if (callback_bound) {
            api.set_callback(backend, nullptr, nullptr);
            callback_bound = false;
        }
        if (dispatcher.joinable()) { dispatcher.join(); }
        if (transfer) {
            for (auto & slot : slots) {
                if (!slot.pending) { continue; }
                while (api.query(slot.event) == GGML_SPACEMIT_TRANSFER_PENDING &&
                        api.cancel(slot.event) == GGML_SPACEMIT_TRANSFER_BUSY) { std::this_thread::yield(); }
            }
            api.drain(transfer);
            for (auto & slot : slots) {
                if (slot.event) { api.event_free(slot.event); slot.event = nullptr; }
            }
            // Every write is now acknowledged, so reservations can be released.
            for (auto & job : jobs) { if (expert_cache && !job.pagewarm) { expert_cache->discard(job.reservation); } }
            api.free(transfer);
            transfer = nullptr;
        }
        if (expert_cache) {
            for (size_t il = 0; il < layers.size(); ++il) {
                if (layers[il].reading) { expert_cache->end_experts(il); }
            }
        }
        jobs.clear();
        if (owns_callback) { spacemit_prefetch_owner.clear(std::memory_order_release); owns_callback = false; }
        if (transfer_failed) { LLAMA_LOG_WARN("spacemit prefetch: transfer failed; using mmap fallback\n"); }
    }
};

llama_spacemit_prefetch::llama_spacemit_prefetch(
        const llama_model & model, const std::vector<ggml_backend_ptr> & backends) {
    try {
        auto state = std::make_unique<impl>();
        state->start(model, backends);
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
