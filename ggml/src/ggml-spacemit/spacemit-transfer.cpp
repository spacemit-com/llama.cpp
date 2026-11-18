// SPDX-License-Identifier: MIT
#include "spacemit-transfer.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct ggml_spacemit_transfer_event {
    std::atomic<ggml_spacemit_transfer_state> state{GGML_SPACEMIT_TRANSFER_IDLE};
    std::mutex mutex;
    std::condition_variable done;
};

struct ggml_spacemit_transfer {
    struct job {
        ggml_spacemit_transfer_request request{};
        ggml_spacemit_transfer_event_t event = nullptr;
    };

    std::mutex mutex;
    std::condition_variable work, idle, startup;
    std::vector<job> queue;
    std::vector<int> cpus;
    std::vector<std::thread> workers;
    size_t head = 0, tail = 0, queued = 0, active = 0, started = 0;
    bool stopping = false, startup_failed = false;

    ~ggml_spacemit_transfer() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        work.notify_all();
        for (auto & worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void run() {
        const bool bound = spacemit_transfer_provider_bind(cpus);
        {
            std::lock_guard<std::mutex> lock(mutex);
            startup_failed |= !bound;
            ++started;
            startup.notify_one();
        }
        if (!bound) {
            return;
        }
        for (;;) {
            job next;
            {
                std::unique_lock<std::mutex> lock(mutex);
                work.wait(lock, [&] { return stopping || queued != 0; });
                if (queued == 0 && stopping) {
                    return;
                }
                next = queue[head];
                head = (head + 1) % queue.size();
                --queued;
                ++active;
            }
            const bool ok = next.request.size == 0 || spacemit_transfer_provider_copy(
                    next.request.dst, next.request.src, next.request.size);
            {
                // Notify while holding the event lock: event_free/wait cannot
                // destroy the condition variable before notification finishes.
                std::lock_guard<std::mutex> lock(next.event->mutex);
                next.event->state.store(ok ? GGML_SPACEMIT_TRANSFER_COMPLETE : GGML_SPACEMIT_TRANSFER_FAILED,
                                        std::memory_order_release);
                next.event->done.notify_all();
            }
            // Do not touch the event again: its owner may already have freed it.
            {
                std::lock_guard<std::mutex> lock(mutex);
                --active;
                if (queued == 0 && active == 0) {
                    idle.notify_all();
                }
            }
        }
    }
};

ggml_spacemit_transfer_t ggml_spacemit_transfer_create(
        const ggml_spacemit_transfer_config * config, ggml_spacemit_transfer_status * status) {
    auto fail = [&](ggml_spacemit_transfer_status value) -> ggml_spacemit_transfer_t {
        if (status) { *status = value; }
        return nullptr;
    };
    try {
        std::unique_ptr<ggml_spacemit_transfer> transfer(new ggml_spacemit_transfer);
        if (!spacemit_transfer_provider_available_cpus(transfer->cpus) || transfer->cpus.empty()) {
            return fail(GGML_SPACEMIT_TRANSFER_UNAVAILABLE);
        }
        const size_t n = config && config->n_workers ? config->n_workers : 1;
        const size_t capacity = config && config->queue_capacity ? config->queue_capacity : 64;
        if (n > transfer->cpus.size()) {
            return fail(GGML_SPACEMIT_TRANSFER_INVALID);
        }
        transfer->queue.resize(capacity);
        transfer->workers.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            transfer->workers.emplace_back([p = transfer.get()] { p->run(); });
        }
        {
            std::unique_lock<std::mutex> lock(transfer->mutex);
            transfer->startup.wait(lock, [&] { return transfer->started == n; });
            if (transfer->startup_failed) {
                return fail(GGML_SPACEMIT_TRANSFER_UNAVAILABLE);
            }
        }
        if (status) { *status = GGML_SPACEMIT_TRANSFER_OK; }
        return transfer.release();
    } catch (...) {
        return fail(GGML_SPACEMIT_TRANSFER_ERROR);
    }
}

void ggml_spacemit_transfer_free(ggml_spacemit_transfer_t transfer) {
    delete transfer;
}

ggml_spacemit_transfer_event_t ggml_spacemit_transfer_event_create(void) {
    try { return new ggml_spacemit_transfer_event; } catch (...) { return nullptr; }
}

void ggml_spacemit_transfer_event_free(ggml_spacemit_transfer_event_t event) {
    if (!event) { return; }
    (void) ggml_spacemit_transfer_wait(event);
    delete event;
}

ggml_spacemit_transfer_status ggml_spacemit_transfer_submit(
        ggml_spacemit_transfer_t transfer, const ggml_spacemit_transfer_request * request,
        ggml_spacemit_transfer_event_t event) {
    if (!transfer || !request || !event || (request->size && (!request->src || !request->dst))) {
        return GGML_SPACEMIT_TRANSFER_INVALID;
    }
    std::unique_lock<std::mutex> lock(transfer->mutex, std::try_to_lock);
    if (!lock.owns_lock()) { return GGML_SPACEMIT_TRANSFER_BUSY; }
    if (transfer->stopping) { return GGML_SPACEMIT_TRANSFER_UNAVAILABLE; }
    if (transfer->queued == transfer->queue.size()) { return GGML_SPACEMIT_TRANSFER_BUSY; }
    std::unique_lock<std::mutex> event_lock(event->mutex, std::try_to_lock);
    if (!event_lock.owns_lock() || event->state.load(std::memory_order_acquire) == GGML_SPACEMIT_TRANSFER_PENDING) {
        return GGML_SPACEMIT_TRANSFER_BUSY;
    }
    transfer->queue[transfer->tail] = {*request, event};
    transfer->tail = (transfer->tail + 1) % transfer->queue.size();
    ++transfer->queued;
    event->state.store(GGML_SPACEMIT_TRANSFER_PENDING, std::memory_order_release);
    transfer->work.notify_one();
    return GGML_SPACEMIT_TRANSFER_OK;
}

ggml_spacemit_transfer_state ggml_spacemit_transfer_query(ggml_spacemit_transfer_event_t event) {
    return event ? event->state.load(std::memory_order_acquire) : GGML_SPACEMIT_TRANSFER_IDLE;
}

ggml_spacemit_transfer_status ggml_spacemit_transfer_wait(ggml_spacemit_transfer_event_t event) {
    if (!event) { return GGML_SPACEMIT_TRANSFER_INVALID; }
    std::unique_lock<std::mutex> lock(event->mutex);
    event->done.wait(lock, [&] {
        return event->state.load(std::memory_order_acquire) != GGML_SPACEMIT_TRANSFER_PENDING;
    });
    switch (event->state.load(std::memory_order_acquire)) {
        case GGML_SPACEMIT_TRANSFER_COMPLETE: return GGML_SPACEMIT_TRANSFER_OK;
        case GGML_SPACEMIT_TRANSFER_FAILED:   return GGML_SPACEMIT_TRANSFER_ERROR;
        default:                            return GGML_SPACEMIT_TRANSFER_INVALID;
    }
}

void ggml_spacemit_transfer_drain(ggml_spacemit_transfer_t transfer) {
    if (!transfer) { return; }
    std::unique_lock<std::mutex> lock(transfer->mutex);
    transfer->idle.wait(lock, [&] { return transfer->queued == 0 && transfer->active == 0; });
}

void * spacemit_transfer_get_proc_address(const char * name) {
#define TRANSFER_PROC(fn) if (std::strcmp(name, #fn) == 0) { return reinterpret_cast<void *>(fn); }
    TRANSFER_PROC(ggml_spacemit_transfer_create)
    TRANSFER_PROC(ggml_spacemit_transfer_free)
    TRANSFER_PROC(ggml_spacemit_transfer_event_create)
    TRANSFER_PROC(ggml_spacemit_transfer_event_free)
    TRANSFER_PROC(ggml_spacemit_transfer_submit)
    TRANSFER_PROC(ggml_spacemit_transfer_query)
    TRANSFER_PROC(ggml_spacemit_transfer_wait)
    TRANSFER_PROC(ggml_spacemit_transfer_drain)
#undef TRANSFER_PROC
    return nullptr;
}
