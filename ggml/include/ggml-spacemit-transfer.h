// SPDX-License-Identifier: MIT
#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// A core-independent asynchronous byte-copy service. Create/free on an idle,
// non-AI control thread, NOT in graph callbacks or compute_forward. The current
// provider uses workers restricted to sampled available X100 CPUs; no A100 work,
// AI-thread registration, DMA, or transfer-specific instructions are exposed.
typedef struct ggml_spacemit_transfer * ggml_spacemit_transfer_t;
typedef struct ggml_spacemit_transfer_event * ggml_spacemit_transfer_event_t;

enum ggml_spacemit_transfer_status {
    GGML_SPACEMIT_TRANSFER_OK,
    GGML_SPACEMIT_TRANSFER_BUSY,
    GGML_SPACEMIT_TRANSFER_INVALID,
    GGML_SPACEMIT_TRANSFER_UNAVAILABLE,
    GGML_SPACEMIT_TRANSFER_ERROR,
    GGML_SPACEMIT_TRANSFER_CANCELLED,
};

enum ggml_spacemit_transfer_state {
    GGML_SPACEMIT_TRANSFER_IDLE,
    GGML_SPACEMIT_TRANSFER_PENDING,
    GGML_SPACEMIT_TRANSFER_COMPLETE,
    GGML_SPACEMIT_TRANSFER_FAILED,
    GGML_SPACEMIT_TRANSFER_CANCELLED_STATE,
};

struct ggml_spacemit_transfer_config {
    size_t n_workers;       // 0: one worker; each binds a distinct available provider CPU
    size_t queue_capacity;  // 0: 64 pending requests (in-flight work is additional)
};

struct ggml_spacemit_transfer_request {
    void * dst;
    const void * src;
    size_t size;
};

// Returns NULL if the provider is unavailable or initialization fails. No silent
// fallback onto a submitting/compute thread. Config may be NULL for defaults.
// Current provider samples CPU load for 100ms during creation; this must not run
// on the submission/compute path. Availability is a snapshot, not a reservation.
GGML_BACKEND_API ggml_spacemit_transfer_t ggml_spacemit_transfer_create(
        const struct ggml_spacemit_transfer_config * config,
        enum ggml_spacemit_transfer_status * status);

// Stops submissions and joins workers, finishing all accepted requests (including
// cancellation acknowledgments). Does not implicitly cancel work. The caller
// must quiesce producers first; no concurrent API call may use this engine.
GGML_BACKEND_API void ggml_spacemit_transfer_free(ggml_spacemit_transfer_t transfer);

// Allocate events on the control thread before inference. One pending request
// per event; reuse only after completion AND all previous observers have left.
GGML_BACKEND_API ggml_spacemit_transfer_event_t ggml_spacemit_transfer_event_create(void);
GGML_BACKEND_API void ggml_spacemit_transfer_event_free(ggml_spacemit_transfer_event_t event);

// Copies only the descriptor into a preallocated bounded queue. Never allocates,
// waits for queue space/completion, or copies payload on the caller. BUSY means
// queue full, queue/event lock contention, or an already-pending event; retry
// later or skip speculative prefetch. Non-OK leaves event and buffers unchanged.
// src/dst must be valid CPU-accessible ordinary coherent memory, kept alive and
// free of conflicting accesses until completion. Overlap within one request is
// allowed (memmove semantics); different requests have no dependency ordering.
// TCM, MMIO, device buffers and noncoherent memory are NOT supported by this API.
// A zero-byte request may use NULL addresses. The event must be non-NULL.
GGML_BACKEND_API enum ggml_spacemit_transfer_status ggml_spacemit_transfer_submit(
        ggml_spacemit_transfer_t transfer,
        const struct ggml_spacemit_transfer_request * request,
        ggml_spacemit_transfer_event_t event);

// Asynchronously read one byte from every OS page intersecting [src, src+size).
// Reuses the bounded queue/events and the same nonblocking submission rules.
// Does not allocate a second payload, change tensor pointers, or pin pages.
// COMPLETE means every page was touched; the OS may subsequently evict pages.
// Concurrent readers are allowed. Keep the CPU-readable mapping and event alive
// until terminal acknowledgment (or drain); no concurrent unmap or writes.
// A zero-size request may use NULL. Payload reads occur only on provider workers.
GGML_BACKEND_API enum ggml_spacemit_transfer_status ggml_spacemit_transfer_prefetch(
        ggml_spacemit_transfer_t transfer, const void * src, size_t size,
        ggml_spacemit_transfer_event_t event);

// Request cooperative cancellation without waiting. OK means accepted (or already
// CANCELLED_STATE), NOT that the worker has stopped. BUSY means event-lock
// contention; retry later. NULL, IDLE, COMPLETE and FAILED return INVALID.
// A queued request is skipped when dequeued; its queue slot is not freed early.
// An active provider operation finishes its current chunk (at most 64 KiB)
// first; cancellation does not interrupt an outstanding kernel page fault.
// Keep event/src/dst alive and free of conflicting accesses until query reports
// a terminal state or wait returns. A cancelled destination may be partly or
// fully written; never consume it as a successful copy. Cancellation accepted
// before terminal publication wins over completion/failure. Reuse requires all
// old observers/cancellers to leave, just like submission; no generation handle.
GGML_BACKEND_API enum ggml_spacemit_transfer_status ggml_spacemit_transfer_cancel(
        ggml_spacemit_transfer_event_t event);

// Acquire observation of COMPLETE publishes the copied bytes to the observer.
// CANCELLED_STATE also acknowledges that all buffer accesses have stopped.
GGML_BACKEND_API enum ggml_spacemit_transfer_state ggml_spacemit_transfer_query(
        ggml_spacemit_transfer_event_t event);
// Wait only at the consumption boundary. IDLE/NULL returns INVALID;
// acknowledged cancellation returns CANCELLED (never OK).
GGML_BACKEND_API enum ggml_spacemit_transfer_status ggml_spacemit_transfer_wait(
        ggml_spacemit_transfer_event_t event);
// Wait until queue and in-flight work are empty. Does not prevent new submissions.
GGML_BACKEND_API void ggml_spacemit_transfer_drain(ggml_spacemit_transfer_t transfer);

#ifdef __cplusplus
}
#endif
