// SPDX-License-Identifier: MIT
#pragma once

#include "ggml-spacemit-transfer.h"
#include <cstdint>
#include <vector>

// Private provider boundary. The public service/graph code does not depend on
// CPU topology. CPU discovery is control-thread-only; bind runs in each worker
// before it accepts any requests. copy runs ONLY in successfully bound workers.
// Returns online, affinity-eligible X100 CPUs with <=10% non-idle time in a
// 100ms sample. Busy CPUs are excluded. Control-thread-only (may sleep).
// This is a load snapshot, NOT a CPU reservation or a future-idleness guarantee.
bool spacemit_transfer_provider_available_cpus(std::vector<int> & cpus);
bool spacemit_transfer_provider_bind(const std::vector<int> & cpus);
bool spacemit_transfer_provider_copy(void * dst, const void * src, size_t size);
// Read-only page touching, also restricted to bound workers.
bool spacemit_transfer_provider_prefetch(const void * src, size_t size);

void * spacemit_transfer_get_proc_address(const char * name);

// Internal sampling predicate, shared with deterministic host tests. All times
// are cumulative /proc/stat ticks; busy includes iowait conservatively.
struct spacemit_transfer_cpu_time {
    uint64_t busy = 0;
    uint64_t idle = 0;
};
inline bool spacemit_transfer_cpu_is_available(const spacemit_transfer_cpu_time & before,
                                               const spacemit_transfer_cpu_time & after) {
    if (after.busy < before.busy || after.idle < before.idle) { return false; }
    const uint64_t busy = after.busy - before.busy;
    const uint64_t idle = after.idle - before.idle;
    // busy / (busy + idle) <= 10%, without overflow or floating point.
    return (busy != 0 || idle != 0) && busy <= idle / 9;
}
