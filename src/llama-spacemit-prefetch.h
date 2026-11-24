// SPDX-License-Identifier: MIT
#pragma once

#include "ggml-cpp.h"
#include <memory>
#include <vector>
struct llama_model;

// Both policies share a model-owned ratio-sized expert buffer and one transfer
// service. Prefill reuses its fixed layer-sized prefix for all current experts.
// With at least two workers, prefill also reserves one request lane for independent
// page warmup while the remaining lanes copy mmap pages into slots. The warmer
// shares the worker service and is cancelled at router READY; it owns no slots.
// Decode consumes predictor IDs before QKV, preserves hits with LRU, and cancels
// outstanding copies at router READY. Complete actual experts are pinned until
// EXPERTS END; misses and unfinished copies read the loader's existing mmap.
// Cancellation is cooperative: destinations cannot be reused until every write
// is terminal. Graph callbacks do descriptor/ID work only, never payload copies,
// waits, drains or mmap creation. ratio=-1 retains page warmup without slots;
// ratio=0 uses mmap only. Neither policy bounds OS page cache or RSS.
//
// Construct/stop while compute is idle. The model and backends outlive stop().
// One manager owns the shared SpacemiT callback process-wide; inference on the
// shared backend is serialized by its stream lock. A direct callback user must
// coordinate with this owner. Predictors currently attach to DeepSeek41 decode.
class llama_spacemit_prefetch {
public:
    llama_spacemit_prefetch(const llama_model & model, const std::vector<ggml_backend_ptr> & backends);
    ~llama_spacemit_prefetch();
    void stop() noexcept;
    llama_spacemit_prefetch(const llama_spacemit_prefetch &) = delete;
    llama_spacemit_prefetch & operator=(const llama_spacemit_prefetch &) = delete;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
