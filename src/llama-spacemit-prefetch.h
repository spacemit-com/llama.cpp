// SPDX-License-Identifier: MIT
#pragma once

#include "ggml-cpp.h"

#include <memory>
#include <vector>

struct llama_model;

// Prefill strategy: asynchronous page warmup or bounded expert copying.
// Decode will use a separate strategy; the transfer API remains shared.
// Construct and stop on an idle control thread; model mappings and backend
// handles must outlive stop(). Construction may sample X100 CPU load for 100ms.
//
// QKV BEGIN only activates preallocated layer state. A sleeping dispatcher
// advances every activated layer through its complete eligible weight ranges
// once per context, refilling <=16 MiB requests as events become terminal.
// At most four requests / 64 MiB can be in flight. With an explicit expert
// ratio, requests copy into the model-owned fixed slot buffer; each complete
// expert is release-published for CPU MUL_MAT_ID, with mmap fallback until then.
// Without a ratio, requests only warm the source mapping without a byte quota.
// Neither mode bounds OS page-cache residency or RSS. No pages are pinned or
// evicted and tensor data pointers are never changed.
//
// Backend handles may share callback state. At most one manager in this llama
// library instance owns it process-wide; competing managers remain inactive.
// Direct users of the backend callback API must coordinate with this owner.
class llama_spacemit_prefetch {
public:
    llama_spacemit_prefetch(const llama_model & model, const std::vector<ggml_backend_ptr> & backends);
    ~llama_spacemit_prefetch();

    // Idempotent, control-thread only. Quiesce compute first. Prevents dispatch,
    // unbinds the callback, joins the dispatcher, then cancels/drains accepted
    // requests before releasing events and the transfer service.
    void stop() noexcept;

    llama_spacemit_prefetch(const llama_spacemit_prefetch &) = delete;
    llama_spacemit_prefetch & operator=(const llama_spacemit_prefetch &) = delete;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
