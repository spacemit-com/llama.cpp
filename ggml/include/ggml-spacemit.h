// SPDX-License-Identifier: MIT

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_spacemit_init(void);

GGML_BACKEND_API bool ggml_backend_is_spacemit(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_spacemit_reg(void);

// Semantic windows recognized from llama graph names and data dependencies.
// QKV includes projection preparation (normalization/RoPE). ATTENTION starts
// after attn_norm and ends at the named attention output; windows may overlap.
enum ggml_spacemit_module {
    GGML_SPACEMIT_MODULE_QKV,
    GGML_SPACEMIT_MODULE_RMS_NORM,
    GGML_SPACEMIT_MODULE_ATTENTION,
    GGML_SPACEMIT_MODULE_ROUTER,
    GGML_SPACEMIT_MODULE_EXPERTS,
    GGML_SPACEMIT_MODULE_PREDICTOR, // speculative top-k, before QKV
};

enum ggml_spacemit_prefetch_phase {
    GGML_SPACEMIT_PREFETCH_BEGIN,
    GGML_SPACEMIT_PREFETCH_END,
    GGML_SPACEMIT_PREFETCH_READY, // ROUTER/PREDICTOR: expert indices are ready
};

struct ggml_spacemit_prefetch_event {
    enum ggml_spacemit_module module;
    enum ggml_spacemit_prefetch_phase phase;
    int layer;
    size_t opnode_index;
    // BEGIN/END: boundary tensor; ROUTER/PREDICTOR READY: the I32 top-k tensor/view.
    const struct ggml_tensor * tensor;
};

// Runs once per event on worker 0. All compute workers wait for the callback
// before continuing. BEGIN can enqueue asynchronous prefetch; EXPERTS BEGIN
// can wait for the corresponding weights. The callback itself must not throw,
// reenter this backend, or do blocking transfer work at a prefetch submission.
// No transfer thread or expert-selection policy is supplied by this API.
// Event/tensor contents are borrowed: copy indices during READY if a background
// task needs them; graph buffers may be reused as soon as the callback returns.
// Callbacks must use the actual tensor strides and backend buffer access rules.
typedef void (*ggml_spacemit_prefetch_callback)(
        const struct ggml_spacemit_prefetch_event * event, void * user_data);

// Enable/disable prefetch planning while idle. Disabled by default. Without a
// callback this only recognizes/caches windows; it does not move weights or add
// worker barriers. Also exposed as "ggml_backend_set_prefetch" through the registry.
// Handles for the same device share this setting.
GGML_BACKEND_API bool ggml_backend_spacemit_set_prefetch(ggml_backend_t backend, bool enabled);

// Configure while idle. Handles for the same device share this setting.
// Installing a callback enables prefetch; removing it disables prefetch. An
// explicit set_prefetch call can subsequently pause/resume the installed callback.
// user_data must outlive all invocations; nullptr disables events. Neither
// recognition nor extra event barriers run when disabled. Also available via
// ggml_backend_reg_get_proc_address("ggml_backend_spacemit_set_prefetch_callback").
GGML_BACKEND_API bool ggml_backend_spacemit_set_prefetch_callback(
        ggml_backend_t backend, ggml_spacemit_prefetch_callback callback, void * user_data);

#ifdef __cplusplus
}
#endif
