#ifndef GGML_PREFETCH_DEBUG_H
#define GGML_PREFETCH_DEBUG_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Debug-only CPU/Spacemit simulation. Configuration is fixed at first use.
GGML_API bool ggml_prefetch_debug_ideal_enabled(void);
// Called by every worker; the normal executor barrier must follow this call.
// Returns false for operations which must execute normally.
GGML_API bool ggml_prefetch_debug_ideal_compute(struct ggml_tensor * tensor, int ith, int nth);

#ifdef __cplusplus
}
#endif

#endif // GGML_PREFETCH_DEBUG_H
