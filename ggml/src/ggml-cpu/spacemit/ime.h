#pragma once

#include "ggml-alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(void);

// Returns the number of preferred (AI) cores, e.g. 8 on K3.
GGML_API int ggml_backend_cpu_riscv64_spacemit_num_prefer_cores(void);

GGML_API void * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment);

GGML_API void ggml_backend_cpu_riscv64_spacemit_free_shared(void * ptr);

#ifdef __cplusplus
}
#endif
