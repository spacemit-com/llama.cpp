// SPDX-FileCopyrightText: Copyright (c) 2026 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT

#include "spacemit_spert_adapter.h"

#include "ggml-cpu-impl.h"
#include "ime_env.h"

// spine-runtime public headers (release package: SPERT_DIR/include).
// spert_abi.h  : the stable C ABI (spine_require_stream / spine_parallel_dispatch_1d /
//   spine_grid / spine_ctx_sync ...).
// spert_engine.hpp : declares the tile-scoped ops still missing from the C ABI in 0.6.0
//   (ctx_shared_buffer). It is an exported symbol; we call it directly. Once a future
//   spine-runtime release exposes a pure-C shared-buffer accessor, drop spert_engine.hpp.
#include "spert_abi.h"
#include "spert_engine.hpp"

#include <cstdio>
#include <vector>

// Thread-local tile handle for the AI core currently running the graph kernel.
// Set before calling the graph callback, cleared after. The C ABI passes the
// tile context as an int64 handle (== spert::detail::Tile* reinterpreted).
static thread_local int64_t t_ctx = 0;

extern "C" {

void spacemit_spert_grid_sync(void) {
    if (t_ctx) {
        // Whole-grid barrier: synchronize all tiles of this launch via the
        // spine-runtime C ABI.
        spine_ctx_sync(t_ctx);
    }
}

void * spacemit_spert_shared_buffer(long * out_size) {
    if (t_ctx) {
        spert::SharedBufferView v = spert::detail::ctx_shared_buffer();
        if (out_size) {
            *out_size = (long) v.size;
        }
        return v.data;
    }
    if (out_size) {
        *out_size = 0;
    }
    return nullptr;
}

struct graph_kernel_ctx {
    void (*fn)(int ith, int nth, void * ud);
    void * user_data;
    int    n_tiles;
};

// C ABI kernel entry: void(int64_t ctx, void * function_args), invoked once per tile.
static void graph_kernel_wrapper(int64_t ctx, void * function_args) {
    graph_kernel_ctx * gk = (graph_kernel_ctx *) function_args;
    t_ctx = ctx;
    const int ith = (int) spine_grid(ctx, 0);
    gk->fn(ith, gk->n_tiles, gk->user_data);
    t_ctx = 0;
}

int spacemit_spert_launch_graph_kernel(void (*graph_kernel_fn)(int ith, int nth, void * user_data),
                                       void * user_data,
                                       int    n_tiles) {
    // Acquire a fresh stream (checks out n_tiles AI cores from the SHM arbiter) per graph
    // compute and release it on return. Keeping a stream alive across calls would pin those
    // cores forever and starve other backends (e.g. the ONNX runtime used for multimodal
    // inference); scoping it to this call releases the cores between graph computes.
    const auto & preferred_core_ids =
        ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
    if (n_tiles <= 0 || static_cast<size_t>(n_tiles) > preferred_core_ids.size()) {
        fprintf(stderr,
                "[spert_adapter] invalid tile count %d for %zu preferred cores\n",
                n_tiles,
                preferred_core_ids.size());
        return -1;
    }

    std::vector<int64_t> core_ids;
    core_ids.reserve(static_cast<size_t>(n_tiles));
    for (int i = 0; i < n_tiles; ++i) {
        core_ids.push_back(preferred_core_ids[static_cast<size_t>(i)]);
    }

    int64_t stream = spine_require_stream_with_config(n_tiles, core_ids.data());
    if (!stream) {
        fprintf(stderr, "[spert_adapter] spine_require_stream_with_config failed\n");
        return -1;
    }

    graph_kernel_ctx gk;
    gk.fn        = graph_kernel_fn;
    gk.user_data = user_data;
    gk.n_tiles   = n_tiles;

    // Synchronous dispatch: launches the 1D grid and waits for all tiles to terminate.
    spine_parallel_dispatch_1d(stream, (void *) graph_kernel_wrapper, &gk, n_tiles);

    spine_release_stream(stream);
    return 0;
}

}  // extern "C"
