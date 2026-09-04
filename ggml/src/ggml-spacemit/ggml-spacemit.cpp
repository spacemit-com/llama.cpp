#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-spacemit.h"

#include "spacemit-session.h"
#include "spacemit-kernels.h"
#include "spacemit-opnode.h"
#include "spacemit-context.h"

#include "ime.h"
#include "repack.h"
#include "spacemit-env.h"
#include "spine_mem_pool.h"
#include "rvv_kernels.h"

#include "ggml-cpu.h"
#include "ggml-profile.h"

// spine-runtime C++ API (hard dependency)
#include <spert.hpp>
#define SPACEMIT_HAS_SPERT 1

using namespace ggml::cpu::riscv64_spacemit;

// global_spine_env_info is initialized from spert::backend_info() and
// GGML_SPACEMIT_WORKERS instead of parsing /proc/cpuinfo.

namespace ggml::cpu::riscv64_spacemit {

spine_env_info::spine_env_info() {
    // Query spine-runtime for hardware info
    spert::BackendInfo info = spert::backend_info();

    // Determine IME support from the CC core architecture id
    uint16_t arch = (uint16_t) info.core_arch_id;

    // A100 = 0xA064, A200 = 0xA0C8, X100 = 0x5064
    perfer_core_arch_id = spine_core_arch_id{ arch };

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;
    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    // num_cores from env, fallback to spert backend_info
    const char * workers_str = getenv("GGML_SPACEMIT_WORKERS");
    if (workers_str) {
        num_cores = atoi(workers_str);
        if (num_cores <= 0) num_cores = 1;
    } else {
        num_cores = (int) info.num_cores;
        if (num_cores <= 0) num_cores = 1;
    }
    vlen = info.vlen;
    mem_backend = spine_mem_pool_backend::transparent_hugepage;
    const char * mem_backend_str = getenv("SPACEMIT_MEM_BACKEND");
    if (mem_backend_str) {
        if (strcmp(mem_backend_str, "hugepage") == 0) {
            mem_backend = spine_mem_pool_backend::transparent_hugepage;
        } else if (strcmp(mem_backend_str, "posix") == 0) {
            mem_backend = spine_mem_pool_backend::posix_memalign;
        } else if (strcmp(mem_backend_str, "hugetlb") == 0) {
            mem_backend = spine_mem_pool_backend::hugetlb_1g;
        }
    }

    // Allocate init_barrier (needed by ime.cpp kernel barriers)
    init_barrier = new spine_barrier_t[spine_init_barrier_count];
    spine_barrier_init(init_barrier, spine_init_barrier_count, 2);

    GGML_LOG_INFO("ggml-spacemit: num_cores=%d, arch_id=0x%x, vlen=%zu, shared_mem=%zu, use_ime1=%d, use_ime2=%d\n",
                  num_cores, (unsigned) arch, info.vlen, info.shared_mem_size, use_ime1, use_ime2);
}

spine_env_info::~spine_env_info() {
    delete[] init_barrier;
    init_barrier = nullptr;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit

//** static config

static int  opt_fusion  = 1;

//** helpers

static inline bool op_is_compute(ggml_tensor * node) {
    return !ggml_op_is_empty(node->op) && !ggml_is_empty(node) && (node->flags & GGML_TENSOR_FLAG_COMPUTE);
}

static spacemit_op_code op_remap_to_spacemit(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_MUL_MAT:    return SPACEMIT_OP_MUL_MAT;
        case GGML_OP_MUL_MAT_ID: return SPACEMIT_OP_MUL_MAT_ID;
        case GGML_OP_ADD:        return SPACEMIT_OP_ADD;
        case GGML_OP_RMS_NORM:   return SPACEMIT_OP_RMS_NORM;
        case GGML_OP_ROPE:       return SPACEMIT_OP_ROPE;
        case GGML_OP_SOFT_MAX:   return SPACEMIT_OP_SOFTMAX;
        case GGML_OP_RESHAPE:    return SPACEMIT_OP_RESHAPE;
        case GGML_OP_VIEW:       return SPACEMIT_OP_VIEW;
        case GGML_OP_PERMUTE:    return SPACEMIT_OP_PERMUTE;
        case GGML_OP_TRANSPOSE:  return SPACEMIT_OP_TRANSPOSE;
        case GGML_OP_NONE:       return SPACEMIT_OP_NONE;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(t)) {
                case GGML_UNARY_OP_SILU: return SPACEMIT_OP_UNARY_SILU;
                case GGML_UNARY_OP_GELU: return SPACEMIT_OP_UNARY_GELU;
                default:                 break;
            }
            break;
        default:
            break;
    }
    return SPACEMIT_OP_INVALID;
}

//** op fusion helpers
//
// Adapted from ggml-hexagon.cpp. Four fusion patterns:
//   1. RMS_NORM + MUL     -> SPACEMIT_OP_RMS_NORM_MUL
//   2. MUL_MAT + ADD      -> SPACEMIT_OP_MUL_MAT_ADD
//   3. QKV merge (3 MUL_MAT with same src1) -> SPACEMIT_OP_MUL_MAT_QKV
//   4. FFN merge (2 MUL_MAT with same src1) -> SPACEMIT_OP_MUL_MAT_FFN

static bool is_mergeable_mul_mat(const ggml_tensor * t) {
    if (!t || t->op != GGML_OP_MUL_MAT)   return false;
    if (!t->src[1])                       return false;
    if (t->src[1]->type != GGML_TYPE_F32) return false;
    return ggml_is_quantized(t->src[0]->type);
}

static bool is_mergeable_mul_mat_pair(const ggml_tensor * n1, const ggml_tensor * n2) {
    if (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        return false;
    }
    if (n1->src[1] != n2->src[1]) {
        return false;
    }
    if (n1->src[0]->ne[0] != n2->src[0]->ne[0] ||
        n1->src[0]->ne[1] != n2->src[0]->ne[1]) {
        return false;
    }
    if (n1->src[0]->type != n2->src[0]->type) {
        return false;
    }
    return true;
}

static bool is_qkv_mergeable(const ggml_tensor * n_q, const ggml_tensor * n_k, const ggml_tensor * n_v) {
    if (!is_mergeable_mul_mat(n_q) || !is_mergeable_mul_mat(n_k) || !is_mergeable_mul_mat(n_v)) {
        return false;
    }
    if (n_q->src[1] != n_k->src[1] || n_q->src[1] != n_v->src[1]) {
        return false;
    }
    if (n_q->src[0]->type != n_k->src[0]->type || n_q->src[0]->type != n_v->src[0]->type) {
        return false;
    }
    if (n_k->src[0]->ne[0] != n_v->src[0]->ne[0] ||
        n_k->src[0]->ne[1] != n_v->src[0]->ne[1]) {
        return false;
    }
    if (n_q->src[0]->ne[0] != n_k->src[0]->ne[0]) {
        return false;
    }
    return true;
}

static bool try_fuse_node(const ggml_cgraph * graph, int & i, std::vector<spacemit_opnode> & nodes) {
    if (!opt_fusion) {
        return false;
    }

    ggml_tensor * n = graph->nodes[i];
    ggml_tensor * next_node = (i + 1 < graph->n_nodes) ? graph->nodes[i + 1] : nullptr;

    // Pattern 1: RMS_NORM + MUL
    if (n->op == GGML_OP_RMS_NORM && next_node) {
        if (next_node->op == GGML_OP_MUL && op_is_compute(next_node) &&
            ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            spacemit_opnode node(n, {}, SPACEMIT_OP_RMS_NORM_MUL);
            node.add_fused(next_node);
            nodes.push_back(std::move(node));
            i++;  // skip the fused MUL node
            return true;
        }
    }

    // Pattern 3: QKV merge (3 consecutive MUL_MAT with same src1)
    // Pattern 4: FFN merge (2 consecutive MUL_MAT with same src1)
    if (is_mergeable_mul_mat(n)) {
        ggml_tensor * n1 = (i + 1 < graph->n_nodes) ? graph->nodes[i + 1] : nullptr;
        ggml_tensor * n2 = (i + 2 < graph->n_nodes) ? graph->nodes[i + 2] : nullptr;

        if (is_qkv_mergeable(n, n1, n2)) {
            // Reorder to KVQ: K (n1), V (n2), Q (n)
            spacemit_opnode node(n1, {}, SPACEMIT_OP_MUL_MAT_QKV);
            node.add_fused(n2, true);
            node.add_fused(n, true);
            nodes.push_back(std::move(node));
            i += 2;
            return true;
        }

        if (is_mergeable_mul_mat_pair(n, n1)) {
            spacemit_opnode node(n, {}, SPACEMIT_OP_MUL_MAT_FFN);
            node.add_fused(n1, true);
            nodes.push_back(std::move(node));
            i += 1;
            return true;
        }
    }

    // Pattern 2: MUL_MAT + ADD
    if (n->op == GGML_OP_MUL_MAT && next_node) {
        if (next_node->op == GGML_OP_ADD && op_is_compute(next_node) &&
            ggml_can_fuse(graph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD })) {
            if (next_node->src[0] == n || next_node->src[1] == n) {
                spacemit_opnode node(n, {}, SPACEMIT_OP_MUL_MAT_ADD);
                node.add_fused(next_node);
                nodes.push_back(std::move(node));
                i += 1;
                return true;
            }
        }
    }

    return false;
}

//** buffer interface
//
// Reuses spine_mem_pool_alloc/free for allocation and repack for init_tensor/set_tensor.
// Pattern adapted from ggml-cpu/spacemit/ime.cpp buffer management.

static void ggml_backend_spacemit_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    void * base = buffer->context;
    if (base == nullptr) {
        return;
    }

    spine_mem_pool_free(base);
}

static void * ggml_backend_spacemit_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    void * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    return base;
}

static enum ggml_status ggml_backend_spacemit_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                   ggml_tensor *         tensor) {
    tensor->extra =
        (void *) const_cast<ggml::spacemit::tensor_traits_base *>(ggml_spacemit_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_spacemit_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                         ggml_tensor *         tensor,
                                                         uint8_t               value,
                                                         size_t                offset,
                                                         size_t                size) {
    GGML_ASSERT(tensor);
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_spacemit_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);

    void * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    memset(base, value, buffer->size);
}

static void ggml_backend_spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                      ggml_tensor *         tensor,
                                                      const void *          data,
                                                      size_t                offset,
                                                      size_t                size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));

    // Quantized tensors with tensor traits are physically repacked, so they
    // must be uploaded atomically. Plain tensors preserve the ggml layout and
    // support the partial writes used by test-backend-ops and graph inputs.
    if (tensor->extra) {
        GGML_ASSERT(offset == 0);
        GGML_ASSERT(size == ggml_nbytes(tensor));
        auto ok = ggml_riscv64_spacemit_repack_tensor(tensor, data, size);
        GGML_ASSERT(ok == 0);
    } else {
        memcpy((char *) tensor->data + offset, data, size);
    }

    GGML_UNUSED(buffer);
}

static void ggml_backend_spacemit_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                     const ggml_tensor *   tensor,
                                                     void *                data,
                                                     size_t                offset,
                                                     size_t                size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static const ggml_backend_buffer_i ggml_backend_spacemit_buffer_i = {
    /* .free_buffer     = */ ggml_backend_spacemit_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_spacemit_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_spacemit_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_spacemit_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_spacemit_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_spacemit_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_spacemit_buffer_clear,
    /* .reset           = */ nullptr,
};

//** buffer type interface

static const char * ggml_backend_spacemit_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "SPACEMIT";
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_spacemit_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                              size_t size) {
    void * base = spine_mem_pool_alloc(size, 64);
    if (base == nullptr) {
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_spacemit_buffer_i, base, size);
}

static size_t ggml_backend_spacemit_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_spacemit_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return SIZE_MAX;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_spacemit_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_spacemit_nbytes(buft, tensor);
}

static bool ggml_backend_spacemit_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_spacemit_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_spacemit_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_spacemit_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_spacemit_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_spacemit_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_spacemit_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_spacemit_buffer_type_is_host,
};

static ggml_backend_buffer_type_t ggml_backend_spacemit_buffer_type(ggml_backend_dev_t dev) {
    static struct ggml_backend_buffer_type buft_s = {
        /* .iface   = */ ggml_backend_spacemit_buffer_type_interface,
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    if (buft_s.device == nullptr) {
        buft_s.device = dev;
    }
    return &buft_s;
}

static bool ggml_backend_buffer_is_spacemit(const struct ggml_backend_buffer * b) {
    return b && b->buft->iface.get_name == ggml_backend_spacemit_buffer_type_get_name;
}

//** backend interface

spacemit_session::~spacemit_session() {
    free(workspace);
}

static const char * ggml_backend_spacemit_name(ggml_backend_t backend) {
    auto sess = static_cast<spacemit_session *>(backend->context);
    return sess->c_name();
}

static void ggml_backend_spacemit_free(ggml_backend_t backend) {
    // The device registry owns the shared session.
    delete backend;
}

static ggml_status ggml_backend_spacemit_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<spacemit_session *>(backend->context);
    // Serialize hardware use across backend handles. The Stream itself remains
    // graph-scoped so idle contexts do not retain CC hardware resources.
    std::lock_guard<std::mutex> stream_lock(sess->stream_mutex);

    GGML_LOG_DEBUG("ggml-spacemit: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    int n_threads = sess->num_cores > 0 ? sess->num_cores : 1;
    size_t workspace_size = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (!op_is_compute(node)) {
            continue;
        }

        size_t node_workspace_size = 0;
        if (ggml_spacemit_get_work_size(n_threads, node, &node_workspace_size)) {
            workspace_size = std::max(workspace_size, node_workspace_size);
        }
    }

    if (sess->workspace_size < workspace_size) {
        void * new_workspace = realloc(sess->workspace, workspace_size);
        if (new_workspace == nullptr) {
            GGML_LOG_ERROR("ggml-spacemit: failed to allocate work buffer (%zu bytes)\n", workspace_size);
            return GGML_STATUS_ALLOC_FAILED;
        }
        sess->workspace      = new_workspace;
        sess->workspace_size = workspace_size;
    }
    auto * workspace = static_cast<uint8_t *>(sess->workspace);

    spert::Stream stream(sess->num_cores);
    if (!stream.valid()) {
        GGML_LOG_ERROR("ggml-spacemit: failed to create spert stream\n");
        return GGML_STATUS_FAILED;
    }

    ggml_tensor ** nodes   = graph->nodes;
    int             n_nodes = graph->n_nodes;

    auto * contexts = sess->contexts.data();
    auto fut = stream.launch(
        spert::Grid{(uint32_t)n_threads},
        [nodes, n_nodes, workspace_size = sess->workspace_size, workspace, contexts,
         shared_mem_size = spert::backend_info().shared_mem_size](spert::Context * runtime) {
            // Match the old ggml-cpu/spacemit path: use the complete
            // worker-local TCM buffer directly.  alloc_shared() is a
            // dynamic sub-allocation API and is not equivalent here; it
            // adds allocator bookkeeping and can select a different shared
            // pool block layout than the original RVV kernels expect.
            auto shared = shared_mem_size > 0 ? runtime->shared_buffer() : spert::SharedBufferView{};
            if (shared_mem_size > 0 && (!shared || shared.size < shared_mem_size)) {
                throw std::runtime_error("ggml-spacemit: worker shared buffer unavailable");
            }

            const uint32_t ith = runtime->program_id(0);
            auto & ctx = contexts[ith];
            ctx.reset(*runtime, ith, runtime->grid_dim(0), workspace, workspace_size, shared);

            for (int i = 0; i < n_nodes; i++) {
                ggml_tensor * node = nodes[i];

                if (!op_is_compute(node)) {
                    continue;
                }

                ggml_profile_log_spacemit_op_begin(node, (int) ith, (int) runtime->grid_dim(0));

                if (!ggml_spacemit_compute_forward(ctx, node)) {
                    throw std::runtime_error(std::string("ggml-spacemit: failed to dispatch op ") +
                                             ggml_op_desc(node) + " (" + ggml_type_name(node->type) + ")");
                }
                ggml_profile_log_spacemit_op_end(node, (int) ith, (int) runtime->grid_dim(0));

                // Every core must finish the current node before any core starts
                // the next one. A bounded pointer look-ahead is not sufficient:
                // dependencies can be farther away or hidden behind aliases/views.
                ctx.sync();
            }

            ctx.clear();
        }
    );

    const spert::Status status = fut.sync();

    if (status != spert::Status::Ok) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_spacemit_synchronize(ggml_backend_t backend) {
    // graph_compute waits on its launch future before returning, so there is
    // no outstanding asynchronous work to synchronize here.
    GGML_UNUSED(backend);
}

static struct ggml_backend_i spacemit_backend_i = {
    /* .get_name                = */ ggml_backend_spacemit_name,
    /* .free                    = */ ggml_backend_spacemit_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_spacemit_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_spacemit_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_spacemit_guid() {
    static ggml_guid guid = { 0x8b, 0x68, 0xed, 0xbf, 0xef, 0x23, 0x2e, 0x4a,
                              0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22 };
    return &guid;
}

bool ggml_backend_is_spacemit(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_spacemit_name;
}

//** device interface

static ggml_backend_t ggml_backend_spacemit_device_init(ggml_backend_dev_t dev, const char * params) {
    auto sess = static_cast<spacemit_session *>(dev->context);

    return new ggml_backend{
        /* .guid      = */ ggml_backend_spacemit_guid(),
        /* .interface = */ spacemit_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

static const char * ggml_backend_spacemit_device_get_name(ggml_backend_dev_t dev) {
    auto sess = static_cast<spacemit_session *>(dev->context);
    return sess->c_name();
}

static const char * ggml_backend_spacemit_device_get_description(ggml_backend_dev_t dev) {
    return "Spacemit K3/X200 AI Engine";
    GGML_UNUSED(dev);
}

static void ggml_backend_spacemit_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // SPACEMIT backend shares host memory (weights are in DRAM accessible by CC cores).
    // Report available system memory so the scheduler assigns tensors to us.
    size_t vfree = 0, vtotal = 0;
    ggml_backend_dev_t cpu_dev = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    if (cpu_dev && cpu_dev->iface.get_memory) {
        cpu_dev->iface.get_memory(cpu_dev, &vfree, &vtotal);
    }
    *free  = vfree;
    *total = vtotal;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_spacemit_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    GGML_UNUSED(dev);
}

static void ggml_backend_spacemit_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_spacemit_device_get_name(dev);
    props->description = ggml_backend_spacemit_device_get_description(dev);
    props->type        = ggml_backend_spacemit_device_get_type(dev);
    ggml_backend_spacemit_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_buffer_type_t ggml_backend_spacemit_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_spacemit_buffer_type(dev);
}

static bool ggml_backend_spacemit_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // all srcs and dst must be on our buffer type
    auto check_buf = [&](const ggml_tensor * t) -> bool {
        if (!t || !t->buffer) return true;  // unallocated is OK
        return ggml_backend_buffer_is_spacemit(t->buffer);
    };

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && !check_buf(op->src[i])) {
            return false;
        }
    }
    if (!check_buf(op)) {
        return false;
    }

    bool supp = false;
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            supp = true;
            break;

        case GGML_OP_SCALE:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_ROPE:
        case GGML_OP_UNARY:
        case GGML_OP_GLU:
        case GGML_OP_GET_ROWS:
        case GGML_OP_CONCAT:
        case GGML_OP_CPY:
        case GGML_OP_SET_ROWS:
        case GGML_OP_CONT:
        case GGML_OP_REPEAT:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_L2_NORM:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_PAD:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SET:
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_SSM_CONV:
            supp = ggml_spacemit_get_tensor_traits(op) != nullptr;
            break;

        default:
            break;
    }

    return supp;

    GGML_UNUSED(dev);
}

static bool ggml_backend_spacemit_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_spacemit_buffer_type_get_name;
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_spacemit_device_i = {
    /* .get_name             = */ ggml_backend_spacemit_device_get_name,
    /* .get_description      = */ ggml_backend_spacemit_device_get_description,
    /* .get_memory           = */ ggml_backend_spacemit_device_get_memory,
    /* .get_type             = */ ggml_backend_spacemit_device_get_type,
    /* .get_props            = */ ggml_backend_spacemit_device_get_props,
    /* .init_backend         = */ ggml_backend_spacemit_device_init,
    /* .get_buffer_type      = */ ggml_backend_spacemit_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_spacemit_device_supports_op,
    /* .supports_buft        = */ ggml_backend_spacemit_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//** backend registry

#define GGML_SPACEMIT_MAX_DEVICES 1

struct ggml_spacemit_registry {
    ggml_spacemit_registry(ggml_backend_reg_t reg);
    ~ggml_spacemit_registry();

    ggml_backend_device devices[GGML_SPACEMIT_MAX_DEVICES];
};

ggml_spacemit_registry::ggml_spacemit_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-spacemit: Spacemit backend (experimental) : allocating new registry\n");

    for (size_t i = 0; i < GGML_SPACEMIT_MAX_DEVICES; i++) {
        devices[i].iface = ggml_backend_spacemit_device_i;
        devices[i].reg   = reg;

        auto * sess = new spacemit_session();

        // populate session from spert backend_info and env
        spert::BackendInfo info = spert::backend_info();
        sess->num_cores = global_spine_env_info.num_cores;
        sess->arch_id   = static_cast<int64_t>(global_spine_env_info.perfer_core_arch_id);
        sess->vlen      = (int64_t) info.vlen;
        sess->use_ime1  = global_spine_env_info.use_ime1;
        sess->use_ime2  = global_spine_env_info.use_ime2;
        sess->contexts.resize(sess->num_cores);

        sess->name = "SPACEMIT" + std::to_string(i);

        devices[i].context = sess;
    }
}

ggml_spacemit_registry::~ggml_spacemit_registry() {
    for (size_t i = 0; i < GGML_SPACEMIT_MAX_DEVICES; i++) {
        auto sess = static_cast<spacemit_session *>(devices[i].context);
        delete sess;
    }
}

static const char * ggml_backend_spacemit_reg_get_name(ggml_backend_reg_t reg) {
    return "SPACEMIT";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_spacemit_reg_get_device_count(ggml_backend_reg_t reg) {
    return GGML_SPACEMIT_MAX_DEVICES;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_spacemit_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto hreg = static_cast<ggml_spacemit_registry *>(reg->context);

    if (index >= GGML_SPACEMIT_MAX_DEVICES || !hreg->devices[index].context) {
        return nullptr;
    }

    return &hreg->devices[index];
}

static void * ggml_backend_spacemit_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    return NULL;
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static void ggml_spacemit_init(ggml_backend_reg * reg) {
    const char * str_fusion  = getenv("GGML_SPACEMIT_FUSION");

    opt_fusion  = str_fusion ? atoi(str_fusion)  : opt_fusion;

    reg->context = new ggml_spacemit_registry(reg);
}

static const struct ggml_backend_reg_i ggml_backend_spacemit_reg_i = {
    /* .get_name         = */ ggml_backend_spacemit_reg_get_name,
    /* .get_device_count = */ ggml_backend_spacemit_reg_get_device_count,
    /* .get_device       = */ ggml_backend_spacemit_reg_get_device,
    /* .get_proc_address = */ ggml_backend_spacemit_get_proc_address,
};

ggml_backend_reg_t ggml_backend_spacemit_reg(void) {
    static bool initialized = false;

    static ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_spacemit_reg_i,
                                    /* .context     = */ NULL };

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_spacemit_init(&reg);
        }
        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_spacemit_init(void) {
    ggml_backend_reg_t reg = ggml_backend_spacemit_reg();
    if (!reg) {
        return nullptr;
    }

    ggml_backend_dev_t dev = ggml_backend_spacemit_reg_get_device(reg, 0);
    if (!dev) {
        return nullptr;
    }

    return ggml_backend_spacemit_device_init(dev, nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_spacemit_reg)
