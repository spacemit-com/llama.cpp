#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "ime.h"

#include "common.h"
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ime_kernels.h"
#include "repack.h"
#include "rvv_kernels.h"
#include "scalar_kernels.h"
#include "spacemit-context.h"
#include "spacemit-env.h"
#include "spine_mem_pool.h"
#include "vec.h"

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>  // for GGML_ASSERT
#include <cstring>
#include <new>
#include <stdexcept>
#include <thread>
// clang-format off
#if defined(__riscv)

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#else
#include <riscv_vector.h>
#endif

#if !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#endif

#if !defined(__riscv_zba)
#error "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#endif

#if defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#else
#error "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#endif

#else

#error "riscv not enabled in this build"

#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> constexpr size_t get_repacked_block_type_size() {
    if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        return sizeof(block_q8_0);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
        return sizeof(block_q4_0) * INTER_SIZE / QK4_0;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K>) {
        return (sizeof(block_q4_0) + sizeof(uint8_t)) * INTER_SIZE / QK4_1;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        return sizeof(spacemit_kernels::nrow_block_q2_k<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        return sizeof(spacemit_kernels::nrow_block_q3_k<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
        return sizeof(spacemit_kernels::nrow_block_mxfp4<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K>) {
        return sizeof(spacemit_kernels::nrow_block_q5_1<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_0>) {
        return sizeof(spacemit_kernels::nrow_block_q5_0<1>);
    } else {
        assert(false);
        return 0;
    }
}


inline bool spine_q4_m_split_enabled(const ggml_tensor * tensor = nullptr) {
    if (!global_spine_env_info.use_ime2) {
        return false;
    }
    return tensor == nullptr || (tensor->ne[0] % QK_K == 0 && tensor->ne[1] % 32 == 0);
}

inline size_t spine_q4_k_compressed_nbytes(const ggml_tensor * tensor) {
    GGML_ASSERT(tensor != nullptr && tensor->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(tensor->ne[0] % QK_K == 0 && tensor->ne[1] % 32 == 0);
    const size_t nrows   = static_cast<size_t>(ggml_nrows(tensor));
    const size_t nblocks = static_cast<size_t>(tensor->ne[0] / QK_K);
    return nrows * nblocks * size_t{ 142 };
}

inline size_t spine_q4_k_expanded_offset(const ggml_tensor * tensor) {
    return GGML_PAD(spine_q4_k_compressed_nbytes(tensor), size_t{ 64 });
}

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
inline int64_t spine_row_stride_b(int64_t b_k_blks) {
    if constexpr (std::is_same_v<BLOC_TYPE, block_q4_K> && INTER_SIZE == 32 && NB_COLS == 32) {
        return (b_k_blks / 8) * 142;
    }
    if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K>) {
        return (b_k_blks / 8) * static_cast<int64_t>(sizeof(spacemit_kernels::nrow_super_q6_k<1>));
    }
    return b_k_blks * static_cast<int64_t>(get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>());
}

struct spine_dyn_n_schedule_state {
    int64_t counters[8];
};

// Two X100 threads unpack Q4 N32 tiles into a bounded DRAM ring while
// the A100 workers consume the preceding tiles.
// Space out readers without imposing 64-byte alignment on realloc's workspace.
struct spine_q4_reader_progress {
    std::atomic<int64_t> tile{ -1 };
    uint8_t              padding[64 - sizeof(std::atomic<int64_t>)];
};

static_assert(sizeof(spine_q4_reader_progress) == 64);

struct spine_q4_stream_state {
    spine_q4_reader_progress consumed[8];
};

struct spine_q4_stream_host {
    std::atomic<uint64_t> generation{0};
    std::atomic<uint64_t> completed[2]{};
    std::atomic<bool>     ready[2]{};
    std::atomic<bool>     failed{false};
    std::atomic<bool>     stop{false};
    void (*run)(void *, int) = nullptr;
    void * arg = nullptr;
    std::thread workers[2];
};

inline bool spine_q4_stream_shape(int64_t m, int64_t k, int64_t n) {
    return m == 128 && k > 0 && k % 32 == 0 && n > 0 && n % 32 == 0;
}

bool spine_q4_stream_graph_enabled(const ggml_cgraph * graph) {
    if (!global_spine_env_info.use_ime2) {
        return false;
    }
    const size_t tcm_size = spert::backend_info().shared_mem_size;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * op = graph->nodes[i];
        if (op->op != GGML_OP_MUL_MAT || op->src[0] == nullptr || op->src[1] == nullptr) {
            continue;
        }
        const ggml_tensor * weight = op->src[0];
        if (weight->type != GGML_TYPE_Q4_0 && weight->type != GGML_TYPE_Q4_1 && weight->type != GGML_TYPE_Q4_K) {
            continue;
        }
        const int64_t k = weight->ne[0];
        const int64_t n = weight->ne[1];
        const int64_t m = op->src[1]->ne[1] * op->src[1]->ne[2] * op->src[1]->ne[3];
        if (!spine_q4_stream_shape(m, k, n)) {
            continue;
        }
        if (!q4_pp_stream_weights_enabled(weight)) {
            continue;
        }
        if (weight->type == GGML_TYPE_Q4_K && !spine_q4_m_split_enabled(weight)) {
            continue;
        }
        const size_t k_blocks = static_cast<size_t>(k / 32);
        if (2 * k_blocks * sizeof(spacemit_kernels::ime2::q4_pp_block_a) <= tcm_size) {
            return true;
        }
    }
    return false;
}

spine_q4_stream_host * spine_q4_stream_host_create() {
    auto * host = new spine_q4_stream_host;
    for (int id = 0; id < 2; ++id) {
        host->workers[id] = std::thread([host, id]() {
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(id, &mask);
            if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) != 0) {
                host->failed.store(true, std::memory_order_release);
            }
            host->ready[id].store(true, std::memory_order_release);
            uint64_t observed = 0;
            while (!host->stop.load(std::memory_order_acquire)) {
                const uint64_t generation = host->generation.load(std::memory_order_acquire);
                if (generation != observed) {
                    host->run(host->arg, id);
                    observed = generation;
                    host->completed[id].store(generation, std::memory_order_release);
                } else {
                    __asm__ volatile("pause" ::: "memory");
                }
            }
        });
    }
    while (!host->ready[0].load(std::memory_order_acquire) ||
           !host->ready[1].load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    if (host->failed.load(std::memory_order_acquire)) {
        spine_q4_stream_host_destroy(host);
        return nullptr;
    }
    return host;
}

void spine_q4_stream_host_destroy(spine_q4_stream_host * host) {
    if (host == nullptr) {
        return;
    }
    host->stop.store(true, std::memory_order_release);
    for (auto & worker : host->workers) {
        worker.join();
    }
    delete host;
}

static void spine_q4_stream_host_submit(spine_q4_stream_host * host, void (*run)(void *, int), void * arg) {
    host->run = run;
    host->arg = arg;
    host->generation.fetch_add(1, std::memory_order_release);
}

static void spine_q4_stream_host_wait(spine_q4_stream_host * host) {
    const uint64_t generation = host->generation.load(std::memory_order_acquire);
    for (int id = 0; id < 2; ++id) {
        while (host->completed[id].load(std::memory_order_acquire) != generation) {
            __asm__ volatile("pause" ::: "memory");
        }
    }
}

struct spine_q40_stream_block {
    uint16_t d[32];
    uint8_t  qs[512];
};

static_assert(sizeof(spine_q40_stream_block) == 576);

struct spine_q41_stream_block {
    uint16_t d[32];
    uint8_t  zp[32];
    uint8_t  qs[512];
};

static_assert(sizeof(spine_q41_stream_block) == 608);

inline void spine_q4_stream_unpack_quarter(const uint8_t * src, const uint8_t * zp4, int quarter, int8_t * dst) {
    const size_t vl64 = __riscv_vsetvl_e64m2(8);
    GGML_ASSERT(vl64 == 8);
    const auto halves = __riscv_vlseg2e64_v_u64m2x2(reinterpret_cast<const uint64_t *>(src), vl64);
    const size_t vl = __riscv_vsetvl_e8m2(64);
    GGML_ASSERT(vl == 64);
    vuint16m4_t zp;
    if (zp4 != nullptr) {
        const auto packed_zp = __riscv_vle8_v_u8m2(zp4 + (quarter & 1) * 8, 8);
        const auto zp_index = __riscv_vsrl_vx_u8m2(__riscv_vid_v_u8m2(vl), 3, vl);
        const auto zp_byte = __riscv_vrgather_vv_u8m2(packed_zp, zp_index, vl);
        const auto zp_nibble = quarter < 2 ? __riscv_vand_vx_u8m2(zp_byte, 15, vl)
                                             : __riscv_vsrl_vx_u8m2(zp_byte, 4, vl);
        zp = __riscv_vzext_vf2_u16m4(zp_nibble, vl);
    } else {
        zp = __riscv_vmv_v_x_u16m4(8, vl);
    }
    const auto store_part = [&](vuint64m2_t words, int part) {
        const auto bytes = __riscv_vreinterpret_v_u64m2_u8m2(words);
        const auto wide = __riscv_vzext_vf2_u16m4(bytes, vl);
        const auto low = __riscv_vsub_vv_u16m4(__riscv_vand_vx_u16m4(wide, 15, vl), zp, vl);
        const auto high = __riscv_vsub_vv_u16m4(__riscv_vsrl_vx_u16m4(wide, 4, vl), zp, vl);
        const auto pair = __riscv_vor_vv_u16m4(
            __riscv_vand_vx_u16m4(low, 255, vl), __riscv_vsll_vx_u16m4(high, 8, vl), vl);
        __riscv_vse16_v_u16m4(reinterpret_cast<uint16_t *>(dst + part * 128), pair, vl);
    };
    store_part(__riscv_vget_v_u64m2x2_u64m2(halves, 0), 0);
    store_part(__riscv_vget_v_u64m2x2_u64m2(halves, 1), 1);
}

inline void spine_q4_stream_unpack_block(const uint8_t * qs, const uint8_t * zp4, int8_t * dst) {
    for (int q = 0; q < 4; ++q) {
        spine_q4_stream_unpack_quarter(qs + q * 128, zp4, q, dst + q * 256);
    }
}

template <typename BLOC_TYPE>
inline void spine_q4_stream_pack_tile(const uint8_t * compressed, int64_t tile, int64_t k_blocks,
                                      spacemit_kernels::ime2::q4_pp_block_b * dst,
                                      const uint16_t * precomputed_scales) {
    if constexpr (std::is_same_v<BLOC_TYPE, block_q4_K>) {
        using super_block = spacemit_kernels::nrow_super_q4_k_ds<32>;
        const auto * src = reinterpret_cast<const super_block *>(compressed) + tile * (k_blocks / 8);
        for (int64_t b = 0; b < k_blocks; ++b) {
            const auto & super = src[b / 8];
            const auto & sub = super.sub[b % 8];
            std::memcpy(dst[b].d, precomputed_scales + (tile * k_blocks + b) * 32, sizeof(dst[b].d));
            spine_q4_stream_unpack_block(sub.qs, sub.zp4, dst[b].qs);
        }
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_1>) {
        const auto * src = reinterpret_cast<const spine_q41_stream_block *>(compressed) + tile * k_blocks;
        for (int64_t b = 0; b < k_blocks; ++b) {
            std::memcpy(dst[b].d, src[b].d, sizeof(dst[b].d));
            uint8_t zp4[16];
            for (size_t col = 0; col < 16; ++col) {
                zp4[col] = (src[b].zp[col] & 15) | (src[b].zp[col + 16] << 4);
            }
            spine_q4_stream_unpack_block(src[b].qs, zp4, dst[b].qs);
        }
    } else {
        const auto * src = reinterpret_cast<const spine_q40_stream_block *>(compressed) + tile * k_blocks;
        for (int64_t b = 0; b < k_blocks; ++b) {
            std::memcpy(dst[b].d, src[b].d, sizeof(dst[b].d));
            spine_q4_stream_unpack_block(src[b].qs, nullptr, dst[b].qs);
        }
    }
}

template <typename BLOC_TYPE>
inline void spine_q4_stream_produce(spine_q4_stream_state & state, std::atomic<int64_t> * slots,
                                   uint32_t ring_slots, const uint8_t * compressed,
                                   spacemit_kernels::ime2::q4_pp_block_b * ring, int64_t k_blocks, int64_t n_tiles,
                                   int id, const uint16_t * precomputed_scales) {
    for (int64_t tile = id; tile < n_tiles; tile += 2) {
        const int slot = tile % ring_slots;
        if (tile >= ring_slots) {
            const int64_t previous = tile - ring_slots;
            for (int reader = 0; reader < 8; ++reader) {
                while (state.consumed[reader].tile.load(std::memory_order_acquire) < previous) {
                    __asm__ volatile("pause" ::: "memory");
                }
            }
        }
        spine_q4_stream_pack_tile<BLOC_TYPE>(compressed, tile, k_blocks, ring + slot * k_blocks,
                                             precomputed_scales);
        slots[slot].store(tile, std::memory_order_release);
    }
}

template <typename Fn>
inline bool spine_dyn_n_schedule(ggml::spacemit::context & ctx, spine_dyn_n_schedule_state & state, int64_t n_tiles,
                                 int64_t ith, int64_t nth, int64_t dyn_groups, int64_t claim_gran,
                                 Fn && process_tile) {
    constexpr int64_t max_groups = sizeof(state.counters) / sizeof(state.counters[0]);
    if (n_tiles <= 0 || dyn_groups <= 0 || dyn_groups > max_groups || dyn_groups > nth ||
        claim_gran <= 0) {
        return false;
    }

    if (ith == 0) {
        for (int64_t g = 0; g < dyn_groups; ++g) {
            __atomic_store_n(&state.counters[g], int64_t{0}, __ATOMIC_RELAXED);
        }
    }
    ctx.sync();

    const int64_t group = std::min<int64_t>(dyn_groups - 1, (ith * dyn_groups) / nth);
    const int64_t group_begin = (n_tiles * group) / dyn_groups;
    const int64_t group_end = (n_tiles * (group + 1)) / dyn_groups;
    const int64_t group_tiles = group_end - group_begin;
    const int64_t gran = std::max<int64_t>(1, std::min(claim_gran, group_tiles));

    for (;;) {
        const int64_t first = __atomic_fetch_add(&state.counters[group], gran, __ATOMIC_RELAXED);
        if (first >= group_tiles) {
            break;
        }
        const int64_t last = std::min(first + gran, group_tiles);
        for (int64_t tile = first; tile < last; ++tile) {
            process_tile(group_begin + tile);
        }
    }
    return true;
}

template <typename BLOC_TYPE> constexpr bool block_type_has_zp() {
    if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0> ||
                  std::is_same_v<BLOC_TYPE, block_q3_K> || std::is_same_v<BLOC_TYPE, block_q4_0> ||
                  std::is_same_v<BLOC_TYPE, block_mxfp4> || std::is_same_v<BLOC_TYPE, block_q5_0>) {
        return false;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K> ||
                         std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q5_1> ||
                         std::is_same_v<BLOC_TYPE, block_q5_K>) {
        return true;
    } else {
        assert(false);
        return false;
    }
}

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
class tensor_traits : public ggml::spacemit::tensor_traits_base {
    bool work_size(int /* n_threads */, uint32_t ring_slots, const ggml_tensor * op, size_t & size) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K>) {
                        size = spacemit_kernels::div_round_up(src1_nelements, size_t(32)) *
                               spacemit_kernels::q8k_blk_size(32);
                    } else if constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, true);
                    } else if constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, true, true);
                    } else {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, sizeof(int64_t));
                    const int64_t gemm_m = op->src[1]->ne[1] * op->src[1]->ne[2] * op->src[1]->ne[3];
                    if (gemm_m == 1 || gemm_m == 4) {
                        size += sizeof(spine_dyn_n_schedule_state);
                    }

                    if constexpr ((std::is_same_v<BLOC_TYPE, block_q4_K> && INTER_SIZE == 32 && NB_COLS == 32) ||
                                  (std::is_same_v<BLOC_TYPE, block_q4_1> && INTER_SIZE == 32 && NB_COLS == 32) ||
                                  (std::is_same_v<BLOC_TYPE, block_q4_0> && NB_COLS == 32 &&
                                   (INTER_SIZE == 32 || INTER_SIZE == 256))) {
                        const int64_t gemm_k = op->src[0]->ne[0];
                        if (q4_pp_stream_weights_enabled(op->src[0]) &&
                            spine_q4_stream_shape(gemm_m, gemm_k, op->src[0]->ne[1])) {
                            const size_t ring_bytes = ring_slots * static_cast<size_t>(gemm_k / 32) *
                                                      sizeof(spacemit_kernels::ime2::q4_pp_block_b);
                            size = GGML_PAD(size, size_t{ 64 }) + sizeof(spine_q4_stream_state) +
                                   ring_slots * sizeof(std::atomic<int64_t>);
                            size = GGML_PAD(size, size_t{ 64 });
                            size += ring_bytes + 64;
                        }
                    }

                    return true;
                }
            case GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K>) {
                        size = spacemit_kernels::div_round_up(src1_nelements, size_t(32)) *
                               spacemit_kernels::q8k_blk_size(32);
                    } else if constexpr (INTER_SIZE == QK4_0) {
                        size = spacemit_kernels::div_round_up(src1_nelements, QK4_0) *
                               spacemit_kernels::q8_blk_size(QK4_0, true);
                    } else if constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, true, true);
                    } else {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, sizeof(int64_t));

                    const int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    const int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    const size_t sizeof_mmid_row_mapping = sizeof(int64_t);
                    size += sizeof_mmid_row_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * sizeof(int64_t);

                    size = GGML_PAD(size, sizeof(int64_t));

                    return true;
                }
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool compute_forward(ggml::spacemit::context & ctx, ggml_tensor * op) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                switch (op->src[0]->type) {
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        return forward_mul_mat(ctx, op);
                    default:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        return false;
                }
                break;
            case GGML_OP_MUL_MAT_ID:
                switch (op->src[0]->type) {
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                        //case GGML_TYPE_MXFP4:
                        return forward_mul_mat_id(ctx, op);
                    default:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        return false;
                }
                break;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool forward_mul_mat(ggml::spacemit::context & ctx, ggml_tensor * op) const {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        int ith = ctx.ith;
        int nth = ctx.nth;

        [[maybe_unused]] const enum ggml_type type = src0->type;

        void *        w_data  = (void *) src0->data;
        const float * feature = (const float *) src1->data;
        float *       output  = (float *) dst->data;

        const int64_t gemm_m = ne11 * ne12 * ne13;
        const int64_t gemm_k = ne10;
        const int64_t gemm_n = ne01;
        bool q4_m_split = false;
        if constexpr (std::is_same_v<BLOC_TYPE, block_q4_K> && INTER_SIZE == 32 && NB_COLS == 32) {
            q4_m_split = spine_q4_m_split_enabled(src0);
        }

        spacemit_kernels::quantize_a_row_def       quantize_a_row_i8;
        spacemit_kernels::quantize_a_row_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        bool                                       set_kernel_impl = false;

        int64_t block_stride_a = spacemit_kernels::q8_blk_size(a_blk_len);

#if defined(RISCV64_SPACEMIT_IME2)
        if (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);

            if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8q6;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8q6;
                block_stride_a     = spacemit_kernels::q8k_blk_size(32);
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i6k;
                set_kernel_impl    = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                if constexpr (INTER_SIZE == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    block_stride_a     = spacemit_kernels::q8_hp_blk_size(a_blk_len, true, true);
                    set_kernel_impl    = true;
                } else {
                    // Q4_1 keeps the standard per-block scale layout; ZPDS expects Q4_K's shared-scale layout.
                    gemm_kernel = std::is_same_v<BLOC_TYPE, block_q4_1>
                                      ? spacemit_kernels::ime2::gemm_kernel_i8i4_legacy
                                      : spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);
                    set_kernel_impl    = true;
                }
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                set_kernel_impl = true;
            }
        }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
        if (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8  = spacemit_kernels::ime1::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = true;
            }
        }
#endif
        if (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        const int64_t a_k_blks = spacemit_kernels::div_round_up(gemm_k, a_blk_len);
        const int64_t b_k_blks = spacemit_kernels::div_round_up(gemm_k, b_blk_len);

        const int64_t row_stride_a        = a_k_blks * block_stride_a;
        const int64_t gemm_workspace_size = GGML_PAD(gemm_m * row_stride_a, alignof(int64_t));
        const size_t dyn_state_offset =
            GGML_PAD(static_cast<size_t>(gemm_workspace_size), alignof(spine_dyn_n_schedule_state));

        if (ith == 0 && ctx.workspace_size < gemm_workspace_size) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(ctx.workspace);

        void *        tcm_buffer      = ctx.shared.data;
        const int64_t tcm_buffer_size = ctx.shared.size;

#if defined(RISCV64_SPACEMIT_IME2)
        if constexpr ((std::is_same_v<BLOC_TYPE, block_q4_K> && INTER_SIZE == 32 && NB_COLS == 32) ||
                      (std::is_same_v<BLOC_TYPE, block_q4_1> && INTER_SIZE == 32 && NB_COLS == 32) ||
                      (std::is_same_v<BLOC_TYPE, block_q4_0> && NB_COLS == 32 &&
                       (INTER_SIZE == 32 || INTER_SIZE == 256))) {
            const bool expanded_weights = [&]() -> bool {
                if constexpr (std::is_same_v<BLOC_TYPE, block_q4_K>) {
                    return q4_m_split && !q4_pp_stream_weights_enabled(src0);
                }
                return false;
            }();
            const size_t expanded_offset = expanded_weights ? spine_q4_k_expanded_offset(src0) : 0;
            const int64_t k_blocks = gemm_k / 32;
            const bool m8_shape = [&]() {
                if constexpr (std::is_same_v<BLOC_TYPE, block_q4_K>) {
                    return gemm_k % QK_K == 0;
                } else {
                    // The Q4_0 K256 HP path is faster for PP8; M8 serves K32-only layouts.
                    return INTER_SIZE == 32 && gemm_k % QK4_0 == 0;
                }
            }();
            const size_t  m8_a_tile_bytes =
                static_cast<size_t>(k_blocks) * sizeof(spacemit_kernels::ime2::q4_pp_block_a);
            // K3 A100 PP8: use M8 only for the small shapes that beat the legacy path.
            if (expanded_weights && m8_shape && gemm_m == 8 && gemm_n >= 512 && gemm_n <= 1024 && gemm_k <= 1024 &&
                gemm_n % 32 == 0 && tcm_buffer != nullptr &&
                reinterpret_cast<uintptr_t>(tcm_buffer) % 64 == 0 &&
                m8_a_tile_bytes <= static_cast<size_t>(tcm_buffer_size)) {
                const auto * expanded_b = reinterpret_cast<const spacemit_kernels::ime2::q4_pp_block_b *>(
                    static_cast<const uint8_t *>(src0->data) + expanded_offset);
                auto * packed_a = static_cast<spacemit_kernels::ime2::q4_pp_block_a *>(tcm_buffer);
                spacemit_kernels::ime2::quantize_a_m8_i8(feature, gemm_k, packed_a);
                for (int64_t ni = static_cast<int64_t>(ith) * 32; ni < gemm_n; ni += 32 * nth) {
                    const auto * b_tile = expanded_b + (ni / 32) * k_blocks;
                    spacemit_kernels::ime2::gemm_kernel_i8i8_m8_q4_prepacked(packed_a, b_tile, output + ni, k_blocks, gemm_n);
                }
                return true;
            }
            const size_t a_tile_bytes =
                static_cast<size_t>(2 * k_blocks) * sizeof(spacemit_kernels::ime2::q4_pp_block_a);
            const size_t packed_b_bytes = static_cast<size_t>(k_blocks) * sizeof(spacemit_kernels::ime2::q4_pp_block_b);
            if (q4_pp_stream_weights_enabled(src0) && spine_q4_stream_shape(gemm_m, gemm_k, gemm_n) && nth == 8 &&
                global_spine_env_info.use_ime2 && tcm_buffer != nullptr &&
                a_tile_bytes <= static_cast<size_t>(tcm_buffer_size) && ctx.q4_stream_host != nullptr) {
                const size_t state_offset = GGML_PAD(static_cast<size_t>(gemm_workspace_size), size_t{ 64 });
                auto *       state =
                    reinterpret_cast<spine_q4_stream_state *>(static_cast<uint8_t *>(ctx.workspace) + state_offset);
                auto * slots = reinterpret_cast<std::atomic<int64_t> *>(state + 1);
                auto * ring  = reinterpret_cast<spacemit_kernels::ime2::q4_pp_block_b *>(
                    GGML_PAD(reinterpret_cast<uintptr_t>(slots + ctx.ring_slots), uintptr_t{ 64 }));
                const size_t ring_bytes = ctx.ring_slots * packed_b_bytes;
                GGML_ASSERT(reinterpret_cast<uintptr_t>(ring) + ring_bytes <=
                            reinterpret_cast<uintptr_t>(ctx.workspace) + ctx.workspace_size);
                if (ith == 0) {
                    new (state) spine_q4_stream_state();
                    for (uint32_t slot = 0; slot < ctx.ring_slots; ++slot) {
                        new (slots + slot) std::atomic<int64_t>(-1);
                    }
                    for (int reader = 0; reader < 8; ++reader) {
                        state->consumed[reader].tile.store(-1, std::memory_order_relaxed);
                    }
                }
                ctx.sync();
                auto * host = static_cast<spine_q4_stream_host *>(ctx.q4_stream_host);
                struct job {
                    spine_q4_stream_state * state;
                    std::atomic<int64_t> * slots;
                    uint32_t ring_slots;
                    const uint8_t * compressed;
                    spacemit_kernels::ime2::q4_pp_block_b * ring;
                    int64_t k_blocks;
                    int64_t n_tiles;
                    const uint16_t * precomputed_scales;
                } producer_job{};
                if (ith == 0) {
                    const uint16_t * precomputed_scales = nullptr;
                    if constexpr (std::is_same_v<BLOC_TYPE, block_q4_K>) {
                        precomputed_scales = reinterpret_cast<const uint16_t *>(
                            static_cast<const uint8_t *>(src0->data) + spine_q4_k_expanded_offset(src0));
                        GGML_ASSERT(precomputed_scales != nullptr);
                    }
                    producer_job = { state, slots, ctx.ring_slots, static_cast<const uint8_t *>(src0->data), ring,
                                     k_blocks, gemm_n / 32, precomputed_scales };
                    spine_q4_stream_host_submit(host, [](void * arg, int id) {
                        const auto & job = *static_cast<const struct job *>(arg);
                        spine_q4_stream_produce<BLOC_TYPE>(*job.state, job.slots, job.ring_slots, job.compressed,
                                                            job.ring, job.k_blocks, job.n_tiles, id,
                                                            job.precomputed_scales);
                    }, &producer_job);
                }
                auto * tcm_a = static_cast<spacemit_kernels::ime2::q4_pp_block_a *>(tcm_buffer);
                const int64_t mi = static_cast<int64_t>(ith) * 16;
                spacemit_kernels::ime2::quantize_a_m8_i8(feature + mi * gemm_k, gemm_k, tcm_a);
                spacemit_kernels::ime2::quantize_a_m8_i8(feature + (mi + 8) * gemm_k, gemm_k,
                                                           tcm_a + k_blocks);
                for (int64_t tile = 0; tile < gemm_n / 32; ++tile) {
                    const int slot = tile % ctx.ring_slots;
                    while (slots[slot].load(std::memory_order_acquire) != tile) {
                        __asm__ volatile("pause" ::: "memory");
                    }
                    const auto * b_tile = ring + slot * k_blocks;
                    spacemit_kernels::ime2::gemm_kernel_i8i8_m8_q4_prepacked(
                        tcm_a, b_tile, output + mi * gemm_n + tile * 32, k_blocks, gemm_n);
                    spacemit_kernels::ime2::gemm_kernel_i8i8_m8_q4_prepacked(
                        tcm_a + k_blocks, b_tile, output + (mi + 8) * gemm_n + tile * 32, k_blocks, gemm_n);
                    state->consumed[ith].tile.store(tile, std::memory_order_release);
                }
                if (ith == 0) {
                    spine_q4_stream_host_wait(host);
                }
                // The graph loop synchronizes all workers after this operator.
                return true;
            }
            if (expanded_weights && gemm_m >= 16 && gemm_m % 16 == 0 && gemm_n % 32 == 0 && gemm_k % QK_K == 0 &&
                tcm_buffer != nullptr && reinterpret_cast<uintptr_t>(tcm_buffer) % 64 == 0 &&
                a_tile_bytes <= static_cast<size_t>(tcm_buffer_size)) {
                const auto * expanded_b = reinterpret_cast<const spacemit_kernels::ime2::q4_pp_block_b *>(
                    static_cast<const uint8_t *>(src0->data) + expanded_offset);
                auto * packed_a = static_cast<spacemit_kernels::ime2::q4_pp_block_a *>(tcm_buffer);
                // Split output N32 tiles across cores when M16 tiles alone cannot fill the chip.
                if (gemm_m / 16 < nth && gemm_n >= 32 * nth) {
                    for (int64_t mi = 0; mi < gemm_m; mi += 16) {
                        spacemit_kernels::ime2::quantize_a_m8_i8(feature + mi * gemm_k, gemm_k, packed_a);
                        spacemit_kernels::ime2::quantize_a_m8_i8(
                            feature + (mi + 8) * gemm_k, gemm_k, packed_a + k_blocks);
                        for (int64_t ni = static_cast<int64_t>(ith) * 32; ni < gemm_n; ni += 32 * nth) {
                            const auto * b_tile = expanded_b + (ni / 32) * k_blocks;
                            spacemit_kernels::ime2::gemm_kernel_i8i8_m8_q4_prepacked(
                                packed_a, b_tile, output + mi * gemm_n + ni, k_blocks, gemm_n);
                            spacemit_kernels::ime2::gemm_kernel_i8i8_m8_q4_prepacked(
                                packed_a + k_blocks, b_tile, output + (mi + 8) * gemm_n + ni, k_blocks, gemm_n);
                        }
                    }
                    return true;
                }
                for (int64_t mi = static_cast<int64_t>(ith) * 16; mi < gemm_m; mi += 16 * nth) {
                    spacemit_kernels::ime2::quantize_a_m8_i8(feature + mi * gemm_k, gemm_k, packed_a);
                    spacemit_kernels::ime2::quantize_a_m8_i8(feature + (mi + 8) * gemm_k, gemm_k, packed_a + k_blocks);
                    for (int64_t ni = 0; ni < gemm_n; ni += 32) {
                        const auto * b_tile = expanded_b + (ni / 32) * k_blocks;
                        spacemit_kernels::ime2::gemm_kernel_i8i8_m8_q4_prepacked(packed_a, b_tile, output + mi * gemm_n + ni,
                                                                        k_blocks, gemm_n);
                        spacemit_kernels::ime2::gemm_kernel_i8i8_m8_q4_prepacked(
                            packed_a + k_blocks, b_tile, output + (mi + 8) * gemm_n + ni, k_blocks, gemm_n);
                    }
                }
                return true;
            }
        }
        if constexpr (std::is_same_v<BLOC_TYPE, block_q8_0> && INTER_SIZE == 32 && NB_COLS == 32) {
            // Q8_0 already has one N32/K32 block with FP16 scales followed by
            // row-major int8 values. The M8 kernel rearranges values in registers.
            const int64_t k_blocks = gemm_k / 32;
            const auto * packed_b = reinterpret_cast<const spacemit_kernels::ime2::q8_0_block_b *>(src0->data);
            auto * packed_a = static_cast<spacemit_kernels::ime2::q4_pp_block_a *>(tcm_buffer);
            const size_t m8_a_tile_bytes =
                static_cast<size_t>(k_blocks) * sizeof(spacemit_kernels::ime2::q4_pp_block_a);
            if (gemm_m == 8 && gemm_n >= 256 && gemm_n % 32 == 0 && gemm_k % QK8_0 == 0 &&
                tcm_buffer != nullptr && reinterpret_cast<uintptr_t>(tcm_buffer) % 64 == 0 &&
                m8_a_tile_bytes <= static_cast<size_t>(tcm_buffer_size)) {
                spacemit_kernels::ime2::quantize_a_m8_i8(feature, gemm_k, packed_a);
                for (int64_t ni = static_cast<int64_t>(ith) * 32; ni < gemm_n; ni += 32 * nth) {
                    const auto * b_tile = packed_b + (ni / 32) * k_blocks;
                    spacemit_kernels::ime2::gemm_kernel_i8i8_m8_q8_0(
                        packed_a, b_tile, output + ni, k_blocks, gemm_n);
                }
                return true;
            }

            const size_t m16_a_tile_bytes = 2 * m8_a_tile_bytes;
            if (gemm_m >= 16 && gemm_m % 16 == 0 && gemm_n % 32 == 0 && gemm_k % QK8_0 == 0 &&
                tcm_buffer != nullptr && reinterpret_cast<uintptr_t>(tcm_buffer) % 64 == 0 &&
                m16_a_tile_bytes <= static_cast<size_t>(tcm_buffer_size)) {
                const auto quantize_m16 = [&](int64_t mi) {
                    spacemit_kernels::ime2::quantize_a_m8_i8(feature + mi * gemm_k, gemm_k, packed_a);
                    spacemit_kernels::ime2::quantize_a_m8_i8(
                        feature + (mi + 8) * gemm_k, gemm_k, packed_a + k_blocks);
                };
                const auto compute_n32 = [&](int64_t mi, int64_t ni) {
                    const auto * b_tile = packed_b + (ni / 32) * k_blocks;
                    spacemit_kernels::ime2::gemm_kernel_i8i8_m8_q8_0(
                        packed_a, b_tile, output + mi * gemm_n + ni, k_blocks, gemm_n);
                    spacemit_kernels::ime2::gemm_kernel_i8i8_m8_q8_0(
                        packed_a + k_blocks, b_tile, output + (mi + 8) * gemm_n + ni, k_blocks, gemm_n);
                };
                if (gemm_m / 16 < nth && gemm_n >= 32 * nth) {
                    for (int64_t mi = 0; mi < gemm_m; mi += 16) {
                        quantize_m16(mi);
                        for (int64_t ni = static_cast<int64_t>(ith) * 32; ni < gemm_n; ni += 32 * nth) {
                            compute_n32(mi, ni);
                        }
                    }
                } else {
                    for (int64_t mi = static_cast<int64_t>(ith) * 16; mi < gemm_m; mi += 16 * nth) {
                        quantize_m16(mi);
                        for (int64_t ni = 0; ni < gemm_n; ni += 32) {
                            compute_n32(mi, ni);
                        }
                    }
                }
                return true;
            }
        }
#endif

        auto * quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t row_align = 4;
        const int64_t     row_blks  = spacemit_kernels::div_round_up(gemm_m, row_align);

        const int64_t row_stride_b      = spine_row_stride_b<BLOC_TYPE, INTER_SIZE, NB_COLS>(b_k_blks);
        const int64_t per_mb_rows_wsize = row_align * row_stride_a;
        const int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        const int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_count);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        if (gemm_m == 1) {
            int task_per_thread = spacemit_kernels::div_round_up(a_k_blks, nth);
            int a_blk_start     = ith * task_per_thread;
            int a_blk_end       = std::min(a_blk_start + task_per_thread, (int) a_k_blks);
            if (a_blk_start < a_blk_end) {
                quantize_a_row_i8(a_blk_len, feature + a_blk_start * a_blk_len, (a_blk_end - a_blk_start) * a_blk_len,
                                  quant_a_buffer + a_blk_start * block_stride_a);
            }
        } else {
            int task_per_thread = spacemit_kernels::div_round_up(row_blks, nth);
            int m_row_blk_start = ith * task_per_thread;
            int m_row_blk_end   = std::min(m_row_blk_start + task_per_thread, (int) row_blks);
            for (int m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
                int m_idx             = m_row_blk * row_align;
                int rows_tobe_handled = (gemm_m - m_idx) > row_align ? row_align : (gemm_m - m_idx);

                if (rows_tobe_handled == row_align && quantize_a_4row_i8 != nullptr) {
                    const float * a_row_ptr       = feature + m_idx * gemm_k;
                    auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                } else {
                    while (rows_tobe_handled) {
                        const float * a_row_ptr       = feature + m_idx * gemm_k;
                        auto *        quant_a_row_ptr = quant_a_buffer + m_idx * row_stride_a;
                        quantize_a_row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                        rows_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }

        ctx.sync();

        const int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        const int64_t gemm_m_blocked    = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
        const int64_t max_gemm_n_stride = spacemit_kernels::div_round_up(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        if (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_round_up(max_gemm_n_stride, NB_COLS) * NB_COLS);
        }

        if (gemm_n_stride == gemm_n && tcm_buffer != nullptr && per_mb_rows_wsize <= tcm_buffer_size) {
            for (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth) {
                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data);
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t m_row_real = std::min(gemm_m - m_start, row_align);

                spacemit_kernels::rvv::memcpy1d(tcm_buffer, quant_a_buffer + m_start * row_stride_a,
                                                m_row_real * row_stride_a);

                int64_t n_blk_real = 0;
                for (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_COLS);

                    uint8_t * a_row_ptr = (uint8_t *) tcm_buffer;
                    float *   c_blk     = output + m_start * gemm_n + ni;

                    int32_t rows_remaining = m_row_real;

                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row_ptr, b_col, b_col_zp, c_blk, rows_remaining,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_ptr += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        } else if (tcm_buffer != nullptr && per_nb_cols_wsize <= tcm_buffer_size) {
            uint8_t * a_row = quant_a_buffer;
            uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
            if ((gemm_workspace_size + per_nb_cols_wsize) <= tcm_buffer_size) {
                a_row = (uint8_t *) tcm_buffer;
                b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + gemm_workspace_size;
            }
            uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

            // Disabled: 2x-wide Q6 M1 tiles reduce thread parallelism and regress TG.
            // if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K>) {
            //     if (gemm_m == 1) {
            //         uint8_t * a_dir = quant_a_buffer;
            //         if (gemm_workspace_size <= tcm_buffer_size) {
            //             a_dir = reinterpret_cast<uint8_t *>(tcm_buffer);
            //             spacemit_kernels::rvv::memcpy1d(a_dir, quant_a_buffer, gemm_workspace_size);
            //         }

            //         constexpr int64_t q6_tile_cols = 2 * NB_COLS;
            //         for (int64_t ni = static_cast<int64_t>(ith) * q6_tile_cols; ni < gemm_n;
            //              ni += q6_tile_cols * nth) {
            //             const int64_t nb_real = std::min(gemm_n - ni, q6_tile_cols);
            //             uint8_t * b_dir = reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b;
            //             gemm_kernel(b_blk_len, a_dir, b_dir, nullptr, output + ni, 1, nb_real, b_k_blks, gemm_n);
            //         }
            //         return true;
            //     }
            // }

            // Direct IME-from-DRAM GEMV path, decode only (gemm_m == 1), q4_0 only:
            // q8_0 always loses to the staged TCM path here.
            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> && INTER_SIZE == 256 && NB_COLS == 32) {
                if (gemm_m == 1 && a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                    constexpr int64_t tile_cols = 4 * NB_COLS;
                    for (int64_t ni = (int64_t) ith * tile_cols; ni < gemm_n; ni += tile_cols * nth) {
                        const int64_t nb_real  = std::min(gemm_n - ni, tile_cols);
                        uint8_t *     b_row    = reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b;
                        uint8_t *     b_row_zp = block_type_has_zp<BLOC_TYPE>() ? b_row : nullptr;
                        gemm_kernel(b_blk_len, a_row, b_row, b_row_zp, output + ni, 1, nb_real, b_k_blks, gemm_n);
                    }
                    return true;
                }
            }

            // Generic M1/M4: single direct tile.
            if (gemm_m == 1 || gemm_m == 4) {
                uint8_t * a_dir = quant_a_buffer;
                if (gemm_workspace_size <= tcm_buffer_size) {
                    a_dir = reinterpret_cast<uint8_t *>(tcm_buffer);
                    spacemit_kernels::rvv::memcpy1d(a_dir, quant_a_buffer, gemm_workspace_size);
                }

                struct dyn_n_config {
                    int64_t groups;
                    int64_t claim_gran;
                };
                const dyn_n_config m1_config{2, 1};
                const dyn_n_config m4_config{2, 1};
                const dyn_n_config & dyn_config = gemm_m == 1 ? m1_config : m4_config;
                const int64_t n_tiles = spacemit_kernels::div_round_up(gemm_n, static_cast<int64_t>(NB_COLS));
                if (ith == 0 && ctx.workspace_size < dyn_state_offset + sizeof(spine_dyn_n_schedule_state)) {
                    GGML_ABORT("wsize less than dynamic N scheduler state");
                }
                auto * dyn_state = reinterpret_cast<spine_dyn_n_schedule_state *>(ws_ptr + dyn_state_offset);
                const bool dyn_scheduled = spine_dyn_n_schedule(
                    ctx, *dyn_state, n_tiles, static_cast<int64_t>(ith), static_cast<int64_t>(nth), dyn_config.groups,
                    dyn_config.claim_gran,
                    [&](int64_t tile) {
                        const int64_t ni = tile * NB_COLS;
                        const int64_t nb_real = std::min(gemm_n - ni, static_cast<int64_t>(NB_COLS));
                        uint8_t * b_dir = reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b;
                        uint8_t * b_dir_zp = block_type_has_zp<BLOC_TYPE>() ? b_dir : nullptr;
                        gemm_kernel(
                            b_blk_len, a_dir, b_dir, b_dir_zp, output + ni, gemm_m, nb_real, b_k_blks, gemm_n);
                    });
                if (dyn_scheduled) {
                    return true;
                }

                for (int64_t ni = static_cast<int64_t>(ith) * NB_COLS; ni < gemm_n; ni += NB_COLS * nth) {
                    const int64_t nb_real = std::min(gemm_n - ni, static_cast<int64_t>(NB_COLS));
                    uint8_t * b_dir       = reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b;
                    uint8_t * b_dir_zp    = block_type_has_zp<BLOC_TYPE>() ? b_dir : nullptr;
                    gemm_kernel(b_blk_len, a_dir, b_dir, b_dir_zp, output + ni, gemm_m, nb_real, b_k_blks, gemm_n);
                }
                return true;
            }

            const bool has_pair = ((ith & 1) != 0) || (ith + 1 < nth);

            const int64_t ni0     = (int64_t) ith * NB_COLS;
            const bool    active0 = ni0 < gemm_n;
            const int64_t nb0     = active0 ? std::min(gemm_n - ni0, (int64_t) NB_COLS) : 0;

            if (ith % 2 == 0 && active0) {
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni0 * row_stride_b,
                                                nb0 * row_stride_b);
                if (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
            }

            if (has_pair) {
                spine_barrier_wait(cur_barrier);
            }

            if (ith % 2 != 0 && active0) {
                if (a_row != quant_a_buffer) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a_buffer, gemm_workspace_size);
                }
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni0 * row_stride_b,
                                                nb0 * row_stride_b);
            }

            const int64_t base_start = (int64_t) (ith & ~1) * NB_COLS;
            const int64_t lane_off   = (int64_t) (ith & 1) * NB_COLS;
            for (int64_t base = base_start; base < gemm_n; base += NB_COLS * nth) {
                const int64_t ni      = base + lane_off;
                const bool    active  = ni < gemm_n;
                const int64_t nb_real = active ? std::min(gemm_n - ni, (int64_t) NB_COLS) : 0;

                if (has_pair && ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                if (active) {
                    int64_t rows_remaining = gemm_m;
                    float * c_blk          = output + ni;
                    auto *  a_row_cur      = a_row;

                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row_cur, b_col, b_col_zp, c_blk, rows_remaining,
                                                        nb_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_cur += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }

                if (has_pair && ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                const int64_t next_ni = ni + NB_COLS * nth;
                if (next_ni < gemm_n) {
                    const int64_t next_nb = std::min(gemm_n - next_ni, (int64_t) NB_COLS);
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + next_ni * row_stride_b,
                                                    next_nb * row_stride_b);
                }
            }
        } else {
            const int64_t task_count_m = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
            const int64_t task_count_n = spacemit_kernels::div_round_up(gemm_n, gemm_n_stride);

            int64_t task_count      = task_count_m * task_count_n;
            int64_t task_per_thread = (task_count + nth - 1) / nth;
            int64_t start           = ith * task_per_thread;
            int64_t end             = std::min((ith + 1) * task_per_thread, task_count);
            for (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                const auto tid_n = compute_idx / task_count_m;
                const auto tid_m = compute_idx % task_count_m;

                const int64_t m_start = tid_m * gemm_m_stride;
                const int64_t m_count = std::min(gemm_m - m_start, (int64_t) gemm_m_stride);

                const int64_t n_start = tid_n * gemm_n_stride;
                const int64_t n_count = std::min(gemm_n - n_start, (int64_t) gemm_n_stride);

                const int64_t n_blk = m_count == 1 ? n_count : NB_COLS;

                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data) + n_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t n_blk_real = 0;
                for (int64_t ni = 0; ni < n_count; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(n_count - ni, n_blk);

                    uint8_t * a_row = quant_a_buffer + m_start * row_stride_a;

                    float * c_blk = output + m_start * gemm_n + n_start + ni;

                    int64_t rows_remaining = m_count;

                    uint8_t * b_col_cur    = b_col;
                    uint8_t * b_col_zp_cur = b_col_zp;

                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row, b_col_cur, b_col_zp_cur, c_blk,
                                                        rows_remaining, n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        }
        return true;
    }

    bool forward_mul_mat_id(ggml::spacemit::context & ctx, ggml_tensor * op) const {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        int ith = ctx.ith;
        int nth = ctx.nth;

        // row groups
        const int n_ids = ids->ne[0];  // n_expert_used
        const int n_as  = ne02;        // n_expert

        struct mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_row_def           quantize_a_row_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        bool                                           set_kernel_impl = false;
        size_t                                         block_stride_a  = spacemit_kernels::q8_blk_size(QK4_0);

#if defined(RISCV64_SPACEMIT_IME2)
        if (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8;
            block_stride_a    = spacemit_kernels::q8_blk_size(QK4_0, true);

            if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8q6;
                block_stride_a    = spacemit_kernels::q8k_blk_size(32);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i6k;
                set_kernel_impl   = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                if constexpr (INTER_SIZE == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    block_stride_a    = spacemit_kernels::q8_hp_blk_size(a_blk_len, true, true);
                    set_kernel_impl   = true;
                } else {
                    gemm_kernel = std::is_same_v<BLOC_TYPE, block_q4_1>
                                      ? spacemit_kernels::ime2::gemm_kernel_i8i4_legacy
                                      : spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);
                    set_kernel_impl    = true;
                }
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl   = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl   = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                set_kernel_impl    = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                set_kernel_impl    = true;
            }
        }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
        if (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8 = spacemit_kernels::ime1::quantize_a_row_i8;

            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = true;
            }
        }
#endif
        if (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        const size_t a_k_blks = spacemit_kernels::div_round_up(ne10, a_blk_len);
        const size_t b_k_blks = spacemit_kernels::div_round_up(ne10, b_blk_len);

        const size_t nbw1                = a_k_blks * block_stride_a;
        const size_t nbw2                = ne11 * nbw1;
        const size_t nbw3                = nbw2 * ne12;
        const size_t gemm_workspace_size = GGML_PAD(nbw3, alignof(int64_t));

        const uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(ctx.workspace);
        auto *          quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        if (ne11 == 1) {
            for (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_row_i8(a_blk_len, (float *) ((char *) src1->data + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_buffer + i12 * nbw2 + ak_blk_id * block_stride_a);
            }
        } else {
            for (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_row_i8(a_blk_len, (float *) ((char *) src1->data + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_buffer + i12 * nbw2 + i11 * nbw1);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) *ne12 + (i1)]

        int64_t *          matrix_row_counts       = (int64_t *) (ws_ptr + gemm_workspace_size);
        int32_t *          valid_ep_count          = (int32_t *) (matrix_row_counts + n_as);
        int32_t *          valid_act_count         = (int32_t *) (valid_ep_count + 1);
        int64_t *          valid_matrix_row_counts = (int64_t *) (valid_act_count + 1);
        mmid_row_mapping * matrix_rows             = (mmid_row_mapping *) (valid_matrix_row_counts + n_as);

        if (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * sizeof(int64_t));

            // group rows by src0 matrix
            for (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                for (int32_t id = 0; id < n_ids; ++id) {
                    const int32_t i02 =
                        *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }

            int32_t valid_ep_count_t  = 0;
            int32_t valid_act_count_t = 0;
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                valid_matrix_row_counts[valid_ep_count_t] = cur_a;
                valid_act_count_t += cne1;
                valid_ep_count_t += 1;
            }
            valid_ep_count[0]  = valid_ep_count_t;
            valid_act_count[0] = valid_act_count_t;
        }

        ctx.sync();

        const size_t row_stride_b      = static_cast<size_t>(spine_row_stride_b<BLOC_TYPE, INTER_SIZE, NB_COLS>(b_k_blks));
        const size_t expert_b_stride   = ne01 * row_stride_b;
        const size_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        const int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_count);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        std::array<const uint8_t *, 2> src_workspaces;
        std::array<float *, 2>         dst_workspaces;

        auto *     tcm_buffer      = ctx.shared.data;
        const auto tcm_buffer_size = ctx.shared.size;

        const auto valid_ep_count_t  = valid_ep_count[0];
        const auto valid_act_count_t = valid_act_count[0];

        int nth_es = 1;
        int nth_n  = nth;

        int ith_es = ith % nth_es;
        int ith_n  = (ith / nth_es) % nth_n;

        if (valid_ep_count_t % nth == 0 && tcm_buffer != nullptr && valid_ep_count_t == n_as &&
            valid_act_count_t == n_as && per_nb_cols_wsize <= tcm_buffer_size) {
            const bool has_pair = ((ith & 1) != 0) || (ith + 1 < nth);

            for (int64_t valid_id = ith; valid_id < valid_ep_count_t; valid_id += nth) {
                const int64_t cur_a = valid_matrix_row_counts[valid_id];

                auto * src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride;

                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, 0);
                const int        id          = row_mapping.i1;
                const int64_t    i11         = id % ne11;
                const int64_t    i12         = row_mapping.i2;
                const int64_t    i1          = id;
                const int64_t    i2          = i12;

                auto *  src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                float * c_blk    = (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2));

                uint8_t * a_row = src1_col;
                uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
                if ((nbw1 + per_nb_cols_wsize) <= tcm_buffer_size) {
                    a_row = (uint8_t *) tcm_buffer;
                    b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + nbw1;
                }
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                if (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);

                    if (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }
                }

                if (has_pair) {
                    spine_barrier_wait(cur_barrier);
                }

                if (ith % 2 != 0) {
                    if (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_COLS);
                for (int64_t ni = 0; ni < ne01; ni += NB_COLS) {
                    if (has_pair && ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }
                    gemm_kernel(b_blk_len, a_row, b_col, b_col_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    if (has_pair && ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    const int64_t next_ni = ni + NB_COLS;
                    if (next_ni < ne01) {
                        nb_real = std::min(ne01 - next_ni, NB_COLS);
                        spacemit_kernels::rvv::memcpy1d(
                            b_col, reinterpret_cast<uint8_t *>(src0_cur) + next_ni * row_stride_b, per_nb_cols_wsize);
                    }
                }
            }
        } else {
            for (int64_t valid_id = ith_es; valid_id < valid_ep_count_t; valid_id += nth_es) {
                const int64_t cur_a = valid_matrix_row_counts[valid_id];
                const int64_t cne1  = matrix_row_counts[cur_a];

                int64_t src1_cur_start = 0;
                int64_t src1_cur_end   = cne1;

                int64_t src0_cur_start = (ith_n * ne01) / nth_n;
                int64_t src0_cur_end   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                if (src1_cur_start >= src1_cur_end || src0_cur_start >= src0_cur_end) {
                    continue;
                }

                src0_cur_start =
                    (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                src0_cur_end =
                    (src0_cur_end % NB_COLS) ? src0_cur_end + NB_COLS - (src0_cur_end % NB_COLS) : src0_cur_end;

                auto *    src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride + src0_cur_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;

                size_t extra_tcm_buffer_size = tcm_buffer_size;
                void * extra_tcm_buffer      = tcm_buffer;
                if (tcm_buffer != nullptr && (src1_cur_end - src1_cur_start) >= 4 &&
                    (src0_cur_end - src0_cur_start) * row_stride_b <= tcm_buffer_size) {
                    spacemit_kernels::rvv::memcpy1d(tcm_buffer, src0_cur,
                                                    (src0_cur_end - src0_cur_start) * row_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_buffer);
                    b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;
                    extra_tcm_buffer_size -= (src0_cur_end - src0_cur_start) * row_stride_b;
                    extra_tcm_buffer = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(tcm_buffer) +
                                                                (src0_cur_end - src0_cur_start) * row_stride_b);
                }

                int ir1 = src1_cur_start;

                if (extra_tcm_buffer_size >= nbw1 && extra_tcm_buffer != nullptr) {
                    int64_t quant_a_tile_size = extra_tcm_buffer_size / nbw1;
                    do {
                        quant_a_tile_size = MIN(quant_a_tile_size, src1_cur_end - ir1);

                        uint8_t * quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);

                        int iir1 = ir1;
                        for (; iir1 < (ir1 + quant_a_tile_size); ++iir1) {
                            mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1);

                            const int id = row_mapping.i1;  // selected expert index

                            const int64_t i11 = id % ne11;
                            const int64_t i12 = row_mapping.i2;  // row index in src1

                            auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_buffer, src1_col, nbw1);
                            quant_a_tile_buffer = quant_a_tile_buffer + nbw1;
                        }

                        quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);
                        iir1                = ir1;

                        if (moe_gemm_kernel_m2 != nullptr) {
                            for (; iir1 < (ir1 + quant_a_tile_size - 1); iir1 += 2, quant_a_tile_buffer += 2 * nbw1) {
                                mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);
                                mmid_row_mapping row_mapping_1 = MMID_MATRIX_ROW(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_buffer;
                                src_workspaces[1] = quant_a_tile_buffer + nbw1;

                                dst_workspaces[0] =
                                    (float *) ((char *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start;
                                dst_workspaces[1] = (float *) ((char *) dst->data +
                                                               ((row_mapping_1.i1) * nb1 + (row_mapping_1.i2) * nb2)) +
                                                    src0_cur_start;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                                   dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks,
                                                   ne01);
                            }
                        }

                        for (; iir1 < (ir1 + quant_a_tile_size); iir1++, quant_a_tile_buffer += nbw1) {
                            mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_buffer, src0_cur, b_col_zp,
                                (float *) ((char *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start,
                                1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_size;
                    } while (ir1 < src1_cur_end);
                } else {
                    if (moe_gemm_kernel_m2 != nullptr) {
                        for (; ir1 < src1_cur_end - 1; ir1 += 2) {
                            for (int iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1 + iir1);

                                const int id = row_mapping.i1;  // selected expert index

                                const int64_t i11 = id % ne11;
                                const int64_t i12 = row_mapping.i2;  // row index in src1

                                const int64_t i1 = id;               // selected expert index
                                const int64_t i2 = i12;              // row

                                src_workspaces[iir1] = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                               dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }
                    }

                    for (; ir1 < src1_cur_end; ir1++) {
                        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        const int id = row_mapping.i1;  // selected expert index

                        const int64_t i11 = id % ne11;
                        const int64_t i12 = row_mapping.i2;  // row index in src1

                        const int64_t i1 = id;               // selected expert index
                        const int64_t i2 = i12;              // row

                        auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_col, src0_cur, b_col_zp,
                                    (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, 1,
                                    src0_cur_end - src0_cur_start, b_k_blks, ne01);
                    }
                }
            }
        }
#undef MMID_MATRIX_ROW
        return true;
    }

    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (int) NB_COLS, (int) INTER_SIZE);
        return ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

class tensor_traits_common : public ggml::spacemit::tensor_traits_base {
    bool work_size(int n_threads, uint32_t /* ring_slots */, const ggml_tensor * op, size_t & size) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0] && op->src[1] && op->src[1]->type == GGML_TYPE_F32) {
                    const auto * traits = ggml_get_type_traits_cpu(op->src[0]->type);
                    if (traits && traits->vec_dot) {
                    const size_t row_size = ggml_row_size(traits->vec_dot_type, op->src[1]->ne[0]);
                    const size_t bytes = row_size * op->src[1]->ne[1] * op->src[1]->ne[2] * op->src[1]->ne[3];
                        if (op->src[1]->type != traits->vec_dot_type) {
                            size = bytes;
                            return true;
                        }
                    }
                }
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                {
                    const int     n_tasks = n_threads;
                    const int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    const int64_t DK      = op->src[1]->ne[0];
                    const int64_t DV      = op->src[2]->ne[0];  // DV

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    size_t prefill = sizeof(float) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      GGML_FA_TILE_KV * DV + GGML_FA_TILE_KV * DK +
                                      ggml::spacemit::cache_line_size_f32) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = sizeof(float) * n_tasks * (DK + 2 * DV + 64);

                    size = MAX(prefill, decode);
                }
                return true;
            default:
                break;
        }
        return false;
    }

    bool compute_forward(ggml::spacemit::context & ctx, ggml_tensor * op) const override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->type == GGML_TYPE_F16 &&
                    op->src[1]->type == GGML_TYPE_F32 &&
                    op->src[0]->nb[0] == sizeof(_Float16) &&
                    op->src[1]->nb[0] == sizeof(float)) {
                    spacemit_kernels::rvv::forward_mul_mat_f16_f32(ctx, op);
                    return true;
                }
                if (op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_get_type_traits_cpu(op->src[0]->type)->vec_dot != nullptr) {
                    spacemit_kernels::scalar::forward_mul_mat(ctx, op);
                    return true;
                }
                return false;
            case GGML_OP_NORM:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::scalar::forward_norm_f16(ctx, op);
                        return true;
                    default:
                        GGML_ABORT("fatal error");
                }
            case GGML_OP_RMS_NORM:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::scalar::forward_rms_norm_f16(ctx, op);
                        return true;
                    default:
                        GGML_ABORT("fatal error");
                }
            case GGML_OP_ADD:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_ADD, float>(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_ADD, _Float16>(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_SUB:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_SUB, float>(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_SUB, _Float16>(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_MUL:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_MUL, float>(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_MUL, _Float16>(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_DIV:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_DIV, float>(ctx, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_DIV, _Float16>(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_SCALE:
                if (op->src[0]->type == GGML_TYPE_F32) {
                    spacemit_kernels::rvv::forward_scale_f32(ctx, op);
                    return true;
                }
                return false;
            case GGML_OP_UNARY:
                switch (ggml_get_unary_op(op)) {
                    case GGML_UNARY_OP_TANH:
                        spacemit_kernels::rvv::forward_unary_tanh_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_GELU:
                        spacemit_kernels::rvv::forward_unary_gelu_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_SILU:
                        spacemit_kernels::rvv::forward_unary_silu_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_EXP:
                        spacemit_kernels::rvv::forward_unary_exp_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_SIGMOID:
                        spacemit_kernels::rvv::forward_unary_sigmoid_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_NEG:
                        spacemit_kernels::rvv::forward_unary_neg_f32(ctx, op);
                        return true;
                    case GGML_UNARY_OP_SOFTPLUS:
                        spacemit_kernels::rvv::forward_unary_softplus_f32(ctx, op);
                        return true;
                    default:
                        return false;
                }
            case GGML_OP_SOFT_MAX:
                if (op->src[0]->type == GGML_TYPE_F32) {
                    spacemit_kernels::rvv::forward_soft_max_f32(ctx, op);
                    return true;
                }
                return false;
            case GGML_OP_GLU:
                if (op->src[0]->type == GGML_TYPE_F32) {
                    switch (ggml_get_glu_op(op)) {
                        case GGML_GLU_OP_GEGLU:
                            spacemit_kernels::rvv::forward_glu_geglu_f32(ctx, op);
                            return true;
                        case GGML_GLU_OP_SWIGLU:
                            spacemit_kernels::rvv::forward_glu_swiglu_f32(ctx, op);
                            return true;
                        default:
                            break;
                    }
                }
                return false;
            case GGML_OP_FLASH_ATTN_EXT:
                // Use the RVV implementation for its supported F32-Q/F16-KV
                // shapes; retain the CPU-equivalent implementation for all
                // other layouts and dtypes.
                if (forward_flash_attn_ext_f16(ctx, op)) {
                    return true;
                }
                spacemit_kernels::scalar::forward_flash_attn_ext(ctx, op);
                return true;
            case GGML_OP_ROPE:
                spacemit_kernels::rvv::forward_rope(ctx, op);
                return true;
            case GGML_OP_CONT:
                {
                    const ggml_tensor * src0 = op->src[0];
                    if (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_permute(ctx, op);
                        return true;
                    }
                    if (op->type == src0->type && ggml_is_contiguous(op) &&
                        ggml_nelements(op) == ggml_nelements(src0) &&
                        (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16)) {
                        if (op->type == GGML_TYPE_F32) {
                            spacemit_kernels::rvv::forward_cont_general<float>(ctx, op);
                        } else {
                            spacemit_kernels::rvv::forward_cont_general<_Float16>(ctx, op);
                        }
                        return true;
                    }
                    return false;
                }
            case GGML_OP_CPY:
                {
                    const ggml_tensor * src0 = op->src[0];
                    if (ggml_nelements(op) == 0) {
                        return true;
                    }
                    if (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_permute(ctx, op);
                        return true;
                    }
                    if (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_strided_f32(ctx, op);
                        return true;
                    }
                    if (ggml_is_contiguous(op) && ggml_is_contiguous(src0) &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        if (op->type == GGML_TYPE_F16 && src0->type == GGML_TYPE_F32) {
                            spacemit_kernels::rvv::forward_cpy_f32_to_f16(ctx, op);
                            return true;
                        }
                        if (op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F16) {
                            spacemit_kernels::rvv::forward_cpy_f16_to_f32(ctx, op);
                            return true;
                        }
                    }
                    return false;
                }
            case GGML_OP_SET_ROWS:
                spacemit_kernels::rvv::forward_set_rows(ctx, op);
                return true;
            case GGML_OP_REPEAT:
                {
                    const bool rows_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    const bool broadcast_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    if (rows_equal && broadcast_or_equal) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(ctx, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(ctx, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    if (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(ctx, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(ctx, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    return false;
                }
                return true;
            case GGML_OP_SUM_ROWS:
                {
                    if (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_sum_rows<float>(ctx, op);
                    } else {
                        return false;
                    }
                }
                return true;
            case GGML_OP_GET_ROWS:
                {
                    if (op->src[0]->type == op->type) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_rows<int32_t>(ctx, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_rows<int16_t>(ctx, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    return false;
                }
                return true;
            case GGML_OP_CONCAT:
                {
                    const int32_t dim = ggml_get_op_params_i32(op, 0);
                    if (dim == 0 && op->type == op->src[0]->type) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(ctx, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(ctx, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    return false;
                }
                return true;
            // Scalar ops dispatched to scalar_kernels
            case GGML_OP_L2_NORM:
                spacemit_kernels::scalar::forward_l2_norm_f32(ctx, op);
                return true;
            case GGML_OP_FILL:
                spacemit_kernels::scalar::forward_fill_f32(ctx, op);
                return true;
            case GGML_OP_CUMSUM:
                spacemit_kernels::scalar::forward_cumsum_f32(ctx, op);
                return true;
            case GGML_OP_PAD:
                spacemit_kernels::scalar::forward_pad_f32(ctx, op);
                return true;
            case GGML_OP_TRI:
                spacemit_kernels::scalar::forward_tri_f32(ctx, op);
                return true;
            case GGML_OP_DIAG:
                spacemit_kernels::scalar::forward_diag_f32(ctx, op);
                return true;
            case GGML_OP_SET:
                spacemit_kernels::scalar::forward_set_f32(ctx, op);
                return true;
            case GGML_OP_SOLVE_TRI:
                spacemit_kernels::scalar::forward_solve_tri_f32(ctx, op);
                return true;
            case GGML_OP_GATED_DELTA_NET:
                spacemit_kernels::rvv::forward_gated_delta_net(ctx, op);
                return true;
            case GGML_OP_SSM_CONV:
                spacemit_kernels::scalar::forward_ssm_conv_f32(ctx, op);
                return true;
            default:
                break;
        }
        return false;
    }

    bool forward_flash_attn_ext_f16(ggml::spacemit::context & ctx, ggml_tensor * dst) const {
        const ggml_tensor * q = dst->src[0];
        const ggml_tensor * k = dst->src[1];
        const ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        const int64_t DK = nek0;
        const int64_t DV = nev0;

        const bool supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        const bool supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        const bool supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        const bool supported_vlen  = (__riscv_vlenb() == 128);

        if (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            return false;
        }

        // total rows in q
        const int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        const int ith = ctx.ith;
        const int nth = ctx.nth;

        static constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        const bool               use_tiled = neq1 >= Q_TILE_SZ;

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        int64_t nchunk = nth;

        // The number of elements in each chunk
        const int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        int current_chunk = ith;

        while (current_chunk < nchunk) {
            const int64_t ir0 = dr * current_chunk;
            const int64_t ir1 = MIN(ir0 + dr, nr);

            if (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    ctx, dst, ir0, ir1);
            } else {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    ctx, dst, ir0, ir1);
            }

            current_chunk += nth;
        }

        return true;
    }

    int repack(ggml_tensor * t, const void * data, size_t data_size) const override {
        memcpy(t->data, data, data_size);
        return 0;
    }
};

// Impl By IME1
static const tensor_traits<block_q4_0, 32, 16>  q4_0_16x32_q8_0;
static const tensor_traits<block_q4_1, 32, 16>  q4_1_16x32_q8_0;
static const tensor_traits<block_q4_K, 32, 16>  q4_k_16x32_q8_0;
// Impl By IME2
static const tensor_traits<block_q2_K, 256, 32> q2_k_32x256_q8_0;
static const tensor_traits<block_q3_K, 256, 32> q3_k_32x256_q8_0;
static const tensor_traits<block_q4_0, 32, 32>  q4_0_32x32_q8_0;
static const tensor_traits<block_q4_1, 32, 32>  q4_1_32x32_q8_0;
static const tensor_traits<block_q4_0, 256, 32> q4_0_32x256_q8_0;
static const tensor_traits<block_q4_1, 256, 32> q4_1_32x256_q8_0;
static const tensor_traits<block_q4_K, 32, 32>  q4_k_32x32_q8_0;
static const tensor_traits<block_q6_K, 32, 32>  q6_k_32x32_q8_0;
static const tensor_traits<block_q8_0, 32, 32>  q8_0_32x32_q8_0;
static const tensor_traits<block_mxfp4, 32, 32> mxfp4_32x32_q8_0;
static const tensor_traits<block_q5_K, 32, 32>  q5_k_32x32_q8_0;
static const tensor_traits<block_q5_1, 32, 32>  q5_1_32x32_q8_0;
static const tensor_traits<block_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
static const tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

const ggml::spacemit::tensor_traits_base * ggml_spacemit_get_optimal_repack_type(const ggml_tensor * cur) {
    switch (cur->type) {
        case GGML_TYPE_Q2_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q3_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_1:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q6_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q8_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_MXFP4:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#endif
            }
            break;
        case GGML_TYPE_Q5_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q5_1:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q5_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#endif
            }
            break;
        default:
            break;
    }

    return nullptr;
}

size_t ggml_spacemit_nbytes(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
    }

    GGML_UNUSED(buft);

    const auto plain_nbytes = [&]() {
        size_t total = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        return total;
    };

    const size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        return plain_nbytes();
    }

    const size_t row_nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;

    const auto add_strided_nbytes = [&](size_t total, size_t src_block_size, size_t dst_block_size) {
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * (tensor->nb[i] / src_block_size) * dst_block_size;
        }
        return total;
    };

    const auto remap_block_nbytes = [&](size_t src_block_size, size_t dst_block_size, int64_t padded_rows = 0) {
        GGML_ASSERT(row_nbytes % src_block_size == 0);

        size_t total =
            add_strided_nbytes((row_nbytes / src_block_size) * dst_block_size, src_block_size, dst_block_size);

        if (padded_rows > 0 && tensor->ne[1] % padded_rows != 0) {
            total += (padded_rows - tensor->ne[1] % padded_rows) * (tensor->nb[1] / src_block_size) * dst_block_size;
        }

        return total;
    };

    size_t nbytes = row_nbytes;
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            break;
        case GGML_TYPE_Q4_K:
            if (ggml::cpu::riscv64_spacemit::q4_pp_stream_weights_enabled(tensor) &&
                ggml::cpu::riscv64_spacemit::spine_q4_m_split_enabled(tensor)) {
                const size_t scale_blocks =
                    static_cast<size_t>(ggml_nrows(tensor)) / 32 * static_cast<size_t>(tensor->ne[0] / 32);
                nbytes = ggml::cpu::riscv64_spacemit::spine_q4_k_expanded_offset(tensor) +
                         scale_blocks * sizeof(uint16_t) * 32;
            } else if (ggml::cpu::riscv64_spacemit::spine_q4_m_split_enabled(tensor)) {
                nbytes =
                    ggml::cpu::riscv64_spacemit::spine_q4_k_expanded_offset(tensor) +
                    (static_cast<size_t>(ggml_nrows(tensor)) * static_cast<size_t>(tensor->ne[0] / QK_K) * 8 / 32) *
                        sizeof(spacemit_kernels::ime2::q4_pp_block_b);
            } else {
                nbytes = remap_block_nbytes(sizeof(block_q4_K), sizeof(block_q4_1) * 8);
            }
            break;
        case GGML_TYPE_Q6_K:
            nbytes = remap_block_nbytes(sizeof(block_q6_K), sizeof(spacemit_kernels::nrow_super_q6_k<1>), 32);
            break;
        case GGML_TYPE_Q8_0:
            nbytes = remap_block_nbytes(sizeof(block_q8_0), sizeof(block_q8_0), 32);
            break;
        case GGML_TYPE_Q2_K:
            nbytes = remap_block_nbytes(sizeof(block_q2_K), sizeof(spacemit_kernels::nrow_block_q2_k<1>));
            break;
        case GGML_TYPE_Q3_K:
            nbytes = remap_block_nbytes(sizeof(block_q3_K), sizeof(spacemit_kernels::nrow_block_q3_k<1>));
            break;
        case GGML_TYPE_MXFP4:
            nbytes = remap_block_nbytes(sizeof(block_mxfp4), sizeof(spacemit_kernels::nrow_block_mxfp4<1>));
            break;
        case GGML_TYPE_Q5_K:
            nbytes = remap_block_nbytes(sizeof(block_q5_K), sizeof(spacemit_kernels::nrow_block_q5_1<1>) * 8);
            break;
        case GGML_TYPE_Q5_1:
            nbytes = remap_block_nbytes(sizeof(block_q5_1), sizeof(spacemit_kernels::nrow_block_q5_1<1>));
            break;
        case GGML_TYPE_Q5_0:
            nbytes = remap_block_nbytes(sizeof(block_q5_0), sizeof(spacemit_kernels::nrow_block_q5_0<1>));
            break;
        default:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            break;
    }

    return nbytes;
}

const ggml::spacemit::tensor_traits_base * ggml_spacemit_get_tensor_traits(const ggml_tensor * op) {
    if (op == nullptr) {
        return nullptr;
    }

    const auto * common = &ggml::cpu::riscv64_spacemit::rvv_impl;
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            if (op->src[0] && op->src[1] && ggml_n_dims(op->src[0]) == 2 &&
                op->src[1]->type == GGML_TYPE_F32) {
                const auto * traits = static_cast<const ggml::spacemit::tensor_traits_base *>(op->src[0]->extra);
                if (traits) return traits;
                const auto * repack = ggml_spacemit_get_optimal_repack_type(op->src[0]);
                if (repack) return repack;
                // F16 dense weight: same condition as compute_forward
                if (op->src[0]->type == GGML_TYPE_F16 &&
                    op->src[0]->nb[0] == sizeof(_Float16) &&
                    op->src[1]->nb[0] == sizeof(float)) {
                    return common;
                }
            }
            break;
        case GGML_OP_MUL_MAT_ID:
            if (op->src[0] && op->src[1] && ggml_n_dims(op->src[0]) == 3 &&
                op->src[1]->type == GGML_TYPE_F32) {
                const auto * traits = static_cast<const ggml::spacemit::tensor_traits_base *>(op->src[0]->extra);
                return traits ? traits : ggml_spacemit_get_optimal_repack_type(op->src[0]);
            }
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            if (op->src[0] && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)) {
                return common;
            }
            break;
        case GGML_OP_SCALE:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32) {
                return common;
            }
            break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (op->src[0] && op->src[1] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == op->src[0]->type && op->type == op->src[0]->type &&
                op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) &&
                op->src[1]->nb[0] == ggml_type_size(op->src[1]->type) &&
                op->nb[0] == ggml_type_size(op->type) && ggml_can_repeat(op->src[1], op->src[0]) &&
                ggml_are_same_shape(op->src[0], op)) {
                return common;
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            if (op->src[0] && op->src[1] && op->src[2] &&
                (op->op_params[3] == GGML_PREC_F32 || op->op_params[3] == GGML_PREC_DEFAULT) &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) &&
                op->src[1]->ne[0] > 0 && op->src[2]->ne[0] > 0) {
                return common;
            }
            break;
        case GGML_OP_ROPE:
            if (op->src[0] && op->src[1]) {
                float freq_base;
                memcpy(&freq_base,  op->op_params + 5, sizeof(float));
                const int n_dims = ggml_get_op_params_i32(op, 1);
                const int mode   = ggml_get_op_params_i32(op, 2);
                const bool src2_ok = op->src[2] == nullptr ||
                    (op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[2]));
                const bool mode_ok = (mode == GGML_ROPE_TYPE_NORMAL ||
                                      mode == GGML_ROPE_TYPE_NEOX   ||
                                      mode == GGML_ROPE_TYPE_MROPE  ||
                                      mode == GGML_ROPE_TYPE_IMROPE);
                const bool c1 = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
                const bool c2 = op->type == op->src[0]->type;
                const bool c3 = op->src[1]->type == GGML_TYPE_I32;
                const bool c4 = src2_ok;
                const bool c5 = op->src[0]->nb[0] == ggml_type_size(op->src[0]->type);
                const bool c6 = op->nb[0] == ggml_type_size(op->type);
                const bool c7 = freq_base > 0.0f;
                const bool c8 = n_dims > 0 && n_dims <= op->ne[0] && n_dims <= 512 && n_dims % 2 == 0;
                const bool c9 = mode_ok;
                if (c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8 && c9) {
                    return common;
                }
            }
            break;
        case GGML_OP_CONT:
            if (op->src[0] && op->type == op->src[0]->type) {
                // permute case
                if (op->nb[0] != op->src[0]->nb[0] && op->nb[0] == op->src[0]->nb[1] &&
                    op->ne[3] * op->ne[2] * op->nb[2] == op->src[0]->ne[3] * op->src[0]->ne[2] * op->src[0]->nb[2]) {
                    return common;
                }
                // general stride-aware copy for F32/F16
                if ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                    ggml_is_contiguous(op) &&
                    ggml_nelements(op) == ggml_nelements(op->src[0])) {
                    return common;
                }
            }
            break;
        case GGML_OP_CPY:
            if (op->src[0] && ggml_nelements(op) == 0) {
                return common;
            }
            if (op->src[0] && op->type == op->src[0]->type && op->nb[0] == op->src[0]->nb[1] &&
                op->src[0]->nb[0] != op->src[0]->nb[1] && ggml_nelements(op->src[0]) == ggml_nelements(op)) {
                return common;
            }
            if (op->src[0] && op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_nelements(op->src[0]) == ggml_nelements(op)) {
                return common;
            }
            if (op->src[0] &&
                ((op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F32) ||
                 (op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F16)) &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) &&
                ggml_nelements(op->src[0]) == ggml_nelements(op)) {
                return common;
            }
            break;
        case GGML_OP_SET_ROWS:
            if (op->src[0] && op->src[1] && op->src[2] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                (op->src[1]->type == GGML_TYPE_I32 || op->src[1]->type == GGML_TYPE_I64) &&
                (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[2]->type == op->type &&
                ggml_are_same_shape(op, op->src[2]) && op->src[0]->ne[1] == op->src[1]->ne[0] &&
                op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) &&
                op->src[2]->nb[0] == ggml_type_size(op->src[2]->type) && op->nb[0] == ggml_type_size(op->type) &&
                op->src[1]->nb[0] == ggml_type_size(op->src[1]->type) && op->ne[0] == op->src[0]->ne[0] &&
                op->ne[2] == op->src[0]->ne[2] && op->ne[3] == op->src[0]->ne[3] &&
                op->src[0]->ne[2] % op->src[1]->ne[1] == 0 && op->src[0]->ne[3] % op->src[1]->ne[2] == 0) {
                return common;
            }
            break;
        case GGML_OP_REPEAT:
            if (op->src[0] && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)) {
                const bool rows_equal = ggml_nrows(op->src[0]) == ggml_nrows(op);
                const bool n0_matches = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];
                if ((rows_equal && n0_matches) ||
                    (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0])) {
                    return common;
                }
            }
            break;
        case GGML_OP_SUM_ROWS:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                return common;
            }
            break;
        case GGML_OP_GET_ROWS:
            if (op->src[0] && op->src[1] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->type == op->src[0]->type &&
                op->src[1]->type == GGML_TYPE_I32 && op->ne[0] == op->src[0]->ne[0] &&
                op->src[0]->ne[2] == op->src[1]->ne[1] &&
                op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) &&
                ggml_nrows(op) == ggml_nelements(op->src[1])) {
                return common;
            }
            break;
        case GGML_OP_CONCAT:
            if (op->src[0] && op->src[1] && ggml_get_op_params_i32(op, 0) == 0 &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == op->src[0]->type && op->type == op->src[0]->type &&
                op->nb[0] == ggml_type_size(op->type) &&
                op->nb[1] == ggml_type_size(op->type) * (op->src[0]->ne[0] + op->src[1]->ne[0])) {
                return common;
            }
            break;
        case GGML_OP_UNARY:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) &&
                (ggml_get_unary_op(op) == GGML_UNARY_OP_TANH    ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_GELU    ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_SILU    ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_EXP     ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_SIGMOID ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_NEG     ||
                 ggml_get_unary_op(op) == GGML_UNARY_OP_SOFTPLUS)) {
                return common;
            }
            break;
        case GGML_OP_SOFT_MAX:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                op->src[0]->nb[0] == sizeof(float) && op->nb[0] == sizeof(float)) {
                return common;
            }
            break;
        case GGML_OP_GLU:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32) {
                if (ggml_get_glu_op(op) == GGML_GLU_OP_GEGLU) {
                    return common;
                }
                if (ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU && ggml_is_contiguous_1(op->src[0]) &&
                    op->type == GGML_TYPE_F32 && op->nb[0] == sizeof(float)) {
                    const int32_t swapped = ggml_get_op_params_i32(op, 1);
                    if (op->src[1]) {
                        if (swapped == 0 && op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous_1(op->src[1]) &&
                            ggml_are_same_shape(op->src[0], op->src[1]) && ggml_are_same_shape(op, op->src[0])) {
                            return common;
                        }
                    } else if (op->ne[0] == op->src[0]->ne[0] / 2 && ggml_nrows(op) == ggml_nrows(op->src[0])) {
                        return common;
                    }
                }
            }
            break;
        default:
            break;
    }

    // Scalar ops: route to common so compute_forward dispatches to scalar_kernels
    switch (op->op) {
        case GGML_OP_L2_NORM:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_FILL:
            if (op->src[0] && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16))
                return common;
            break;
        case GGML_OP_CUMSUM:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_PAD:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_TRI:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op->src[0]))
                return common;
            break;
        case GGML_OP_DIAG:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_SET:
            if (op->src[0] && op->src[1] &&
                op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_SOLVE_TRI:
            if (op->src[0] && op->src[1] &&
                op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_GATED_DELTA_NET:
            return common;
        case GGML_OP_SSM_CONV:
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32)
                return common;
            break;
        case GGML_OP_MUL_MAT:
            // Generic CPU dot fallback for dense or unsupported quantized
            // layouts which do not have a repacked Spacemit trait.
            if (op->src[0] && op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
                ggml_get_type_traits_cpu(op->src[0]->type)->vec_dot != nullptr) {
                return common;
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            if (op->src[0] && op->src[1] && op->src[2] &&
                (op->op_params[3] == GGML_PREC_F32 || op->op_params[3] == GGML_PREC_DEFAULT) &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) &&
                op->src[1]->ne[0] > 0 && op->src[2]->ne[0] > 0) {
                return common;
            }
            break;
        default:
            break;
    }

    return nullptr;
}

int ggml_riscv64_spacemit_repack_tensor(ggml_tensor * tensor, const void * data, size_t size) {
    auto traits = (ggml::spacemit::tensor_traits_base *) tensor->extra;
    if (traits) {
        return traits->repack(tensor, data, size);
    }
    memcpy(tensor->data, data, size);
    return 0;
}

bool ggml_spacemit_get_work_size(int n_threads, uint32_t ring_slots, const ggml_tensor * op, size_t * size) {
    const auto * traits = ggml_spacemit_get_tensor_traits(op);
    return traits != nullptr && traits->work_size(n_threads, ring_slots, op, *size);
}

bool ggml_spacemit_compute_forward(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const auto * traits = ggml_spacemit_get_tensor_traits(op);
    return traits != nullptr && traits->compute_forward(ctx, op);
}
