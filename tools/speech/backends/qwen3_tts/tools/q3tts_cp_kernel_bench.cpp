#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <vector>

#include "spacemit/ime_kernels.h"
#include "spacemit/rvv_kernels.h"

extern "C" {
void ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(int thread_n);
void ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(int thread_n);
}

namespace {

struct block_q4_0x32_layout {
    _Float16 d[32];
    uint8_t  qs[16 * 32];
};

static uint32_t xorshift32(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void fill_f32(std::vector<float> & data) {
    uint32_t state = 0x12345678u;
    for (float & v : data) {
        const int x = (int) (xorshift32(state) & 0x3ffu) - 512;
        v = (float) x / 256.0f;
    }
}

static void fill_q4_repacked(std::vector<uint8_t> & data, int64_t k_blks, int64_t n) {
    constexpr int64_t k_subblks_per_superblk = 8;
    const int64_t b_superblk_stride = (int64_t) sizeof(block_q4_0x32_layout) * k_subblks_per_superblk;
    const int64_t b_tile_stride = k_blks * b_superblk_stride;
    uint32_t state = 0x87654321u;

    for (int64_t ni = 0; ni < n; ni += 32) {
        uint8_t * tile = data.data() + (ni / 32) * b_tile_stride;
        for (int64_t kb = 0; kb < k_blks; ++kb) {
            uint8_t * superblk = tile + kb * b_superblk_stride;
            auto * blocks = reinterpret_cast<block_q4_0x32_layout *>(superblk);
            for (int s = 0; s < k_subblks_per_superblk; ++s) {
                for (int i = 0; i < 32; ++i) {
                    blocks[s].d[i] = (_Float16) 0.02f;
                }
                for (uint8_t & q : blocks[s].qs) {
                    q = (uint8_t) xorshift32(state);
                }
            }
        }
    }
}

static double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static float checksum(const std::vector<float> & values) {
    float sum = 0.0f;
    for (float v : values) {
        sum += v * 0.000001f;
    }
    return sum;
}

struct shape_buffers {
    static constexpr int64_t block_len = 256;

    int64_t k = 0;
    int64_t n = 0;
    int64_t k_blks = 0;
    size_t a_stride = 0;
    size_t b_superblk_stride = 0;
    size_t b_tile_stride = 0;
    std::vector<float> src;
    std::vector<uint8_t> a;
    std::vector<uint8_t> b;
    std::vector<float> out;
};

static shape_buffers make_shape(int64_t k, int64_t n) {
    shape_buffers shape;
    shape.k = k;
    shape.n = n;
    shape.k_blks = spacemit_kernels::div_round_up(k, shape_buffers::block_len);
    shape.a_stride = spacemit_kernels::q8_hp_blk_size(shape_buffers::block_len, true, true);
    shape.b_superblk_stride = sizeof(block_q4_0x32_layout) * 8;
    shape.b_tile_stride = (size_t) shape.k_blks * shape.b_superblk_stride;

    const size_t a_bytes = (size_t) shape.k_blks * shape.a_stride;
    const size_t b_bytes = (size_t) spacemit_kernels::div_round_up(n, int64_t(32)) * shape.b_tile_stride;

    shape.src.resize((size_t) k);
    shape.a.resize(a_bytes);
    shape.b.resize(b_bytes);
    shape.out.resize((size_t) n);
    fill_f32(shape.src);
    fill_q4_repacked(shape.b, shape.k_blks, n);
    spacemit_kernels::rvv::quantize_a_row_i8_hp(shape_buffers::block_len, shape.src.data(), (size_t) k,
                                                shape.a.data());
    return shape;
}

static bool env_enabled(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0;
}

static void run_shape_single(shape_buffers & shape, int iters) {
    for (int i = 0; i < 50; ++i) {
        spacemit_kernels::ime2::gemm_kernel_i8i4_hp(shape_buffers::block_len, shape.a.data(), shape.b.data(), nullptr,
                                                    shape.out.data(), 1, (size_t) shape.n, (size_t) shape.k_blks,
                                                    (size_t) shape.n);
    }

    const double t0 = now_s();
    for (int i = 0; i < iters; ++i) {
        spacemit_kernels::ime2::gemm_kernel_i8i4_hp(shape_buffers::block_len, shape.a.data(), shape.b.data(), nullptr,
                                                    shape.out.data(), 1, (size_t) shape.n, (size_t) shape.k_blks,
                                                    (size_t) shape.n);
    }
    const double t1 = now_s();
    const double us = (t1 - t0) * 1000000.0 / (double) iters;
    std::printf("single   m=1 k=%lld n=%lld k_blks=%lld iters=%d time_us=%.3f checksum=%g\n", (long long) shape.k,
                (long long) shape.n, (long long) shape.k_blks, iters, us, checksum(shape.out));
}

struct dispatch_context {
    shape_buffers *   shape = nullptr;
    pthread_barrier_t start;
    pthread_barrier_t done;
    int               nth = 4;
    int               iters = 0;
    int               warmup = 50;
    int64_t           tile_cols = 32;
};

struct worker_context {
    dispatch_context * dispatch = nullptr;
    int                ith = 0;
};

static void run_dispatch_tiles(dispatch_context * dispatch, int ith) {
    shape_buffers & shape = *dispatch->shape;
    for (int64_t ni = (int64_t) ith * dispatch->tile_cols; ni < shape.n; ni += dispatch->tile_cols * dispatch->nth) {
        const int64_t nb_real = std::min(shape.n - ni, dispatch->tile_cols);
        uint8_t *     b_row = shape.b.data() + (size_t) (ni / 32) * shape.b_tile_stride;
        float *       c_blk = shape.out.data() + ni;
        spacemit_kernels::ime2::gemm_kernel_i8i4_hp(shape_buffers::block_len, shape.a.data(), b_row, nullptr, c_blk, 1,
                                                    (size_t) nb_real, (size_t) shape.k_blks, (size_t) shape.n);
    }
}

static void * dispatch_worker(void * arg) {
    worker_context *  worker = static_cast<worker_context *>(arg);
    dispatch_context * dispatch = worker->dispatch;

    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(worker->ith);
    for (int i = 0; i < dispatch->warmup + dispatch->iters; ++i) {
        pthread_barrier_wait(&dispatch->start);
        run_dispatch_tiles(dispatch, worker->ith);
        pthread_barrier_wait(&dispatch->done);
    }
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(worker->ith);
    return nullptr;
}

static void run_shape_dispatch(shape_buffers & shape, int iters) {
    dispatch_context dispatch;
    dispatch.shape = &shape;
    dispatch.iters = iters;
    dispatch.tile_cols = env_enabled("SPACEMIT_Q4_HP_M1_N64") ? 64 : 32;

    pthread_barrier_init(&dispatch.start, nullptr, (unsigned) dispatch.nth + 1);
    pthread_barrier_init(&dispatch.done, nullptr, (unsigned) dispatch.nth + 1);

    std::vector<pthread_t> threads((size_t) dispatch.nth);
    std::vector<worker_context> workers((size_t) dispatch.nth);
    for (int ith = 0; ith < dispatch.nth; ++ith) {
        workers[(size_t) ith].dispatch = &dispatch;
        workers[(size_t) ith].ith = ith;
        pthread_create(&threads[(size_t) ith], nullptr, dispatch_worker, &workers[(size_t) ith]);
    }

    for (int i = 0; i < dispatch.warmup; ++i) {
        pthread_barrier_wait(&dispatch.start);
        pthread_barrier_wait(&dispatch.done);
    }

    const double t0 = now_s();
    for (int i = 0; i < iters; ++i) {
        pthread_barrier_wait(&dispatch.start);
        pthread_barrier_wait(&dispatch.done);
    }
    const double t1 = now_s();

    for (pthread_t thread : threads) {
        pthread_join(thread, nullptr);
    }

    pthread_barrier_destroy(&dispatch.start);
    pthread_barrier_destroy(&dispatch.done);

    const double us = (t1 - t0) * 1000000.0 / (double) iters;
    std::printf("dispatch4 m=1 k=%lld n=%lld tile_cols=%lld iters=%d time_us=%.3f checksum=%g\n",
                (long long) shape.k, (long long) shape.n, (long long) dispatch.tile_cols, iters, us,
                checksum(shape.out));
}

static void run_shape(int64_t k, int64_t n, int iters, bool dispatch) {
    shape_buffers shape = make_shape(k, n);
    if (dispatch) {
        run_shape_dispatch(shape, iters);
    } else {
        run_shape_single(shape, iters);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    int iters = 2000;
    if (argc > 1) {
        iters = std::atoi(argv[1]);
        if (iters <= 0) {
            iters = 2000;
        }
    }
    const bool dispatch = argc > 2 && std::strcmp(argv[2], "dispatch") == 0;

    if (!dispatch) {
        ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(0);
    }
    run_shape(1024, 1024, iters, dispatch);
    run_shape(2048, 1024, iters, dispatch);
    run_shape(3072, 1024, iters, dispatch);
    run_shape(1024, 4096, iters, dispatch);
    run_shape(1024, 6144, iters, dispatch);
    run_shape(3072, 1024, iters, dispatch);
    if (!dispatch) {
        ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(0);
    }
    return 0;
}
