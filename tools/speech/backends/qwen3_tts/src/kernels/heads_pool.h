// X100 4-thread fp32 GEMV pool for the 15 cp lm-heads.
// (Production slice of cp_forward.h; in-process cp forward experiments live in ref/.)
#pragma once
#include <pthread.h>
#include <riscv_vector.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef Q3TTS_HEADS_NTH
#define Q3TTS_HEADS_NTH 4
#endif
#define NTH Q3TTS_HEADS_NTH
typedef struct { const uint16_t *wf, *xf; int fo, fi; float *y; int t0, t1; } Job;
static Job hd_job[NTH];
static pthread_t hd_th[NTH];
static int hd_core_base = 0;
static int hd_idle_yield = 0;
static int hd_main_work = 0;
static int hd_best_idx[NTH];
static float hd_best_val[NTH];
static volatile int hd_go[NTH], hd_done[NTH], hd_quit = 0;

// w is fp16 (halves DDR traffic), x is f32 pre-narrowed by caller to fp16; widening MAC keeps f32 accum.
static inline float dot_f16(const uint16_t *w, const uint16_t *x, int n) {
    size_t vl = __riscv_vsetvl_e16m4(128);
    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.f, vl);
    for (int d = 0; d < n; d += (int)vl) {
        acc = __riscv_vfwmacc_vv_f32m8(acc, __riscv_vle16_v_f16m4((const _Float16 *)(w + d), vl),
                                       __riscv_vle16_v_f16m4((const _Float16 *)(x + d), vl), vl);
    }
    size_t vl32 = __riscv_vsetvl_e32m8(vl);
    vfloat32m1_t r = __riscv_vfredusum_vs_f32m8_f32m1(acc, __riscv_vfmv_v_f_f32m1(0.f, 1), vl32);
    return __riscv_vfmv_f_s_f32m1_f32(r);
}

static void *hd_worker(void *a) {
    long id = (long)a;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(hd_core_base + id, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    for (;;) {
        while (!__atomic_load_n(&hd_go[id], __ATOMIC_ACQUIRE)) {
            if (hd_quit) return NULL;
            if (hd_idle_yield) {
                sched_yield();
            } else {
                __asm__ volatile("pause");
            }
        }
        __atomic_store_n(&hd_go[id], 0, __ATOMIC_RELAXED);
        Job *j = &hd_job[id];
        if (j->y) {
            for (int r = j->t0; r < j->t1; r++) {
                __builtin_prefetch(j->wf + (long)(r + 1) * j->fi);
                j->y[r] = dot_f16(j->wf + (long)r * j->fi, j->xf, j->fi);
            }
        } else {
            float best = -1e30f;
            int best_i = j->t0;
            for (int r = j->t0; r < j->t1; r++) {
                __builtin_prefetch(j->wf + (long)(r + 1) * j->fi);
                float v = dot_f16(j->wf + (long)r * j->fi, j->xf, j->fi);
                if (v > best) {
                    best = v;
                    best_i = r;
                }
            }
            hd_best_val[id] = best;
            hd_best_idx[id] = best_i;
        }
        __atomic_store_n(&hd_done[id], 1, __ATOMIC_RELEASE);
    }
}
static void pools_init(void) {
    const char *base = getenv("Q3TTS_HEADS_BASE");
    if (base) {
        hd_core_base = atoi(base);
    }
    hd_idle_yield = getenv("Q3TTS_HEADS_IDLE_YIELD") != NULL;
    hd_main_work = getenv("Q3TTS_HEADS_MAIN_WORK") != NULL;
    for (long i = 0; i < NTH; i++) pthread_create(&hd_th[i], NULL, hd_worker, (void *)i);
}
static inline void prepare_xh(const float *x, uint16_t *xh, int i) {
    size_t vl;
    for (int d = 0; d < i; d += (int)vl) {
        vl = __riscv_vsetvl_e32m4(i - d);
        vfloat16m2_t h = __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(x + d, vl), vl);
        __riscv_vse16_v_f16m2((_Float16 *)(xh + d), h, vl);
    }
}
static inline void mv_f16(const uint16_t *wf, int o, int i, const float *x, float *y) {
    static uint16_t xh[4096];
    prepare_xh(x, xh, i);
    const int parts = hd_main_work ? (NTH + 1) : NTH;
    int per = (o + parts - 1) / parts;
    for (int t = 0; t < NTH; t++) {
        hd_job[t] = (Job){wf, xh, o, i, y, t * per, (t + 1) * per > o ? o : (t + 1) * per};
        __atomic_store_n(&hd_go[t], 1, __ATOMIC_RELEASE);
    }
    if (hd_main_work) {
        int t0 = NTH * per;
        int t1 = (NTH + 1) * per > o ? o : (NTH + 1) * per;
        for (int r = t0; r < t1; r++) {
            __builtin_prefetch(wf + (long)(r + 1) * i);
            y[r] = dot_f16(wf + (long)r * i, xh, i);
        }
    }
    for (int t = 0; t < NTH; t++) {
        while (!__atomic_load_n(&hd_done[t], __ATOMIC_ACQUIRE)) { __asm__ volatile("pause"); }
        __atomic_store_n(&hd_done[t], 0, __ATOMIC_RELAXED);
    }
}
static inline int mv_f16_argmax(const uint16_t *wf, int o, int i, const float *x, float *best_out) {
    static uint16_t xh[4096];
    prepare_xh(x, xh, i);
    const int parts = hd_main_work ? (NTH + 1) : NTH;
    int per = (o + parts - 1) / parts;
    for (int t = 0; t < NTH; t++) {
        hd_job[t] = (Job){wf, xh, o, i, NULL, t * per, (t + 1) * per > o ? o : (t + 1) * per};
        __atomic_store_n(&hd_go[t], 1, __ATOMIC_RELEASE);
    }
    int main_best_i = NTH * per;
    float main_best = -1e30f;
    if (hd_main_work) {
        int t0 = NTH * per;
        int t1 = (NTH + 1) * per > o ? o : (NTH + 1) * per;
        for (int r = t0; r < t1; r++) {
            __builtin_prefetch(wf + (long)(r + 1) * i);
            float v = dot_f16(wf + (long)r * i, xh, i);
            if (v > main_best) {
                main_best = v;
                main_best_i = r;
            }
        }
    }
    for (int t = 0; t < NTH; t++) {
        while (!__atomic_load_n(&hd_done[t], __ATOMIC_ACQUIRE)) { __asm__ volatile("pause"); }
        __atomic_store_n(&hd_done[t], 0, __ATOMIC_RELAXED);
    }
    int best_i = 0;
    float best = -1e30f;
    for (int t = 0; t < NTH; t++) {
        if (hd_best_val[t] > best) {
            best = hd_best_val[t];
            best_i = hd_best_idx[t];
        }
    }
    if (hd_main_work && main_best > best) {
        best = main_best;
        best_i = main_best_i;
    }
    if (best_out) {
        *best_out = best;
    }
    return best_i;
}
static inline int mv_f16_argmax_eos(const uint16_t *wf, int o, int eos, int i, const float *x, float *best_out, float *eos_out) {
    static uint16_t xh[4096];
    prepare_xh(x, xh, i);
    const int parts = hd_main_work ? (NTH + 1) : NTH;
    int per = (o + parts - 1) / parts;
    for (int t = 0; t < NTH; t++) {
        hd_job[t] = (Job){wf, xh, o, i, NULL, t * per, (t + 1) * per > o ? o : (t + 1) * per};
        __atomic_store_n(&hd_go[t], 1, __ATOMIC_RELEASE);
    }
    int main_best_i = NTH * per;
    float main_best = -1e30f;
    if (hd_main_work) {
        int t0 = NTH * per;
        int t1 = (NTH + 1) * per > o ? o : (NTH + 1) * per;
        for (int r = t0; r < t1; r++) {
            __builtin_prefetch(wf + (long)(r + 1) * i);
            float v = dot_f16(wf + (long)r * i, xh, i);
            if (v > main_best) {
                main_best = v;
                main_best_i = r;
            }
        }
    }
    for (int t = 0; t < NTH; t++) {
        while (!__atomic_load_n(&hd_done[t], __ATOMIC_ACQUIRE)) { __asm__ volatile("pause"); }
        __atomic_store_n(&hd_done[t], 0, __ATOMIC_RELAXED);
    }
    int best_i = 0;
    float best = -1e30f;
    for (int t = 0; t < NTH; t++) {
        if (hd_best_val[t] > best) {
            best = hd_best_val[t];
            best_i = hd_best_idx[t];
        }
    }
    if (hd_main_work && main_best > best) {
        best = main_best;
        best_i = main_best_i;
    }
    if (best_out) {
        *best_out = best;
    }
    if (eos_out) {
        *eos_out = dot_f16(wf + (long)eos * i, xh, i);
    }
    return best_i;
}
