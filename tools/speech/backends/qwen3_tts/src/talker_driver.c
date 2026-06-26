// Qwen3-TTS frame loop in C: talker + code-predictor in one process (two llama contexts),
// cp lm-heads as in-process GEMV. Streams 16-code frames (int32) to stdout for the codec.
// usage: talker_driver TALKER.gguf CP.gguf _ PREFILL.bin NPREFILL TRAILING.bin NTRAIL PAD.bin MAXFRAMES
//        talker_driver TALKER.gguf CP.gguf --jobs JOBS.txt
//        talker_driver TALKER.gguf CP.gguf --jobs-stdin MAX_PREFILL MAXFRAMES
#include "llama.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <math.h>
#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "heads_pool.h"

#define HID 1024
#define NCB 16
#define CPV 2048
#define EOS 2150

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void log_to_stderr(enum ggml_log_level level, const char *text, void *user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
}

static int saved_stdout = -1;
static int sticky_qwen3_embed_only = 0;

static void default_llama_env(void) {
    if (getenv("LLAMA_GRAPH_REUSE_2WAY") == NULL) {
        setenv("LLAMA_GRAPH_REUSE_2WAY", "1", 0);
    }
}

static void stdout_to_stderr(void) {
    fflush(stdout);
    saved_stdout = dup(STDOUT_FILENO);
    if (saved_stdout >= 0) {
        dup2(STDERR_FILENO, STDOUT_FILENO);
    }
}

static void restore_stdout(void) {
    fflush(stdout);
    if (saved_stdout >= 0) {
        dup2(saved_stdout, STDOUT_FILENO);
        close(saved_stdout);
        saved_stdout = -1;
    }
}

static int cp_decode(struct llama_context *ctx, struct llama_batch batch) {
    if (sticky_qwen3_embed_only) {
        return llama_decode(ctx, batch);
    }
    if (getenv("Q3TTS_CP_EMBED_ONLY") == NULL) {
        return llama_decode(ctx, batch);
    }
    setenv("LLAMA_QWEN3_EMBED_ONLY", "1", 1);
    int ret = llama_decode(ctx, batch);
    unsetenv("LLAMA_QWEN3_EMBED_ONLY");
    return ret;
}

static void set_ctx_threads_if_changed(struct llama_context *ctx, int *cur_threads, int *cur_threads_batch,
                                       int threads, int threads_batch) {
    if (threads <= 0 || threads_batch <= 0) {
        return;
    }
    if (*cur_threads == threads && *cur_threads_batch == threads_batch) {
        return;
    }
    llama_set_n_threads(ctx, threads, threads_batch);
    *cur_threads = threads;
    *cur_threads_batch = threads_batch;
}

static int talker_decode(struct llama_context *ctx, struct llama_batch batch, int embed_only) {
    if (sticky_qwen3_embed_only) {
        return llama_decode(ctx, batch);
    }
    if (!embed_only) {
        return llama_decode(ctx, batch);
    }
    setenv("LLAMA_QWEN3_EMBED_ONLY", "1", 1);
    int ret = llama_decode(ctx, batch);
    unsetenv("LLAMA_QWEN3_EMBED_ONLY");
    return ret;
}

static float *load_bin(const char *path, long n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s fail\n", path); exit(1); }
    float *p = malloc(n * 4);
    if (fread(p, 4, n, f) != (size_t)n) { fprintf(stderr, "read %s fail\n", path); exit(1); }
    fclose(f);
    return p;
}
static uint16_t *load_bin16(const char *path, long n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s fail\n", path); exit(1); }
    uint16_t *p = malloc(n * 2);
    if (fread(p, 2, n, f) != (size_t)n) { fprintf(stderr, "read %s fail\n", path); exit(1); }
    fclose(f);
    return p;
}

static void *load_gguf_tensor(const char *path, const char *name, enum ggml_type type, size_t bytes) {
    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx = NULL,
    };
    struct gguf_context *ctx = gguf_init_from_file(path, params);
    if (!ctx) { fprintf(stderr, "open gguf %s fail\n", path); exit(1); }

    int64_t tid = gguf_find_tensor(ctx, name);
    if (tid < 0) { fprintf(stderr, "tensor %s not found in %s\n", name, path); exit(1); }
    if (gguf_get_tensor_type(ctx, tid) != type) {
        fprintf(stderr, "tensor %s type mismatch in %s\n", name, path);
        exit(1);
    }
    if (gguf_get_tensor_size(ctx, tid) != bytes) {
        fprintf(stderr, "tensor %s size mismatch in %s\n", name, path);
        exit(1);
    }

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s fail\n", path); exit(1); }
    if (fseek(f, (long)(gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, tid)), SEEK_SET) != 0) {
        fprintf(stderr, "seek %s:%s fail\n", path, name);
        exit(1);
    }
    void *p = malloc(bytes);
    if (!p) { fprintf(stderr, "malloc %zu fail\n", bytes); exit(1); }
    if (fread(p, 1, bytes, f) != bytes) {
        fprintf(stderr, "read %s:%s fail\n", path, name);
        exit(1);
    }
    fclose(f);
    gguf_free(ctx);
    return p;
}

static int env_int(const char *name, int fallback) {
    const char *v = getenv(name);
    if (!v || !*v) {
        return fallback;
    }
    return atoi(v);
}

static float env_float(const char *name, float fallback) {
    const char *v = getenv(name);
    if (!v || !*v) {
        return fallback;
    }
    return (float)atof(v);
}

static int env_bool_default(const char *name, int fallback) {
    const char *v = getenv(name);
    if (!v || !*v) {
        return fallback;
    }
    if (v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N') {
        return 0;
    }
    return 1;
}

static int max_i(int a, int b) {
    return a > b ? a : b;
}

static struct ggml_threadpool *create_llama_threadpool(int n_threads) {
    if (env_int("Q3TTS_LLAMA_THREADPOOL", 1) == 0 || n_threads <= 1) {
        return NULL;
    }

    struct ggml_threadpool_params tp = ggml_threadpool_params_default(n_threads);
    int poll = env_int("Q3TTS_LLAMA_THREADPOOL_POLL", -1);
    if (poll >= 0) {
        if (poll > 100) {
            poll = 100;
        }
        tp.poll = (uint32_t)poll;
    }

    struct ggml_threadpool *threadpool = ggml_threadpool_new(&tp);
    if (!threadpool) {
        fprintf(stderr, "llama_threadpool create failed threads %d\n", n_threads);
        return NULL;
    }
    fprintf(stderr, "llama_threadpool threads %d poll %u\n", n_threads, tp.poll);
    return threadpool;
}

static struct llama_context *mk_ctx(struct llama_model *m, int n_ctx, int n_batch, int n_ubatch, int n_threads, int n_threads_batch, int n_seq_max) {
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_batch; cp.n_ubatch = n_ubatch;
    if (n_seq_max > 0) { cp.n_seq_max = n_seq_max; }
    cp.embeddings = true; cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
#ifdef Q3TTS_LLAMA_BOOL_FLASH_ATTN
    cp.flash_attn = env_bool_default("Q3TTS_FLASH_ATTN", 1) != 0;
#else
    cp.flash_attn_type = env_bool_default("Q3TTS_FLASH_ATTN", 1) != 0 ?
        LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
#endif
    if (n_threads > 0) { cp.n_threads = n_threads; }
    if (n_threads_batch > 0) { cp.n_threads_batch = n_threads_batch; }
    else if (n_threads > 0) { cp.n_threads_batch = n_threads; }
    return llama_init_from_model(m, cp);
}

struct job_spec {
    char prefill[512];
    int n_prefill;
    char trailing[512];
    int n_trail;
    char pad[512];
    int max_frames;
    char out[512];
    char done[512];
};

static int read_job_line(FILE *f, struct job_spec *j) {
    memset(j, 0, sizeof(*j));
    int got = fscanf(f, "%511s %d %511s %d %511s %d %511s %511s",
                     j->prefill, &j->n_prefill, j->trailing, &j->n_trail,
                     j->pad, &j->max_frames, j->out, j->done);
    if (got == EOF) {
        return 0;
    }
    if (got != 7 && got != 8) {
        return -1;
    }
    return 1;
}

static int read_jobs(const char *path, struct job_spec **jobs_out, int *n_jobs_out) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "open jobs %s fail\n", path);
        return 1;
    }
    int cap = 8;
    int n = 0;
    struct job_spec *jobs = calloc((size_t)cap, sizeof(*jobs));
    if (!jobs) {
        fclose(f);
        return 1;
    }
    while (1) {
        struct job_spec j;
        int got = read_job_line(f, &j);
        if (got == 0) {
            break;
        }
        if (got < 0) {
            fprintf(stderr, "bad jobs line in %s\n", path);
            free(jobs);
            fclose(f);
            return 1;
        }
        if (n == cap) {
            cap *= 2;
            struct job_spec *next = realloc(jobs, (size_t)cap * sizeof(*jobs));
            if (!next) {
                free(jobs);
                fclose(f);
                return 1;
            }
            jobs = next;
        }
        jobs[n++] = j;
    }
    fclose(f);
    *jobs_out = jobs;
    *n_jobs_out = n;
    return n == 0;
}

static int run_job(struct llama_context *ct, struct llama_context *cc,
                   struct llama_batch *bt, struct llama_batch *bc,
                   float *cemb, float **pemb, uint16_t **heads, uint16_t *talker_head,
                   const struct job_spec *job, FILE *out) {
    float *prefill = load_bin(job->prefill, (long)job->n_prefill * HID);
    float *trail = load_bin(job->trailing, (long)job->n_trail * HID);
    float *pad = load_bin(job->pad, HID);

    llama_memory_clear(llama_get_memory(ct), true);
    llama_memory_clear(llama_get_memory(cc), true);

    double t0 = now_s();
    memcpy(bt->embd, prefill, (size_t)job->n_prefill * HID * 4);
    for (int i = 0; i < job->n_prefill; i++) {
        bt->pos[i] = i;
        bt->n_seq_id[i] = 1;
        bt->seq_id[i][0] = 0;
        bt->logits[i] = i == job->n_prefill - 1;
    }
    bt->n_tokens = job->n_prefill;
    if (talker_decode(ct, *bt, talker_head != NULL)) {
        return 1;
    }
    fprintf(stderr, "prefill %d tok %.3fs\n", job->n_prefill, now_s() - t0);

    double t_talker = 0, t_cp = 0, t_heads = 0;
    double t_cp_clear = 0, t_cp_init_decode = 0, t_cp_step_decode = 0;
    const int cp_seq_per_frame = getenv("Q3TTS_CP_SEQ_PER_FRAME") != NULL;
    const int head_argmax = getenv("Q3TTS_HEAD_ARGMAX") != NULL;
    const int cp_meta_clear = getenv("Q3TTS_CP_META_CLEAR") != NULL;
    const int cp_seq_rm = getenv("Q3TTS_CP_SEQ_RM") != NULL;
    const int cp_profile = env_int("Q3TTS_CP_PROFILE", 0) != 0;
    int cp_active_heads = env_int("Q3TTS_CP_ACTIVE_HEADS", 15);
    if (cp_active_heads < 1) {
        cp_active_heads = 1;
    }
    if (cp_active_heads > 15) {
        cp_active_heads = 15;
    }
    int cp_fill_code = env_int("Q3TTS_CP_FILL_CODE", 0);
    const int cp_switch_threads =
        getenv("Q3TTS_CP_INIT_THREADS") || getenv("Q3TTS_CP_INIT_THREADS_BATCH") ||
        getenv("Q3TTS_CP_STEP_THREADS") || getenv("Q3TTS_CP_STEP_THREADS_BATCH");
    int cp_cur_threads = llama_n_threads(cc);
    int cp_cur_threads_batch = llama_n_threads_batch(cc);
    const int cp_init_threads = env_int("Q3TTS_CP_INIT_THREADS", cp_cur_threads);
    const int cp_init_threads_batch = env_int("Q3TTS_CP_INIT_THREADS_BATCH", cp_cur_threads_batch);
    const int cp_step_threads = env_int("Q3TTS_CP_STEP_THREADS", cp_init_threads);
    const int cp_step_threads_batch = env_int("Q3TTS_CP_STEP_THREADS_BATCH", cp_init_threads_batch);
    int frames = 0;
    int codes[NCB];
    unsigned char seen_c0[CPV];
    memset(seen_c0, 0, sizeof(seen_c0));
    const float repetition_penalty = env_float("Q3TTS_TALKER_REPETITION_PENALTY", 1.0f);
    const int use_repetition_penalty = repetition_penalty > 1.0f;
    for (int f = 0; f < job->max_frames; f++) {
        for (int i = 0; i < NCB; i++) {
            codes[i] = cp_fill_code;
        }
        const float *hid = llama_get_embeddings_ith(ct, bt->n_tokens - 1);
        const float *lg = talker_head ? NULL : llama_get_logits_ith(ct, bt->n_tokens - 1);
        int c0 = 0;
        float best = -1e30f;
        float eos_logit = -1e30f;
        if (talker_head) {
            if (use_repetition_penalty) {
                static float talker_logits[EOS + 1];
                mv_f16(talker_head, EOS + 1, HID, hid, talker_logits);
                for (int i = 0; i < CPV; i++) {
                    float v = talker_logits[i];
                    if (seen_c0[i]) {
                        v = v < 0.0f ? v * repetition_penalty : v / repetition_penalty;
                    }
                    if (v > best) {
                        best = v;
                        c0 = i;
                    }
                }
                eos_logit = talker_logits[EOS];
            } else {
                c0 = mv_f16_argmax_eos(talker_head, 2048, EOS, HID, hid, &best, &eos_logit);
            }
        } else {
            for (int i = 0; i < 2048; i++) {
                float v = lg[i];
                if (use_repetition_penalty && seen_c0[i]) {
                    v = v < 0.0f ? v * repetition_penalty : v / repetition_penalty;
                }
                if (v > best) {
                    best = v;
                    c0 = i;
                }
            }
            eos_logit = lg[EOS];
        }
        if (f >= 2 && eos_logit > best) {
            fprintf(stderr, "EOS at %d\n", f);
            break;
        }
        codes[0] = c0;
        if (c0 >= 0 && c0 < CPV) {
            seen_c0[c0] = 1;
        }

        double tc = now_s();
        const int cp_seq = cp_seq_per_frame ? (f + 1) : 0;
        if (!cp_seq_per_frame) {
            double tclear = now_s();
            if (cp_seq_rm) {
                llama_memory_seq_rm(llama_get_memory(cc), 0, -1, -1);
            } else {
                llama_memory_clear(llama_get_memory(cc), !cp_meta_clear);
            }
            t_cp_clear += now_s() - tclear;
        }
        memcpy(bc->embd, hid, HID * 4);
        memcpy(bc->embd + HID, cemb + (long)c0 * HID, HID * 4);
        for (int i = 0; i < 2; i++) {
            bc->pos[i] = i;
            bc->n_seq_id[i] = 1;
            bc->seq_id[i][0] = cp_seq;
            bc->logits[i] = i == 1;
        }
        bc->n_tokens = 2;
        if (cp_switch_threads) {
            set_ctx_threads_if_changed(cc, &cp_cur_threads, &cp_cur_threads_batch,
                                       cp_init_threads, cp_init_threads_batch);
        }
        double td = now_s();
        if (cp_decode(cc, *bc)) {
            return 1;
        }
        t_cp_init_decode += now_s() - td;
        if (cp_switch_threads) {
            set_ctx_threads_if_changed(cc, &cp_cur_threads, &cp_cur_threads_batch,
                                       cp_step_threads, cp_step_threads_batch);
        }
        for (int s = 0; s < cp_active_heads; s++) {
            const float *h = llama_get_embeddings_ith(cc, bc->n_tokens - 1);
            double th = now_s();
            static float hl[CPV];
            int g = 0;
            float gb = -1e30f;
            if (head_argmax) {
                g = mv_f16_argmax(heads[s], CPV, HID, h, &gb);
            } else {
                mv_f16(heads[s], CPV, HID, h, hl);
                for (int v = 0; v < CPV; v++) {
                    if (hl[v] > gb) {
                        gb = hl[v];
                        g = v;
                    }
                }
            }
            t_heads += now_s() - th;
            codes[s + 1] = g;
            if (s == cp_active_heads - 1) {
                break;
            }
            memcpy(bc->embd, pemb[s] + (long)g * HID, HID * 4);
            bc->pos[0] = 2 + s;
            bc->n_seq_id[0] = 1;
            bc->seq_id[0][0] = cp_seq;
            bc->logits[0] = 1;
            bc->n_tokens = 1;
            td = now_s();
            if (cp_decode(cc, *bc)) {
                return 1;
            }
            t_cp_step_decode += now_s() - td;
        }
        if (cp_fill_code < 0 && cp_active_heads < 15) {
            int fill = codes[cp_active_heads];
            for (int i = cp_active_heads + 1; i < NCB; i++) {
                codes[i] = fill;
            }
        }
        t_cp += now_s() - tc;

        fwrite(codes, 4, NCB, out);
        fflush(out);
        frames++;

        double tt = now_s();
        float *e = bt->embd;
        const float *tr = f < job->n_trail ? trail + (long)f * HID : pad;
        for (int d = 0; d < HID; d++) {
            float acc = cemb[(long)codes[0] * HID + d] + tr[d];
            for (int i = 0; i < 15; i++) {
                acc += pemb[i][(long)codes[i + 1] * HID + d];
            }
            e[d] = acc;
        }
        bt->pos[0] = job->n_prefill + f;
        bt->n_seq_id[0] = 1;
        bt->seq_id[0][0] = 0;
        bt->logits[0] = 1;
        bt->n_tokens = 1;
        if (talker_decode(ct, *bt, talker_head != NULL)) {
            return 1;
        }
        t_talker += now_s() - tt;
    }

    if (frames > 0) {
        fprintf(stderr, "frames %d  talker %.1fms/f  cp %.1fms/f (heads %.1f)  total %.2fs\n",
                frames, t_talker / frames * 1000, t_cp / frames * 1000,
                t_heads / frames * 1000, now_s() - t0);
        if (cp_profile) {
            fprintf(stderr,
                    "cp_detail clear %.3fms/f  init_decode %.3fms/f  step_decode %.3fms/f  heads %.3fms/f\n",
                    t_cp_clear / frames * 1000,
                    t_cp_init_decode / frames * 1000,
                    t_cp_step_decode / frames * 1000,
                    t_heads / frames * 1000);
        }
    } else {
        fprintf(stderr, "frames 0  total %.2fs\n", now_s() - t0);
    }

    free(prefill);
    free(trail);
    free(pad);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "args\n");
        return 1;
    }

    struct job_spec *jobs = NULL;
    int n_jobs = 0;
    int jobs_mode = argc >= 5 && strcmp(argv[3], "--jobs") == 0;
    int jobs_stdin_mode = argc >= 6 && strcmp(argv[3], "--jobs-stdin") == 0;
    if (jobs_mode) {
        if (read_jobs(argv[4], &jobs, &n_jobs)) {
            return 1;
        }
    } else if (jobs_stdin_mode) {
        jobs = NULL;
        n_jobs = 0;
    } else {
        if (argc < 10) {
            fprintf(stderr, "args\n");
            return 1;
        }
        jobs = calloc(1, sizeof(*jobs));
        if (!jobs) {
            return 1;
        }
        n_jobs = 1;
        snprintf(jobs[0].prefill, sizeof(jobs[0].prefill), "%s", argv[4]);
        jobs[0].n_prefill = atoi(argv[5]);
        snprintf(jobs[0].trailing, sizeof(jobs[0].trailing), "%s", argv[6]);
        jobs[0].n_trail = atoi(argv[7]);
        snprintf(jobs[0].pad, sizeof(jobs[0].pad), "%s", argv[8]);
        jobs[0].max_frames = atoi(argv[9]);
        jobs[0].out[0] = '\0';
    }

    int max_prefill = 0;
    int max_frames = 0;
    if (jobs_stdin_mode) {
        max_prefill = atoi(argv[4]);
        max_frames = atoi(argv[5]);
        if (max_prefill <= 0 || max_frames <= 0) {
            fprintf(stderr, "bad --jobs-stdin limits\n");
            return 1;
        }
    } else {
        for (int i = 0; i < n_jobs; i++) {
            if (jobs[i].n_prefill > max_prefill) {
                max_prefill = jobs[i].n_prefill;
            }
            if (jobs[i].max_frames > max_frames) {
                max_frames = jobs[i].max_frames;
            }
        }
    }

    char p[512];
    float *cemb = load_gguf_tensor(argv[1], "q3tts.codec_embedding.weight", GGML_TYPE_F32, 3072UL * HID * sizeof(float));
    float *pemb[15];
    uint16_t *heads[15];
    for (int i = 0; i < 15; i++) {
        snprintf(p, sizeof(p), "q3tts.cp_embedding.%d.weight", i);
        pemb[i] = load_gguf_tensor(argv[2], p, GGML_TYPE_F32, (size_t)CPV * HID * sizeof(float));
        snprintf(p, sizeof(p), "q3tts.cp_head_f16.%d.weight", i);
        heads[i] = load_gguf_tensor(argv[2], p, GGML_TYPE_F16, (size_t)CPV * HID * sizeof(uint16_t));
    }
    uint16_t *talker_head = load_gguf_tensor(argv[1], "q3tts.talker_head_f16.weight", GGML_TYPE_F16, 3072UL * HID * sizeof(uint16_t));
    sticky_qwen3_embed_only = talker_head != NULL && getenv("Q3TTS_CP_EMBED_ONLY") != NULL;
    if (sticky_qwen3_embed_only) {
        setenv("LLAMA_QWEN3_EMBED_ONLY", "1", 1);
    }

    if (getenv("Q3TTS_OUTPUT_ONLY_EMBEDDINGS") != NULL) {
        setenv("LLAMA_EMBEDDINGS_OUTPUT_ONLY", "1", 1);
    }
    default_llama_env();
    stdout_to_stderr();
    llama_log_set(log_to_stderr, NULL);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    struct llama_model *mt = llama_model_load_from_file(argv[1], mp);
    if (!mt) {
        return 1;
    }
    struct llama_model *mc = llama_model_load_from_file(argv[2], mp);
    if (!mc) {
        return 1;
    }

    int talker_ctx = env_int("Q3TTS_TALKER_CTX", 4096);
    int min_talker_ctx = max_prefill + max_frames + 1;
    if (talker_ctx < min_talker_ctx) {
        talker_ctx = min_talker_ctx;
    }
    const int cp_seq_per_frame = getenv("Q3TTS_CP_SEQ_PER_FRAME") != NULL;
    int cp_ctx = env_int("Q3TTS_CP_CTX", cp_seq_per_frame ? (max_frames * 16 + 16) : 64);
    int cp_ctx_min = env_int("Q3TTS_CP_CTX_MIN", 32);
    if (cp_ctx_min < 16) {
        cp_ctx_min = 16;
    }
    if (cp_ctx < cp_ctx_min) {
        cp_ctx = cp_ctx_min;
    }
    int cp_seq_max = env_int("Q3TTS_CP_SEQ_MAX", cp_seq_per_frame ? (max_frames + 1) : 1);
    int talker_batch = env_int("Q3TTS_TALKER_BATCH", talker_ctx);
    if (talker_batch < max_prefill) {
        talker_batch = max_prefill;
    }
    int talker_ubatch = env_int("Q3TTS_TALKER_UBATCH", 512);
    if (talker_ubatch < 1) {
        talker_ubatch = 1;
    }
    int cp_batch = env_int("Q3TTS_CP_BATCH", cp_ctx);
    if (cp_batch < 2) {
        cp_batch = 2;
    }
    int cp_ubatch = env_int("Q3TTS_CP_UBATCH", 512);
    if (cp_ubatch < 1) {
        cp_ubatch = 1;
    }
    int talker_threads = env_int("Q3TTS_TALKER_THREADS", 0);
    int cp_threads = env_int("Q3TTS_CP_THREADS", 0);
    int talker_threads_batch = env_int("Q3TTS_TALKER_THREADS_BATCH", talker_threads);
    int cp_threads_batch = env_int("Q3TTS_CP_THREADS_BATCH", cp_threads);
    int cp_step_threads = env_int("Q3TTS_CP_STEP_THREADS", cp_threads);
    int cp_step_threads_batch = env_int("Q3TTS_CP_STEP_THREADS_BATCH", cp_threads_batch);
    struct llama_context *ct = mk_ctx(mt, talker_ctx, talker_batch, talker_ubatch, talker_threads,
                                      talker_threads_batch, 1);
    struct llama_context *cc = mk_ctx(mc, cp_ctx, cp_batch, cp_ubatch, cp_threads,
                                      cp_threads_batch, cp_seq_max);
    int llama_threads = max_i(llama_n_threads(ct), llama_n_threads_batch(ct));
    llama_threads = max_i(llama_threads, llama_n_threads(cc));
    llama_threads = max_i(llama_threads, llama_n_threads_batch(cc));
    llama_threads = max_i(llama_threads, cp_step_threads);
    llama_threads = max_i(llama_threads, cp_step_threads_batch);
    struct ggml_threadpool *llama_threadpool = create_llama_threadpool(llama_threads);
    if (llama_threadpool) {
        llama_attach_threadpool(ct, llama_threadpool, llama_threadpool);
        llama_attach_threadpool(cc, llama_threadpool, llama_threadpool);
    }
    struct llama_batch bt = llama_batch_init(max_prefill + max_frames + 1, HID, 1);
    struct llama_batch bc = llama_batch_init(2, HID, 1);
    pools_init();
    restore_stdout();
    if (jobs_stdin_mode) {
        fprintf(stderr, "talker_stdin_ready\n");
        fflush(stderr);
    }

    int exit_code = 0;
    for (int i = 0; jobs_stdin_mode || i < n_jobs; i++) {
        struct job_spec stdin_job;
        struct job_spec *job = jobs_stdin_mode ? &stdin_job : &jobs[i];
        if (jobs_stdin_mode) {
            int got = read_job_line(stdin, job);
            if (got == 0) {
                break;
            }
            if (got < 0) {
                fprintf(stderr, "bad stdin jobs line\n");
                exit_code = 1;
                break;
            }
            if (job->n_prefill > max_prefill || job->max_frames > max_frames) {
                fprintf(stderr, "stdin job exceeds limits: prefill %d/%d frames %d/%d\n",
                        job->n_prefill, max_prefill, job->max_frames, max_frames);
                exit_code = 1;
                break;
            }
        }
        FILE *out = stdout;
        if (job->out[0]) {
            out = fopen(job->out, "wb");
            if (!out) {
                fprintf(stderr, "open output %s fail\n", job->out);
                exit_code = 1;
                break;
            }
        }
        int rc = run_job(ct, cc, &bt, &bc, cemb, pemb, heads, talker_head, job, out);
        if (job->out[0]) {
            fclose(out);
        }
        if (rc) {
            exit_code = rc;
            break;
        }
        if (job->done[0]) {
            FILE *done = fopen(job->done, "wb");
            if (!done) {
                fprintf(stderr, "open done marker %s fail\n", job->done);
                exit_code = 1;
                break;
            }
            fclose(done);
        }
    }

    if (exit_code == 0 && getenv("Q3TTS_LLAMA_PERF_PRINT") != NULL) {
        fprintf(stderr, "talker perf:\n");
        llama_perf_context_print(ct);
        fprintf(stderr, "cp perf:\n");
        llama_perf_context_print(cc);
    }
    if (llama_threadpool) {
        llama_detach_threadpool(ct);
        llama_detach_threadpool(cc);
        ggml_threadpool_free(llama_threadpool);
    }
    return exit_code;
}
