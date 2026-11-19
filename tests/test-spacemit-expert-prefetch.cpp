// Hardware experiment: QKV-triggered whole-layer expert staging. Not a default CTest.
#include "llama.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-spacemit.h"
#include "ggml-spacemit-transfer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

using clock_type = std::chrono::steady_clock;
static double ms(clock_type::time_point a, clock_type::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}
static long proc_kib(const char * field) {
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        if (line.compare(0, std::strlen(field), field) == 0) {
            return std::strtol(line.c_str() + std::strlen(field), nullptr, 10);
        }
    }
    return -1;
}
static uint64_t read_bytes() {
    std::ifstream in("/proc/self/io");
    std::string key;
    uint64_t value;
    while (in >> key >> value) { if (key == "read_bytes:") { return value; } }
    return 0;
}
struct expert_copy {
    ggml_tensor * tensor;
    void * source;
    void * destination;
    size_t bytes;
    ggml_spacemit_transfer_event_t event;
};
struct experiment {
    bool prefetch = false;
    bool sequential = true;
    bool mmap_warmup = false;
    bool mmap_normal = false;
    size_t pending_at_experts = 0, pending_after_first = 0;
    bool submitted = false;
    bool ready = false;
    size_t bytes = 0, qkv_events = 0, expert_nodes = 0;
    int submit_cpu = -1;
    ggml_spacemit_transfer_t mover = nullptr;
    std::vector<expert_copy> copies;
    clock_type::time_point submitted_at;
    double overlap_ms = 0, wait_ms = 0, ready_ms = 0, submit_us = 0;

    ~experiment() { reset(); }
    void reset() {
        // Cancellation is deliberately unused. Accepted requests always finish.
        if (mover) { ggml_spacemit_transfer_drain(mover); }
        for (auto & c : copies) {
            c.tensor->data = c.source;
            ggml_spacemit_transfer_event_free(c.event);
            if (!mmap_warmup) { munmap(c.destination, c.bytes); }
        }
        copies.clear();
        ggml_spacemit_transfer_free(mover);
        mover = nullptr;
    }
    void stage(llama_model * model) {
        // This experiment supports one layer; extending it needs a layer policy.
        require(llama_model_n_layer(model) == 1, "experiment requires a single-layer model");
        size_t tensor_count = 0;
        copies.reserve(3);
        for (const auto & entry : model->tensors_by_name) {
            const auto & name = entry.first;
            if (name != "blk.0.ffn_gate_exps.weight" &&
                name != "blk.0.ffn_up_exps.weight" &&
                name != "blk.0.ffn_down_exps.weight") { continue; }
            auto * t = entry.second;
            require(t->data && t->buffer && ggml_backend_buffer_is_host(t->buffer),
                    "expert weights must use CPU-accessible host buffers");
            ++tensor_count;
            bytes += ggml_nbytes(t);
            if (!prefetch && !mmap_normal) { continue; }
            const size_t size = ggml_nbytes(t);
            if (sequential) {
                // Undo lazy mmap MADV_RANDOM without reading pages here. Warmup
                // uses NORMAL to allow readahead without a streaming eviction
                // hint. Actual page touches occur later on X100 at QKV BEGIN.
                const size_t page = size_t(sysconf(_SC_PAGESIZE));
                const uintptr_t address = reinterpret_cast<uintptr_t>(t->data);
                const uintptr_t begin = address & ~(uintptr_t(page) - 1);
                require(madvise(reinterpret_cast<void *>(begin), size + address - begin,
                                (mmap_warmup || mmap_normal) ? MADV_NORMAL : MADV_SEQUENTIAL) == 0, "cannot advise sequential expert source");
            }
            if (!prefetch) { continue; }
            if (mmap_warmup) {
                auto event = ggml_spacemit_transfer_event_create();
                require(event != nullptr, "cannot create warmup event");
                copies.push_back({t, t->data, t->data, size, event});
                continue; // Keep the loader's mapping and every tensor address.
            }
            void * dst = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            require(dst != MAP_FAILED, "cannot reserve expert DDR staging space");
            auto event = ggml_spacemit_transfer_event_create();
            if (!event) { munmap(dst, size); throw std::runtime_error("cannot create transfer event"); }
            copies.push_back({t, t->data, dst, size, event});
            // Set stable destinations BEFORE context creation/reserve. The CPU
            // graph borrows these addresses, and must wait before reading them.
            t->data = dst;
        }
        require(tensor_count == 3 && bytes > 0 && (!prefetch || copies.size() == 3), "expected all three expert tensors");
    }
    void submit() {
        if (!prefetch || submitted) { return; }
        submitted_at = clock_type::now();
        submit_cpu = sched_getcpu();
        for (auto & c : copies) {
            const ggml_spacemit_transfer_request request{c.destination, c.source, c.bytes};
            // A failed submission must never expose uninitialized expert bytes.
            // No synchronous copy fallback is allowed on an A100 worker.
            const auto status = mmap_warmup
                    ? ggml_spacemit_transfer_prefetch(mover, c.source, c.bytes, c.event)
                    : ggml_spacemit_transfer_submit(mover, &request, c.event);
            if (status != GGML_SPACEMIT_TRANSFER_OK) {
                std::fprintf(stderr, "FAIL expert transfer submission\n");
                std::abort();
            }
        }
        submitted = true;
        submit_us = 1000 * ms(submitted_at, clock_type::now());
        std::fprintf(stderr, "PREFETCH submit before QKV layer=0 cpu=%d tensors=%zu bytes=%zu submit_us=%.3f\n",
                     submit_cpu, copies.size(), bytes, submit_us);
    }
    void wait() {
        if (!prefetch || ready) { return; }
        if (!submitted) {
            std::fprintf(stderr, "FAIL expert consumption reached without a QKV BEGIN event\n");
            std::abort();
        }
        const auto begin = clock_type::now();
        overlap_ms = ms(submitted_at, begin);
        if (mmap_warmup) {
            for (const auto & c : copies) {
                require(c.tensor->data == c.source, "mmap tensor address changed");
                pending_at_experts += ggml_spacemit_transfer_query(c.event) == GGML_SPACEMIT_TRANSFER_PENDING;
            }
            ready = true; // Observed consumption, NOT whole-range residency.
            std::fprintf(stderr, "MMAP warmup: experts proceed without wait, pending=%zu overlap_ms=%.3f\n",
                         pending_at_experts, overlap_ms);
            return;
        }
        for (auto & c : copies) {
            if (ggml_spacemit_transfer_wait(c.event) != GGML_SPACEMIT_TRANSFER_OK) {
                std::fprintf(stderr, "FAIL expert transfer completion\n");
                std::abort();
            }
        }
        const auto end = clock_type::now();
        wait_ms = ms(begin, end);
        ready_ms = ms(submitted_at, end);
        ready = true;
        std::fprintf(stderr, "PREFETCH ready before CPU expert split overlap_ms=%.3f wait_ms=%.3f ready_ms=%.3f\n",
                     overlap_ms, wait_ms, ready_ms);
    }
};
static void on_module(const ggml_spacemit_prefetch_event * event, void * opaque) {
    auto & state = *static_cast<experiment *>(opaque);
    if (event->module == GGML_SPACEMIT_MODULE_QKV &&
        event->phase == GGML_SPACEMIT_PREFETCH_BEGIN && event->layer == 0) {
        ++state.qkv_events;
        state.submit();
    }
}
static bool on_eval(ggml_tensor * tensor, bool ask, void * opaque) {
    auto & state = *static_cast<experiment *>(opaque);
    if (ask && tensor->op == GGML_OP_MUL_MAT_ID) {
        ++state.expert_nodes;
        // The scheduler calls 'ask' before executing this CPU subgraph. A
        // SpacemiT-only EXPERTS callback would miss these MXFP4 expert GEMMs.
        // All experiment modes install the same observer. Returning false
        // preserves each scheduler split as one graph, without node chopping.
        state.wait();
    }
    return false;
}
static uint64_t save_logits(llama_context * ctx, const std::string & path, int32_t vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    require(logits != nullptr, "missing logits");
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int32_t i = 0; i < vocab; ++i) { require(std::isfinite(logits[i]), "nonfinite logits"); }
    const auto * bytes = reinterpret_cast<const unsigned char *>(logits);
    for (size_t i = 0; i < size_t(vocab) * sizeof(float); ++i) { hash = (hash ^ bytes[i]) * UINT64_C(1099511628211); }
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(logits), size_t(vocab) * sizeof(float));
    require(out.good(), "cannot save logits");
    return hash;
}
int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s MODEL resident|mmap|mmap-eager|prefetch|prefetch-random|mmap-prefetch|mmap-normal OUTPUT_PREFIX PP_TOKENS\n", argv[0]);
        return 2;
    }
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    ggml_backend_t observer = nullptr;
    int result = 1;
    experiment state; // Outlives context/backend cleanup, but is reset before model_free.
    const auto process_start = clock_type::now();
    try {
        const std::string mode = argv[2], prefix = argv[3];
        const int pp = std::stoi(argv[4]);
        require(mode == "resident" || mode == "mmap" || mode == "mmap-eager" || mode == "prefetch" || mode == "prefetch-random" || mode == "mmap-prefetch" || mode == "mmap-normal", "unknown mode");
        require(pp > 0 && pp <= 128, "PP tokens must be 1..128");
        state.mmap_warmup = mode == "mmap-prefetch";
        state.mmap_normal = mode == "mmap-normal";
        state.prefetch = state.mmap_warmup || mode == "prefetch" || mode == "prefetch-random";
        state.sequential = mode != "prefetch-random";
        if (state.prefetch) {
            ggml_spacemit_transfer_status status;
            const ggml_spacemit_transfer_config config{1, 8};
            state.mover = ggml_spacemit_transfer_create(&config, &status);
            require(state.mover && status == GGML_SPACEMIT_TRANSFER_OK, "no available X100 transfer worker");
        }
        ggml_backend_load_all();
        llama_backend_init();
        observer = ggml_backend_spacemit_init();
        require(observer != nullptr, "SpacemiT backend unavailable");
        // The callback registration is shared by handles on this device.
        require(ggml_backend_spacemit_set_prefetch_callback(observer, on_module, &state), "callback registration failed");
        auto params = llama_model_default_params();
        params.n_gpu_layers = 99;
        params.load_mode = mode == "resident" ? LLAMA_LOAD_MODE_NONE : LLAMA_LOAD_MODE_MMAP;
        params.lazy_mode = (mode == "resident" || mode == "mmap-eager") ? LLAMA_LAZY_MODE_OFF : LLAMA_LAZY_MODE_ON;
        // This benchmark measures CPU expert consumption, including when future
        // device kernels acquire MXFP4 support. Keep baseline placement identical.
        const llama_model_tensor_buft_override overrides[] = {
            {".*ffn_(gate|up|down)_exps.*", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
        params.tensor_buft_overrides = overrides;
        const bool io_available = access("/proc/self/io", R_OK) == 0;
        const uint64_t io_before_load = read_bytes();
        const auto load_begin = clock_type::now();
        model = llama_model_load_from_file(argv[1], params);
        require(model != nullptr, "model load failed");
        const auto load_end = clock_type::now();
        const uint64_t io_after_load = read_bytes();
        state.stage(model);
        auto cparams = llama_context_default_params();
        cparams.n_ctx = 256;
        cparams.n_batch = pp;
        cparams.n_ubatch = pp;
        cparams.n_threads = 8;
        cparams.n_threads_batch = 8;
        cparams.prefetch = true;
        cparams.cb_eval = on_eval;
        cparams.cb_eval_user_data = &state;
        const auto context_begin = clock_type::now();
        ctx = llama_init_from_model(model, cparams);
        require(ctx != nullptr, "context creation failed");
        const auto context_end = clock_type::now();
        const long rss_after_load = proc_kib("VmRSS:");
        std::vector<llama_token> tokens(pp);
        for (int i = 0; i < pp; ++i) { tokens[i] = 100 + i; }
        const auto first_begin = clock_type::now();
        require(llama_decode(ctx, llama_batch_get_one(tokens.data(), pp)) == 0, "first prefill failed");
        llama_synchronize(ctx);
        const auto first_end = clock_type::now();
        if (state.mmap_warmup) {
            for (const auto & c : state.copies) {
                state.pending_after_first += ggml_spacemit_transfer_query(c.event) == GGML_SPACEMIT_TRANSFER_PENDING;
            }
        }
        const uint64_t io_after_first = read_bytes();
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const auto first_hash = save_logits(ctx, prefix + ".first.logits", vocab);
        require(state.qkv_events > 0 && state.expert_nodes > 0, "missing QKV or expert observation");
        require(!state.prefetch || state.ready, "expert staging was not consumed");
        const auto * logits = llama_get_logits_ith(ctx, -1);
        llama_token token = std::max_element(logits, logits + vocab) - logits;
        const auto decode_begin = clock_type::now();
        for (int i = 0; i < 16; ++i) {
            require(llama_decode(ctx, llama_batch_get_one(&token, 1)) == 0, "decode failed");
            logits = llama_get_logits_ith(ctx, -1);
            token = std::max_element(logits, logits + vocab) - logits;
        }
        llama_synchronize(ctx);
        const auto decode_end = clock_type::now();
        const auto decode_hash = save_logits(ctx, prefix + ".decode.logits", vocab);
        double warm_ms = 0;
        uint64_t warm_hash = 0;
        for (int r = 0; r < 3; ++r) {
            llama_memory_clear(llama_get_memory(ctx), true);
            const auto begin = clock_type::now();
            require(llama_decode(ctx, llama_batch_get_one(tokens.data(), pp)) == 0, "warm prefill failed");
            llama_synchronize(ctx);
            const auto end = clock_type::now();
            warm_ms += ms(begin, end);
            warm_hash = save_logits(ctx, prefix + ".warm.logits", vocab);
            require(warm_hash == first_hash, "repeat prefill logits changed");
        }
        const auto measured_end = clock_type::now();
        const long rss_end = proc_kib("VmRSS:"), rss_anon = proc_kib("RssAnon:"), rss_file = proc_kib("RssFile:");
        const uint64_t io_after_measurement = read_bytes();
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        // Drain only after all inference measurements; never at expert consumption.
        const auto drain_begin = clock_type::now();
        if (state.mover) { ggml_spacemit_transfer_drain(state.mover); }
        const double final_drain_ms = ms(drain_begin, clock_type::now());
        // Copy mode verifies all experts outside timing. Warmup checks completion
        // and original tensor addresses; it never claims a byte-copy check.
        for (const auto & c : state.copies) {
            require(ggml_spacemit_transfer_query(c.event) == GGML_SPACEMIT_TRANSFER_COMPLETE, "transfer failed");
            if (state.mmap_warmup) {
                require(c.tensor->data == c.source, "mmap tensor address changed");
            } else {
                require(std::memcmp(c.source, c.destination, c.bytes) == 0, "expert copy mismatch");
            }
        }
        std::ofstream out(prefix + ".json");
        out << "{\n  \"mode\": \"" << mode << "\",\n  \"pp_tokens\": " << pp
            << ",\n  \"expert_bytes\": " << state.bytes
            << ",\n  \"load_ms\": " << ms(load_begin, load_end)
            << ",\n  \"context_ms\": " << ms(context_begin, context_end)
            << ",\n  \"first_pp_ms\": " << ms(first_begin, first_end)
            << ",\n  \"warm_pp_ms\": " << warm_ms / 3
            << ",\n  \"decode16_ms\": " << ms(decode_begin, decode_end)
            << ",\n  \"total_measured_ms\": " << ms(process_start, measured_end)
            << ",\n  \"submit_us\": " << state.submit_us
            << ",\n  \"submit_cpu\": " << state.submit_cpu
            << ",\n  \"overlap_ms\": " << state.overlap_ms
            << ",\n  \"wait_ms\": " << state.wait_ms
            << ",\n  \"ready_ms\": " << state.ready_ms
            << ",\n  \"mmap_tensor_addresses_unchanged\": " << (state.mmap_warmup ? "true" : "false")
            << ",\n  \"pending_at_experts\": " << state.pending_at_experts
            << ",\n  \"pending_after_first\": " << state.pending_after_first
            << ",\n  \"final_drain_ms\": " << final_drain_ms
            << ",\n  \"io_counters_available\": " << (io_available ? "true" : "false")
            << ",\n  \"load_read_bytes\": " << (io_after_load - io_before_load)
            << ",\n  \"first_pp_read_bytes\": " << (io_after_first - io_after_load)
            << ",\n  \"total_read_bytes\": " << (io_after_measurement - io_before_load)
            << ",\n  \"load_plus_first_ms\": " << ms(load_begin, first_end)
            << ",\n  \"rss_after_load_kib\": " << rss_after_load
            << ",\n  \"rss_end_kib\": " << rss_end
            << ",\n  \"rss_anon_kib\": " << rss_anon
            << ",\n  \"rss_file_kib\": " << rss_file
            << ",\n  \"maxrss_kib\": " << usage.ru_maxrss
            << ",\n  \"major_faults\": " << usage.ru_majflt
            << ",\n  \"minor_faults\": " << usage.ru_minflt
            << ",\n  \"qkv_events\": " << state.qkv_events
            << ",\n  \"expert_nodes\": " << state.expert_nodes
            << ",\n  \"first_logits_hash\": \"" << first_hash
            << "\",\n  \"decode_logits_hash\": \"" << decode_hash
            << "\",\n  \"all_expert_bytes_verified\": " << (state.prefetch && !state.mmap_warmup ? "true" : "false") << "\n}\n";
        require(out.good(), "cannot write result JSON");
        result = 0;
        // Stop callbacks and free graphs before state restores model addresses.
        ggml_backend_spacemit_set_prefetch_callback(observer, nullptr, nullptr);
        llama_free(ctx); ctx = nullptr;
        ggml_backend_free(observer); observer = nullptr;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        if (observer) { ggml_backend_spacemit_set_prefetch_callback(observer, nullptr, nullptr); }
    }
    if (ctx) { llama_free(ctx); }
    if (observer) { ggml_backend_free(observer); }
    state.reset();
    if (model) { llama_model_free(model); }
    llama_backend_free();
    return result;
}
