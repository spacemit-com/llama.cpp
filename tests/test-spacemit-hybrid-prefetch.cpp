// Production hybrid-prefetch hardware benchmark; deliberately not a default CTest.
// Usage: test-spacemit-hybrid-prefetch MODEL MODE PREFIX 128
// Native float32, full-vocabulary last-token logits: .first/.decode/.warm.logits
// and .warm1/.warm2/.warm3.logits. Run modes in separate, exclusively queued processes.
#include "llama.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>

using clock_type = std::chrono::steady_clock;
static constexpr int pp_tokens = 128;
static constexpr int decode_tokens = 16;
static constexpr int warm_repeats = 3;
static constexpr const char * expert_override = ".*ffn_(gate|up|down)_exps.*";

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}
static double ms(clock_type::time_point begin, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}
static std::string json_string(const std::string & text) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { out << '\\' << char(c); }
        else if (c < 0x20) { out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(c); }
        else { out << char(c); }
    }
    out << '"';
    return out.str();
}
static void save_bytes(const std::string & path, const void * data, size_t size) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    require(out.good(), "cannot open output file");
    out.write(static_cast<const char *>(data), std::streamsize(size));
    out.close();
    require(out.good(), "cannot save output file");
}
struct snapshot {
    long rss_kib = -1, hwm_kib = -1, anon_kib = -1, file_kib = -1, swap_kib = -1;
    rusage usage{};
};
static snapshot sample() {
    snapshot s;
    require(getrusage(RUSAGE_SELF, &s.usage) == 0, "getrusage failed");
    std::ifstream status("/proc/self/status");
    require(status.good(), "cannot read /proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        std::istringstream in(line);
        std::string key;
        long value;
        if (!(in >> key >> value)) { continue; }
        if (key == "VmRSS:") { s.rss_kib = value; }
        if (key == "VmHWM:") { s.hwm_kib = value; }
        if (key == "RssAnon:") { s.anon_kib = value; }
        if (key == "RssFile:") { s.file_kib = value; }
        if (key == "VmSwap:") { s.swap_kib = value; }
    }
    return s;
}
static void write_snapshot(std::ostream & out, const snapshot & s) {
    out << "{\"rss_kib\":" << s.rss_kib << ",\"hwm_kib\":" << s.hwm_kib
        << ",\"anon_kib\":" << s.anon_kib << ",\"file_kib\":" << s.file_kib
        << ",\"swap_kib\":" << s.swap_kib << ",\"maxrss_kib\":" << s.usage.ru_maxrss
        << ",\"minor_faults\":" << s.usage.ru_minflt << ",\"major_faults\":" << s.usage.ru_majflt << '}';
}
static void write_fault_delta(std::ostream & out, const snapshot & a, const snapshot & b) {
    out << "{\"minor\":" << b.usage.ru_minflt - a.usage.ru_minflt
        << ",\"major\":" << b.usage.ru_majflt - a.usage.ru_majflt << '}';
}
static uint64_t logits_hash(const std::vector<float> & logits) {
    const auto * bytes = reinterpret_cast<const unsigned char *>(logits.data());
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < logits.size() * sizeof(float); ++i) {
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    }
    return hash;
}
static const float * finite_logits(llama_context * ctx, int32_t vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    require(logits != nullptr, "missing last-token logits");
    for (int32_t i = 0; i < vocab; ++i) { require(std::isfinite(logits[i]), "nonfinite logits"); }
    return logits;
}
static std::vector<float> capture_logits(llama_context * ctx, int32_t vocab, const std::string & path) {
    const float * data = finite_logits(ctx, vocab);
    std::vector<float> result(data, data + vocab);
    save_bytes(path, result.data(), result.size() * sizeof(float));
    return result;
}
static llama_token greedy(const float * logits, int32_t vocab) {
    return llama_token(std::max_element(logits, logits + vocab) - logits);
}
static bool is_expert(const std::string & name) {
    // Match the routed matrix kinds supported by the production loader. The CPU
    // override above intentionally stays identical to the original experiment.
    static const char * kinds[] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps",
                                  "ffn_gate_chexps", "ffn_up_chexps", "ffn_down_chexps"};
    for (const char * kind : kinds) {
        if (name.find(std::string(".") + kind + ".weight") != std::string::npos) { return true; }
    }
    return false;
}
struct mapping_report {
    size_t ordinary_bytes = 0, expert_bytes = 0, can_prefetch_bytes = 0, mapped_expert_bytes = 0;
    size_t ordinary_tensors = 0, expert_tensors = 0, mapped_expert_tensors = 0;
    std::string entries;
};
static mapping_report inspect_mapping(const llama_model & model) {
    mapping_report result;
    std::ostringstream entries;
    entries << '[';
    bool first = true;
    // Inspect metadata only: never dereference tensor data, advise mappings, or
    // change buffer/data pointers. Byte counts are tensor payloads, not map extents.
    for (const auto & entry : model.tensors_by_name) {
        const auto * tensor = entry.second;
        const size_t bytes = ggml_nbytes(tensor);
        const bool expert = is_expert(entry.first);
        const bool mapped = model.can_prefetch.count(tensor) != 0;
        if (expert) { result.expert_bytes += bytes; ++result.expert_tensors; }
        else { result.ordinary_bytes += bytes; ++result.ordinary_tensors; }
        if (mapped) { result.can_prefetch_bytes += bytes; }
        if (expert && mapped) { result.mapped_expert_bytes += bytes; ++result.mapped_expert_tensors; }
        if (!expert && !mapped) { continue; }
        if (!first) { entries << ','; }
        first = false;
        const bool host = tensor->buffer && ggml_backend_buffer_is_host(tensor->buffer);
        entries << "{\"name\":" << json_string(entry.first) << ",\"bytes\":" << bytes
                << ",\"expert\":" << (expert ? "true" : "false")
                << ",\"can_prefetch\":" << (mapped ? "true" : "false")
                << ",\"host_buffer\":" << (host ? "true" : "false") << '}';
        require(!expert || host, "expert weights must use CPU-accessible host buffers");
    }
    entries << ']';
    result.entries = entries.str();
    require(result.expert_tensors > 0, "model has no routed expert matrix weights");
    return result;
}
static void save_text(const llama_vocab * vocab, const std::vector<llama_token> & tokens, const std::string & path) {
    // Text is a convenience artifact; token IDs and full logits are authoritative.
    std::vector<char> text(4096);
    int32_t size = llama_detokenize(vocab, tokens.data(), int32_t(tokens.size()), text.data(), int32_t(text.size()), false, true);
    if (size < 0) {
        text.resize(size_t(-int64_t(size)));
        size = llama_detokenize(vocab, tokens.data(), int32_t(tokens.size()), text.data(), int32_t(text.size()), false, true);
    }
    require(size >= 0, "detokenization failed");
    save_bytes(path, text.data(), size_t(size));
}
int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s MODEL resident|mmap|mmap-eager|hybrid-off|hybrid-on PREFIX PP (PP must be 128)\n", argv[0]);
        return 2;
    }
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    bool backend_initialized = false;
    int result = 1;
    const auto process_start = clock_type::now();
    try {
        const std::string mode = argv[2], prefix = argv[3];
        require(mode == "resident" || mode == "mmap" || mode == "mmap-eager" ||
                mode == "hybrid-off" || mode == "hybrid-on", "unknown mode");
        size_t parsed = 0;
        const int pp = std::stoi(argv[4], &parsed);
        require(parsed == std::strlen(argv[4]) && pp == pp_tokens, "PP must be exactly 128");
        const bool hybrid = mode == "hybrid-off" || mode == "hybrid-on";
        ggml_backend_load_all();
        llama_backend_init();
        backend_initialized = true;
        bool has_spacemit = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            const auto reg = ggml_backend_dev_backend_reg(ggml_backend_dev_get(i));
            has_spacemit |= std::strcmp(ggml_backend_reg_name(reg), "SPACEMIT") == 0;
        }
        require(has_spacemit, "SpacemiT backend unavailable");
        auto params = llama_model_default_params();
        params.n_gpu_layers = 99;
        params.load_mode = (mode == "mmap" || mode == "mmap-eager") ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
        params.lazy_mode = mode == "mmap" ? LLAMA_LAZY_MODE_ON : LLAMA_LAZY_MODE_OFF;
        params.prefetch = hybrid;
        const llama_model_tensor_buft_override overrides[] = {
            {expert_override, ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
        params.tensor_buft_overrides = overrides;
        const snapshot before_load = sample();
        const auto load_begin = clock_type::now();
        model = llama_model_load_from_file(argv[1], params);
        const auto load_end = clock_type::now();
        require(model != nullptr, "model load failed");
        const snapshot after_load = sample();
        require(llama_model_n_layer(model) == 1, "benchmark requires a single-layer model");
        const auto mapping = inspect_mapping(*model);
        require(!hybrid || mapping.mapped_expert_tensors == mapping.expert_tensors,
                "hybrid loader did not mark all routed expert weights can_prefetch");
        std::fprintf(stderr, "LOADER can_prefetch=%zu ordinary_bytes=%zu expert_bytes=%zu mapped_expert_bytes=%zu\n",
                     model->can_prefetch.size(), mapping.ordinary_bytes, mapping.expert_bytes, mapping.mapped_expert_bytes);
        const auto * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        require(n_vocab > 227, "vocabulary does not contain input IDs 100..227");
        auto cparams = llama_context_default_params();
        cparams.n_ctx = 256;
        cparams.n_batch = pp_tokens;
        cparams.n_ubatch = pp_tokens;
        cparams.n_threads = 8;
        cparams.n_threads_batch = 8;
        cparams.prefetch = mode == "hybrid-on";
        const snapshot before_context = sample();
        const auto context_begin = clock_type::now();
        ctx = llama_init_from_model(model, cparams);
        const auto context_end = clock_type::now();
        require(ctx != nullptr, "context creation failed");
        const snapshot after_context = sample();
        std::vector<llama_token> tokens(pp_tokens);
        for (int i = 0; i < pp_tokens; ++i) { tokens[i] = 100 + i; }
        const snapshot before_first = sample();
        const auto first_begin = clock_type::now();
        require(llama_decode(ctx, llama_batch_get_one(tokens.data(), pp_tokens)) == 0, "first prefill failed");
        llama_synchronize(ctx);
        const auto first_end = clock_type::now();
        const snapshot after_first = sample();
        const auto first_logits = capture_logits(ctx, n_vocab, prefix + ".first.logits");
        const uint64_t first_hash = logits_hash(first_logits);
        std::vector<llama_token> generated;
        generated.reserve(decode_tokens);
        llama_token token = greedy(first_logits.data(), n_vocab);
        const snapshot before_decode = sample();
        const auto decode_begin = clock_type::now();
        for (int i = 0; i < decode_tokens; ++i) {
            generated.push_back(token); // These exact 16 greedy tokens are fed back.
            require(llama_decode(ctx, llama_batch_get_one(&token, 1)) == 0, "greedy decode failed");
            token = greedy(finite_logits(ctx, n_vocab), n_vocab);
        }
        llama_synchronize(ctx);
        const auto decode_end = clock_type::now();
        const snapshot after_decode = sample();
        const auto decode_logits = capture_logits(ctx, n_vocab, prefix + ".decode.logits");
        std::vector<double> warm_ms;
        std::vector<uint64_t> warm_hashes;
        std::vector<snapshot> before_warm, after_warm;
        bool repeat_equal = true;
        for (int r = 0; r < warm_repeats; ++r) {
            // Reset outside the timed prefill and restart token positions at zero.
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_synchronize(ctx);
            before_warm.push_back(sample());
            const auto begin = clock_type::now();
            require(llama_decode(ctx, llama_batch_get_one(tokens.data(), pp_tokens)) == 0, "warm prefill failed");
            llama_synchronize(ctx);
            const auto end = clock_type::now();
            after_warm.push_back(sample());
            warm_ms.push_back(ms(begin, end));
            const auto logits = capture_logits(ctx, n_vocab, prefix + ".warm" + std::to_string(r + 1) + ".logits");
            const uint64_t hash = logits_hash(logits);
            warm_hashes.push_back(hash);
            repeat_equal &= hash == first_hash &&
                std::memcmp(first_logits.data(), logits.data(), logits.size() * sizeof(float)) == 0;
            if (r + 1 == warm_repeats) { save_bytes(prefix + ".warm.logits", logits.data(), logits.size() * sizeof(float)); }
        }
        const auto measured_end = clock_type::now();
        const snapshot end = sample();
        try { save_text(vocab, generated, prefix + ".text"); }
        catch (const std::exception & error) { std::fprintf(stderr, "Optional text output: %s\n", error.what()); }
        std::ofstream out(prefix + ".json");
        require(out.good(), "cannot open result JSON");
        out << std::setprecision(17) << std::boolalpha;
        out << "{\n  \"model\": " << json_string(argv[1]) << ",\n  \"mode\": " << json_string(mode)
            << ",\n  \"pp_tokens\":128, \"decode_tokens\":16, \"warm_repeats\":3, \"n_ctx\":256,"
            << " \"n_batch\":128, \"n_ubatch\":128, \"n_threads\":8, \"n_threads_batch\":8, \"n_gpu_layers\":99, \"n_layers\":1"
            << ",\n  \"load_mode\": " << json_string(params.load_mode == LLAMA_LOAD_MODE_MMAP ? "MMAP" : "NONE")
            << ",\n  \"lazy_mode\": " << json_string(params.lazy_mode == LLAMA_LAZY_MODE_ON ? "ON" : "OFF")
            << ",\n  \"model_prefetch\": " << params.prefetch << ", \"context_prefetch\": " << cparams.prefetch
            << ",\n  \"cpu_tensor_override\": " << json_string(expert_override)
            << ",\n  \"ordinary_bytes\": " << mapping.ordinary_bytes << ", \"expert_bytes\": " << mapping.expert_bytes
            << ",\n  \"ordinary_tensors\": " << mapping.ordinary_tensors << ", \"expert_tensors\": " << mapping.expert_tensors
            << ",\n  \"can_prefetch_tensors\": " << model->can_prefetch.size() << ", \"can_prefetch_bytes\": " << mapping.can_prefetch_bytes
            << ",\n  \"mapped_expert_tensors\": " << mapping.mapped_expert_tensors << ", \"mapped_expert_bytes\": " << mapping.mapped_expert_bytes
            << ",\n  \"loader_expert_mapping\": " << mapping.entries
            << ",\n  \"load_ms\": " << ms(load_begin, load_end) << ", \"context_ms\": " << ms(context_begin, context_end)
            << ",\n  \"first_pp_ms\": " << ms(first_begin, first_end)
            << ",\n  \"warm_pp_ms\": " << (warm_ms[0] + warm_ms[1] + warm_ms[2]) / warm_repeats
            << ", \"warm_pp_ms_each\": [" << warm_ms[0] << ',' << warm_ms[1] << ',' << warm_ms[2] << ']'
            << ",\n  \"decode16_ms\": " << ms(decode_begin, decode_end)
            << ",\n  \"total_measured_ms\": " << ms(process_start, measured_end)
            << ",\n  \"load_plus_first_ms\": " << ms(load_begin, first_end)
            << ",\n  \"maxrss_kib\": " << end.usage.ru_maxrss
            << ", \"minor_faults\": " << end.usage.ru_minflt << ", \"major_faults\": " << end.usage.ru_majflt
            << ",\n  \"snapshots\": {\"before_load\":";
        write_snapshot(out, before_load);
        out << ",\"after_load\":"; write_snapshot(out, after_load);
        out << ",\"after_context\":"; write_snapshot(out, after_context);
        out << ",\"after_first_pp\":"; write_snapshot(out, after_first);
        out << ",\"after_decode16\":"; write_snapshot(out, after_decode);
        for (int r = 0; r < warm_repeats; ++r) {
            out << ",\"after_warm" << r + 1 << "\":"; write_snapshot(out, after_warm[r]);
        }
        out << ",\"end\":"; write_snapshot(out, end);
        out << "},\n  \"phase_faults\": {\"load\":"; write_fault_delta(out, before_load, after_load);
        out << ",\"context\":"; write_fault_delta(out, before_context, after_context);
        out << ",\"first_pp\":"; write_fault_delta(out, before_first, after_first);
        out << ",\"decode16\":"; write_fault_delta(out, before_decode, after_decode);
        for (int r = 0; r < warm_repeats; ++r) {
            out << ",\"warm" << r + 1 << "\":"; write_fault_delta(out, before_warm[r], after_warm[r]);
        }
        out << "},\n  \"logits_format\":\"native-float32\", \"logits_count\":" << n_vocab
            << ", \"logits_bytes\":" << first_logits.size() * sizeof(float)
            << ", \"hash_algorithm\":\"fnv1a64-native-bytes\""
            << ",\n  \"first_logits_hash\": " << json_string(std::to_string(first_hash))
            << ", \"decode_logits_hash\": " << json_string(std::to_string(logits_hash(decode_logits)))
            << ",\n  \"warm_logits_hash\": " << json_string(std::to_string(warm_hashes.back()))
            << ", \"warm_logits_hashes\": [";
        for (size_t i = 0; i < warm_hashes.size(); ++i) {
            if (i) { out << ','; }
            out << json_string(std::to_string(warm_hashes[i]));
        }
        out << "],\n  \"all_logits_finite\":true, \"repeat_first_warm_bytes_equal\":" << repeat_equal
            << ", \"repeat_first_warm_hash_equal\":" << std::all_of(warm_hashes.begin(), warm_hashes.end(),
                [first_hash](uint64_t hash) { return hash == first_hash; })
            << ",\n  \"input_token_ids\": [";
        for (size_t i = 0; i < tokens.size(); ++i) { if (i) { out << ','; } out << tokens[i]; }
        out << "],\n  \"generated_token_ids\": [";
        for (size_t i = 0; i < generated.size(); ++i) { if (i) { out << ','; } out << generated[i]; }
        out << "],\n  \"next_token_id\": " << token << "\n}\n";
        out.close();
        require(out.good(), "cannot save result JSON");
        require(repeat_equal, "repeat prefill logits changed (see saved logits and JSON)");
        std::fprintf(stderr, "PASS mode=%s first_pp_ms=%.3f warm_pp_ms=%.3f decode16_ms=%.3f JSON=%s.json\n",
                     mode.c_str(), ms(first_begin, first_end), (warm_ms[0] + warm_ms[1] + warm_ms[2]) / warm_repeats,
                     ms(decode_begin, decode_end), prefix.c_str());
        result = 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
    }
    if (ctx) { llama_free(ctx); }
    if (model) { llama_model_free(model); }
    if (backend_initialized) { llama_backend_free(); }
    return result;
}
