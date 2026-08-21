#include "multi-asr-common.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

// Parse one segment: either a single core "8" or a dash range "8-11".
// Returns true on success and sets the corresponding mask bits.
static bool parse_one_segment(const std::string & seg, bool (&mask)[GGML_MAX_N_THREADS]) {
    if (seg.empty()) {
        return false;
    }

    const size_t dash = seg.find('-');
    if (dash == std::string::npos) {
        // single cpu index
        char *     endp = nullptr;
        const long cpu  = std::strtol(seg.c_str(), &endp, 10);
        if (endp == seg.c_str() || *endp != '\0' || cpu < 0 || cpu >= GGML_MAX_N_THREADS) {
            return false;
        }
        mask[cpu] = true;
        return true;
    }

    // dash range "lo-hi"
    const std::string lo_s = seg.substr(0, dash);
    const std::string hi_s = seg.substr(dash + 1);
    char *            e1   = nullptr;
    char *            e2   = nullptr;
    const long        lo   = std::strtol(lo_s.c_str(), &e1, 10);
    const long        hi   = std::strtol(hi_s.c_str(), &e2, 10);
    if (e1 == lo_s.c_str() || *e1 != '\0' || e2 == hi_s.c_str() || *e2 != '\0') {
        return false;
    }
    if (lo < 0 || hi < 0 || lo >= GGML_MAX_N_THREADS || hi >= GGML_MAX_N_THREADS || lo > hi) {
        return false;
    }
    for (long c = lo; c <= hi; ++c) {
        mask[c] = true;
    }
    return true;
}

int multi_asr_parse_cpu_range(const std::string & range, bool (&mask)[GGML_MAX_N_THREADS]) {
    std::memset(mask, 0, sizeof(bool) * GGML_MAX_N_THREADS);
    if (range.empty()) {
        return 0;
    }

    // Split on commas; each part is a single cpu or a dash range.
    std::stringstream ss(range);
    std::string       part;
    while (std::getline(ss, part, ',')) {
        // trim ASCII whitespace
        size_t b = 0;
        size_t e = part.size();
        while (b < e && std::isspace((unsigned char) part[b])) {
            ++b;
        }
        while (e > b && std::isspace((unsigned char) part[e - 1])) {
            --e;
        }
        const std::string seg = part.substr(b, e - b);
        if (seg.empty()) {
            continue;
        }
        if (!parse_one_segment(seg, mask)) {
            return -1;
        }
    }

    int count = 0;
    for (int i = 0; i < GGML_MAX_N_THREADS; ++i) {
        count += mask[i] ? 1 : 0;
    }
    return count;
}

std::string multi_asr_cpu_mask_to_string(const bool (&mask)[GGML_MAX_N_THREADS]) {
    std::string out;
    for (int i = 0; i < GGML_MAX_N_THREADS; ++i) {
        if (mask[i]) {
            if (!out.empty()) {
                out += ",";
            }
            out += std::to_string(i);
        }
    }
    return out;
}
