// SPDX-License-Identifier: MIT
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "spacemit-transfer.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

namespace {
std::string trim(const std::string & s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) { return {}; }
    return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

bool read_cpu_times(std::vector<spacemit_transfer_cpu_time> & times, std::vector<bool> & present) {
    std::ifstream stat("/proc/stat");
    if (!stat) { return false; }
    times.assign(CPU_SETSIZE, {});
    present.assign(CPU_SETSIZE, false);
    std::string line;
    while (std::getline(stat, line)) {
        std::istringstream fields(line);
        std::string label;
        fields >> label;
        if (label.size() <= 3 || label.compare(0, 3, "cpu") != 0) { continue; }
        size_t consumed = 0;
        int cpu;
        try {
            cpu = std::stoi(label.substr(3), &consumed);
        } catch (...) { continue; }
        if (consumed != label.size() - 3 || cpu < 0 || cpu >= CPU_SETSIZE) { continue; }
        uint64_t user, nice, system, idle, iowait, irq, softirq, steal;
        if (!(fields >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal)) { continue; }
        // guest/guest_nice are already included in user/nice; do not double count.
        // iowait is not proof of an idle CPU and is treated as unavailable time.
        times[cpu] = {user + nice + system + iowait + irq + softirq + steal, idle};
        present[cpu] = true;
    }
    return !stat.bad();
}
} // namespace
#endif

bool spacemit_transfer_provider_available_cpus(std::vector<int> & cpus) {
    cpus.clear();
#if defined(__linux__)
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) { return false; }
    // First discover candidates; then exclude busy CPUs using /proc/stat deltas.
    // Do not infer core type from CPU-number ranges or vector length.
    std::ifstream online_file("/sys/devices/system/cpu/online");
    if (!online_file) { return false; }
    std::string online_text;
    std::getline(online_file, online_text);
    cpu_set_t online;
    CPU_ZERO(&online);
    size_t pos = 0;
    while (pos < online_text.size()) {
        size_t end = online_text.find(',', pos);
        if (end == std::string::npos) { end = online_text.size(); }
        size_t dash = online_text.find('-', pos);
        if (dash >= end) { dash = std::string::npos; }
        try {
            const std::string first_text = online_text.substr(pos, (dash == std::string::npos ? end : dash) - pos);
            const std::string last_text = dash == std::string::npos ? first_text : online_text.substr(dash + 1, end - dash - 1);
            size_t first_end = 0, last_end = 0;
            int first = std::stoi(first_text, &first_end);
            int last = std::stoi(last_text, &last_end);
            if (first_end != first_text.size() || last_end != last_text.size()) { return false; }
            if (first < 0 || last < first || last >= CPU_SETSIZE) { return false; }
            for (int id = first; id <= last; ++id) { CPU_SET(id, &online); }
        } catch (...) { return false; }
        pos = end + 1;
    }
    std::ifstream info("/proc/cpuinfo");
    if (!info) { return false; }
    std::string line, model;
    int cpu = -1;
    auto flush = [&] {
        if (cpu >= 0 && cpu < CPU_SETSIZE && model == "Spacemit(R) X100" && CPU_ISSET(cpu, &online) && CPU_ISSET(cpu, &allowed)) {
            cpus.push_back(cpu);
        }
        cpu = -1;
        model.clear();
    };
    while (std::getline(info, line)) {
        if (trim(line).empty()) { flush(); continue; }
        const auto colon = line.find(':');
        if (colon == std::string::npos) { continue; }
        const std::string key = trim(line.substr(0, colon));
        const std::string value = trim(line.substr(colon + 1));
        if (key == "processor") {
            size_t consumed = 0;
            try {
                cpu = std::stoi(value, &consumed);
                if (consumed != value.size()) { cpu = -1; }
            } catch (...) { cpu = -1; }
        } else if (key == "model name") {
            model = value;
        }
    }
    flush();
    if (cpus.empty()) { return false; }
    std::vector<spacemit_transfer_cpu_time> before, after;
    std::vector<bool> had_before, had_after;
    if (!read_cpu_times(before, had_before)) { cpus.clear(); return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (!read_cpu_times(after, had_after)) { cpus.clear(); return false; }
    cpus.erase(std::remove_if(cpus.begin(), cpus.end(), [&](int id) {
        return !had_before[id] || !had_after[id] || !spacemit_transfer_cpu_is_available(before[id], after[id]);
    }), cpus.end());
    return !cpus.empty();
#else
    return false;
#endif
}

bool spacemit_transfer_provider_bind(const std::vector<int> & cpus) {
#if defined(__linux__)
    cpu_set_t requested, actual;
    CPU_ZERO(&requested);
    for (int cpu : cpus) {
        if (cpu < 0 || cpu >= CPU_SETSIZE) { return false; }
        CPU_SET(cpu, &requested);
    }
    // Workers run only on the sampled available X100 set. Other processes can
    // still start work here later: this mask is NOT an exclusive reservation.
    if (cpus.empty() || pthread_setaffinity_np(pthread_self(), sizeof(requested), &requested) != 0 ||
        pthread_getaffinity_np(pthread_self(), sizeof(actual), &actual) != 0) {
        return false;
    }
    const int current = sched_getcpu();
    return CPU_EQUAL(&actual, &requested) && current >= 0 && current < CPU_SETSIZE && CPU_ISSET(current, &actual);
#else
    (void) cpus;
    return false;
#endif
}

bool spacemit_transfer_provider_copy(void * dst, const void * src, size_t size) {
    // Only ordinary CPU-accessible memory. No TCM mapping/coherency assumption.
    std::memmove(dst, src, size);
    return true;
}

bool spacemit_transfer_provider_prefetch(const void * src, size_t size) {
#if defined(__linux__)
    // Initialized on a worker, never on the submitting A100 thread.
    static const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) { return false; }
    const size_t page = static_cast<size_t>(page_size);
    const auto * bytes = static_cast<const volatile unsigned char *>(src);
    size_t offset = 0;
    while (offset < size) {
        // Volatile preserves the read even though only its page-fault side
        // effect is needed. Never read outside an unaligned caller range.
        const unsigned char value = bytes[offset];
        (void) value;
        const size_t step = page - ((reinterpret_cast<uintptr_t>(src) + offset) % page);
        if (step >= size - offset) { break; }
        offset += step;
    }
    return true;
#else
    (void) src;
    (void) size;
    return false;
#endif
}
