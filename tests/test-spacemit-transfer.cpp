// SPDX-License-Identifier: MIT
#include "spacemit-transfer.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); std::abort(); } } while (0)
using clock_type = std::chrono::steady_clock;

#ifndef SPACEMIT_TRANSFER_TEST_REAL
namespace {
std::atomic<bool> unavailable{false}, bind_failed{false}, copy_failed{false};
std::atomic<int> copies{0}, bindings{0};
std::mutex gate_mutex;
std::condition_variable gate_cv;
bool blocked = false, entered = false;
const auto main_thread = std::this_thread::get_id();
}

bool spacemit_transfer_provider_available_cpus(std::vector<int> & cpus) {
    cpus = unavailable ? std::vector<int>{} : std::vector<int>{0, 1, 2, 3};
    return !cpus.empty();
}
bool spacemit_transfer_provider_bind(const std::vector<int> &) {
    CHECK(std::this_thread::get_id() != main_thread);
    ++bindings;
    return !bind_failed;
}
bool spacemit_transfer_provider_copy(void * dst, const void * src, size_t size) {
    CHECK(std::this_thread::get_id() != main_thread);
    ++copies;
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        entered = true;
        gate_cv.notify_all();
        gate_cv.wait(lock, [] { return !blocked; });
    }
    if (copy_failed) { return false; }
    std::memmove(dst, src, size);
    return true;
}
#endif

static void submit(ggml_spacemit_transfer_t transfer, const ggml_spacemit_transfer_request & request,
                   ggml_spacemit_transfer_event_t event) {
    const auto deadline = clock_type::now() + std::chrono::seconds(10);
    for (;;) {
        const auto status = ggml_spacemit_transfer_submit(transfer, &request, event);
        if (status == GGML_SPACEMIT_TRANSFER_OK) { return; }
        CHECK(status == GGML_SPACEMIT_TRANSFER_BUSY);
        CHECK(clock_type::now() < deadline);
        std::this_thread::yield();
    }
}

static ggml_spacemit_transfer_t create(size_t workers = 1, size_t capacity = 8) {
    ggml_spacemit_transfer_config config{workers, capacity};
    ggml_spacemit_transfer_status status = GGML_SPACEMIT_TRANSFER_ERROR;
    auto transfer = ggml_spacemit_transfer_create(&config, &status);
    CHECK(transfer && status == GGML_SPACEMIT_TRANSFER_OK);
    return transfer;
}

static void basic() {
    auto transfer = create();
    auto event = ggml_spacemit_transfer_event_create();
    CHECK(event);
    CHECK(ggml_spacemit_transfer_query(event) == GGML_SPACEMIT_TRANSFER_IDLE);
    CHECK(ggml_spacemit_transfer_wait(event) == GGML_SPACEMIT_TRANSFER_INVALID);
    CHECK(ggml_spacemit_transfer_wait(nullptr) == GGML_SPACEMIT_TRANSFER_INVALID);
    char src[] = "asynchronous transfer", dst[sizeof(src)]{};
    ggml_spacemit_transfer_request request{dst, src, sizeof(src)};
    CHECK(ggml_spacemit_transfer_submit(nullptr, &request, event) == GGML_SPACEMIT_TRANSFER_INVALID);
    CHECK(ggml_spacemit_transfer_submit(transfer, nullptr, event) == GGML_SPACEMIT_TRANSFER_INVALID);
    CHECK(ggml_spacemit_transfer_submit(transfer, &request, nullptr) == GGML_SPACEMIT_TRANSFER_INVALID);
    ggml_spacemit_transfer_request bad{nullptr, src, 1};
    CHECK(ggml_spacemit_transfer_submit(transfer, &bad, event) == GGML_SPACEMIT_TRANSFER_INVALID);
    submit(transfer, request, event);
    CHECK(ggml_spacemit_transfer_wait(event) == GGML_SPACEMIT_TRANSFER_OK);
    CHECK(ggml_spacemit_transfer_query(event) == GGML_SPACEMIT_TRANSFER_COMPLETE);
    CHECK(std::memcmp(src, dst, sizeof(src)) == 0);
    // Reuse event, including a zero-byte event and overlapping buffers.
    submit(transfer, {nullptr, nullptr, 0}, event);
    CHECK(ggml_spacemit_transfer_wait(event) == GGML_SPACEMIT_TRANSFER_OK);
    char overlap[] = "0123456789";
    submit(transfer, {overlap + 1, overlap, 9}, event);
    CHECK(ggml_spacemit_transfer_wait(event) == GGML_SPACEMIT_TRANSFER_OK);
    CHECK(std::memcmp(overlap, "0012345678", 10) == 0);
    // Pending event destruction waits for the actual copy.
    std::memset(dst, 0, sizeof(dst));
    submit(transfer, request, event);
    ggml_spacemit_transfer_event_free(event);
    CHECK(std::memcmp(src, dst, sizeof(src)) == 0);
    ggml_spacemit_transfer_drain(transfer);
    ggml_spacemit_transfer_free(transfer);
    ggml_spacemit_transfer_free(nullptr);
    ggml_spacemit_transfer_event_free(nullptr);
    CHECK(spacemit_transfer_get_proc_address("ggml_spacemit_transfer_submit") != nullptr);
    CHECK(spacemit_transfer_get_proc_address("not_a_transfer_function") == nullptr);
    std::puts("PASS basic/reuse/zero/overlap/lifetime/proc-address");
}

#ifndef SPACEMIT_TRANSFER_TEST_REAL
static void bounded_queue() {
    auto transfer = create(1, 1);
    auto a = ggml_spacemit_transfer_event_create();
    auto b = ggml_spacemit_transfer_event_create();
    auto c = ggml_spacemit_transfer_event_create();
    CHECK(a && b && c);
    char src = 'X', dst[3]{};
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        blocked = true;
        entered = false;
    }
    submit(transfer, {dst, &src, 1}, a);
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        CHECK(gate_cv.wait_for(lock, std::chrono::seconds(10), [] { return entered; }));
    }
    // The first copy is blocked in the provider, not on the submitter.
    CHECK(ggml_spacemit_transfer_query(a) == GGML_SPACEMIT_TRANSFER_PENDING);
    ggml_spacemit_transfer_request request{dst + 1, &src, 1};
    CHECK(ggml_spacemit_transfer_submit(transfer, &request, a) == GGML_SPACEMIT_TRANSFER_BUSY);
    submit(transfer, request, b);
    request.dst = dst + 2;
    CHECK(ggml_spacemit_transfer_submit(transfer, &request, c) == GGML_SPACEMIT_TRANSFER_BUSY);
    CHECK(ggml_spacemit_transfer_query(c) == GGML_SPACEMIT_TRANSFER_IDLE);
    CHECK(dst[2] == 0);
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        blocked = false;
        gate_cv.notify_all();
    }
    CHECK(ggml_spacemit_transfer_wait(a) == GGML_SPACEMIT_TRANSFER_OK);
    CHECK(ggml_spacemit_transfer_wait(b) == GGML_SPACEMIT_TRANSFER_OK);
    submit(transfer, request, c);
    CHECK(ggml_spacemit_transfer_wait(c) == GGML_SPACEMIT_TRANSFER_OK);
    CHECK(std::memcmp(dst, "XXX", 3) == 0);
    ggml_spacemit_transfer_event_free(a);
    ggml_spacemit_transfer_event_free(b);
    ggml_spacemit_transfer_event_free(c);
    ggml_spacemit_transfer_free(transfer);
    std::puts("PASS nonblocking submission/full queue/pending-event protection");
}

static void failures() {
    ggml_spacemit_transfer_status status;
    unavailable = true;
    CHECK(!ggml_spacemit_transfer_create(nullptr, &status));
    CHECK(status == GGML_SPACEMIT_TRANSFER_UNAVAILABLE);
    unavailable = false;
    bind_failed = true;
    CHECK(!ggml_spacemit_transfer_create(nullptr, &status));
    CHECK(status == GGML_SPACEMIT_TRANSFER_UNAVAILABLE);
    bind_failed = false;
    ggml_spacemit_transfer_config too_many{5, 1};
    CHECK(!ggml_spacemit_transfer_create(&too_many, &status));
    CHECK(status == GGML_SPACEMIT_TRANSFER_INVALID);
    auto transfer = create();
    auto event = ggml_spacemit_transfer_event_create();
    CHECK(event);
    char src = 'x', dst = 0;
    copy_failed = true;
    submit(transfer, {&dst, &src, 1}, event);
    CHECK(ggml_spacemit_transfer_wait(event) == GGML_SPACEMIT_TRANSFER_ERROR);
    CHECK(ggml_spacemit_transfer_query(event) == GGML_SPACEMIT_TRANSFER_FAILED);
    CHECK(dst == 0);
    copy_failed = false;
    submit(transfer, {&dst, &src, 1}, event);
    CHECK(ggml_spacemit_transfer_wait(event) == GGML_SPACEMIT_TRANSFER_OK);
    CHECK(dst == src);
    ggml_spacemit_transfer_event_free(event);
    ggml_spacemit_transfer_free(transfer);
    std::puts("PASS unavailable/binding failure/copy failure/recovery");
}
#endif

static void concurrent_and_shutdown(size_t n_workers) {
    auto transfer = create(n_workers, 4);
    constexpr size_t count = 128, size = 1024;
    std::vector<unsigned char> src(count * size), dst(count * size, 0);
    std::vector<ggml_spacemit_transfer_event_t> events(count);
    for (size_t i = 0; i < src.size(); ++i) { src[i] = static_cast<unsigned char>(i * 17 + i / size); }
    for (auto & event : events) { event = ggml_spacemit_transfer_event_create(); CHECK(event); }
    std::vector<std::thread> producers;
    for (size_t p = 0; p < 4; ++p) {
        producers.emplace_back([&, p] {
            for (size_t i = p; i < count; i += 4) {
                submit(transfer, {dst.data() + i * size, src.data() + i * size, size}, events[i]);
            }
        });
    }
    for (auto & producer : producers) { producer.join(); }
    // Destroy without explicit drain: all accepted jobs must finish.
    ggml_spacemit_transfer_free(transfer);
    for (auto event : events) {
        CHECK(ggml_spacemit_transfer_wait(event) == GGML_SPACEMIT_TRANSFER_OK);
        ggml_spacemit_transfer_event_free(event);
    }
    CHECK(src == dst);
    std::printf("PASS concurrent producers/draining destruction (%zu workers)\n", n_workers);
}

static void cpu_availability() {
    using time = spacemit_transfer_cpu_time;
    CHECK(spacemit_transfer_cpu_is_available(time{100, 200}, time{100, 210})); // idle
    CHECK(spacemit_transfer_cpu_is_available(time{100, 200}, time{101, 209})); // 10%
    CHECK(!spacemit_transfer_cpu_is_available(time{100, 200}, time{102, 208})); // busy
    CHECK(!spacemit_transfer_cpu_is_available(time{100, 200}, time{110, 200})); // saturated
    CHECK(!spacemit_transfer_cpu_is_available(time{100, 200}, time{100, 200})); // no sample
    CHECK(!spacemit_transfer_cpu_is_available(time{100, 200}, time{99, 210})); // counter reset
    CHECK(!spacemit_transfer_cpu_is_available(time{100, 200}, time{101, 199}));
    CHECK(!spacemit_transfer_cpu_is_available(time{100, 200}, time{101, 208})); // just over 10%
    CHECK(spacemit_transfer_cpu_is_available(time{0, 0}, time{UINT64_MAX / 10, UINT64_MAX}));
    std::puts("PASS CPU availability: idle/busy/threshold/missing/reset/overflow");
}

int main(int argc, char ** argv) {
    cpu_availability();
#ifdef SPACEMIT_TRANSFER_TEST_REAL
    if (argc == 2 && std::strcmp(argv[1], "--expect-unavailable") == 0) {
        ggml_spacemit_transfer_status status;
        auto transfer = ggml_spacemit_transfer_create(nullptr, &status);
        CHECK(!transfer && status == GGML_SPACEMIT_TRANSFER_UNAVAILABLE);
        std::puts("PASS no available CPUs: service returns UNAVAILABLE");
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--probe-cpus") == 0) {
        std::vector<int> cpus;
        spacemit_transfer_provider_available_cpus(cpus);
        std::printf("AVAILABLE");
        for (int cpu : cpus) { std::printf(" %d", cpu); }
        std::puts("");
        return 0;
    }
#else
    (void) argc;
    (void) argv;
#endif
#ifdef SPACEMIT_TRANSFER_TEST_REAL
    std::vector<int> cpus;
    CHECK(spacemit_transfer_provider_available_cpus(cpus));
    std::printf("X100 provider sampled available CPUs:");
    for (int cpu : cpus) { std::printf(" %d", cpu); }
    std::puts("");
    const size_t workers = cpus.size() > 1 ? 2 : 1;
#else
    const size_t workers = 2;
    failures();
    bounded_queue();
#endif
    basic();
    concurrent_and_shutdown(1);
    concurrent_and_shutdown(workers);
    std::puts("ALL TRANSFER TESTS PASSED");
}
