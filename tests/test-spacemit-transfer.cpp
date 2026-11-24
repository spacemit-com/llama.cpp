// SPDX-License-Identifier: MIT
#include "spacemit-transfer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <mutex>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#include <fcntl.h>
#include <cerrno>
#include <unistd.h>
#endif

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); std::abort(); } } while (0)
using clock_type = std::chrono::steady_clock;

#ifndef SPACEMIT_TRANSFER_TEST_REAL
namespace {
std::atomic<bool> unavailable{false}, bind_failed{false}, copy_failed{false};
std::atomic<int> copies{0}, bindings{0};
std::atomic<size_t> warmed_bytes{0};
std::mutex gate_mutex;
std::condition_variable gate_cv;
bool blocked = false, entered = false;
std::atomic<unsigned> cpu_binding_mask{0};
const auto main_thread = std::this_thread::get_id();
}

bool spacemit_transfer_provider_available_cpus(std::vector<int> & cpus) {
    cpus = unavailable ? std::vector<int>{} : std::vector<int>{0, 1, 2, 3};
    return !cpus.empty();
}
bool spacemit_transfer_provider_bind(const std::vector<int> & cpus) {
    CHECK(std::this_thread::get_id() != main_thread);
    CHECK(cpus.size() == 1 && cpus[0] >= 0 && cpus[0] < 4);
    cpu_binding_mask.fetch_or(1u << cpus[0]);
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
bool spacemit_transfer_provider_prefetch(const void * src, size_t size) {
    CHECK(std::this_thread::get_id() != main_thread);
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        entered = true;
        gate_cv.notify_all();
        gate_cv.wait(lock, [] { return !blocked; });
    }
    if (copy_failed) { return false; }
    const auto * bytes = static_cast<const volatile unsigned char *>(src);
    for (size_t i = 0; i < size; ++i) { const unsigned char value = bytes[i]; (void) value; }
    warmed_bytes += size;
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

static ggml_spacemit_transfer_status cancel(ggml_spacemit_transfer_event_t event) {
    const auto deadline = clock_type::now() + std::chrono::seconds(10);
    for (;;) {
        const auto status = ggml_spacemit_transfer_cancel(event);
        if (status != GGML_SPACEMIT_TRANSFER_BUSY) { return status; }
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
#ifndef SPACEMIT_TRANSFER_TEST_REAL
    cpu_binding_mask.store(0);
    auto bound_transfer = create(4, 4);
    CHECK(cpu_binding_mask.load() == 0xf); // four distinct singleton CPU bindings
    ggml_spacemit_transfer_free(bound_transfer);
#endif
    auto transfer = create();
    auto event = ggml_spacemit_transfer_event_create();
    CHECK(event);
    CHECK(ggml_spacemit_transfer_query(event) == GGML_SPACEMIT_TRANSFER_IDLE);
    CHECK(ggml_spacemit_transfer_wait(event) == GGML_SPACEMIT_TRANSFER_INVALID);
    CHECK(ggml_spacemit_transfer_wait(nullptr) == GGML_SPACEMIT_TRANSFER_INVALID);
    CHECK(cancel(nullptr) == GGML_SPACEMIT_TRANSFER_INVALID);
    CHECK(cancel(event) == GGML_SPACEMIT_TRANSFER_INVALID);
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
    CHECK(cancel(event) == GGML_SPACEMIT_TRANSFER_INVALID);
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
    CHECK(spacemit_transfer_get_proc_address("ggml_spacemit_transfer_cancel") != nullptr);
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

static void cancellation() {
    auto transfer = create(1, 2);
    auto active_event = ggml_spacemit_transfer_event_create();
    auto queued_event = ggml_spacemit_transfer_event_create();
    auto normal_event = ggml_spacemit_transfer_event_create();
    CHECK(active_event && queued_event && normal_event);
    constexpr size_t chunk = 64 * 1024;
    std::vector<unsigned char> src(3 * chunk + 37, 0x5a), dst(src.size(), 0);
    unsigned char queued_dst = 0, normal_dst = 0;
    const int before = copies.load();
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        blocked = true;
        entered = false;
    }
    submit(transfer, {dst.data(), src.data(), src.size()}, active_event);
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        CHECK(gate_cv.wait_for(lock, std::chrono::seconds(10), [] { return entered; }));
    }
    submit(transfer, {&queued_dst, src.data(), 1}, queued_event);
    submit(transfer, {&normal_dst, src.data(), 1}, normal_event);
    CHECK(cancel(active_event) == GGML_SPACEMIT_TRANSFER_OK);
    CHECK(cancel(queued_event) == GGML_SPACEMIT_TRANSFER_OK);
    CHECK(cancel(queued_event) == GGML_SPACEMIT_TRANSFER_OK); // idempotent
    CHECK(ggml_spacemit_transfer_query(active_event) == GGML_SPACEMIT_TRANSFER_PENDING);
    CHECK(ggml_spacemit_transfer_query(queued_event) == GGML_SPACEMIT_TRANSFER_PENDING);
    // Cancellation cannot release a buffer/event while a provider still uses it.
    auto freeing = std::async(std::launch::async, [=] {
        ggml_spacemit_transfer_event_free(active_event);
    });
    CHECK(freeing.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        blocked = false;
        gate_cv.notify_all();
    }
    CHECK(freeing.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    freeing.get();
    CHECK(ggml_spacemit_transfer_wait(queued_event) == GGML_SPACEMIT_TRANSFER_CANCELLED);
    CHECK(ggml_spacemit_transfer_query(queued_event) == GGML_SPACEMIT_TRANSFER_CANCELLED_STATE);
    CHECK(cancel(queued_event) == GGML_SPACEMIT_TRANSFER_OK);
    CHECK(ggml_spacemit_transfer_wait(normal_event) == GGML_SPACEMIT_TRANSFER_OK);
    ggml_spacemit_transfer_drain(transfer);
    CHECK(copies == before + 2); // one active chunk + ordinary queued request
    CHECK(queued_dst == 0 && normal_dst == src[0]);
    for (size_t i = 0; i < dst.size(); ++i) { CHECK(dst[i] == (i < chunk ? src[i] : 0)); }
    // A fresh submission must clear the old request's cancellation flag.
    submit(transfer, {dst.data(), src.data(), src.size()}, queued_event);
    CHECK(ggml_spacemit_transfer_wait(queued_event) == GGML_SPACEMIT_TRANSFER_OK);
    CHECK(src == dst);
    ggml_spacemit_transfer_event_free(queued_event);
    ggml_spacemit_transfer_event_free(normal_event);
    ggml_spacemit_transfer_free(transfer);
    std::puts("PASS queued/active cancellation, chunk stop, safe destruction, unrelated work and reuse");
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
    CHECK(cancel(event) == GGML_SPACEMIT_TRANSFER_INVALID);
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

static void chunked_overlap() {
    auto transfer = create();
    auto event = ggml_spacemit_transfer_event_create();
    CHECK(event);
    const size_t bytes = 3 * 64 * 1024 + 117;
    for (size_t shift : {size_t(13), size_t(70001)}) {
        for (bool backwards : {false, true}) {
            std::vector<unsigned char> actual(bytes + shift), expected;
            for (size_t i = 0; i < actual.size(); ++i) { actual[i] = static_cast<unsigned char>(i * 17 + i / 251); }
            expected = actual;
            const size_t dst = backwards ? shift : 0, src = backwards ? 0 : shift;
            std::memmove(expected.data() + dst, expected.data() + src, bytes);
            submit(transfer, {actual.data() + dst, actual.data() + src, bytes}, event);
            CHECK(ggml_spacemit_transfer_wait(event) == GGML_SPACEMIT_TRANSFER_OK);
            CHECK(actual == expected);
        }
    }
    ggml_spacemit_transfer_event_free(event);
    ggml_spacemit_transfer_free(transfer);
    std::puts("PASS chunked memmove overlap in both directions");
}

static void cancellation_races() {
    auto transfer = create(1, 1);
    auto event = ggml_spacemit_transfer_event_create();
    CHECK(event);
    std::vector<unsigned char> src(1024 * 1024 + 13, 0xa5), dst(src.size(), 0);
    for (size_t i = 0; i < 64; ++i) {
        // Include zero-byte requests, cancellation during copying and late cancellation.
        const size_t bytes = i % 3 == 0 ? 0 : src.size();
        submit(transfer, {dst.data(), src.data(), bytes}, event);
        if (i % 3 == 1) { std::this_thread::yield(); }
        const auto status = cancel(event);
        const auto result = ggml_spacemit_transfer_wait(event);
        if (status == GGML_SPACEMIT_TRANSFER_OK) {
            CHECK(result == GGML_SPACEMIT_TRANSFER_CANCELLED);
            CHECK(ggml_spacemit_transfer_query(event) == GGML_SPACEMIT_TRANSFER_CANCELLED_STATE);
        } else {
            CHECK(status == GGML_SPACEMIT_TRANSFER_INVALID);
            CHECK(result == GGML_SPACEMIT_TRANSFER_OK);
            if (bytes) { CHECK(src == dst); }
        }
    }
    submit(transfer, {dst.data(), src.data(), src.size()}, event);
    CHECK(ggml_spacemit_transfer_wait(event) == GGML_SPACEMIT_TRANSFER_OK);
    CHECK(src == dst);
    ggml_spacemit_transfer_event_free(event);
    ggml_spacemit_transfer_free(transfer);
    std::puts("PASS cancel/completion races, zero-byte cancellation and repeated event reuse");
}

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

static void page_prefetch() {
    auto transfer = create(1, 4);
    auto event = ggml_spacemit_transfer_event_create();
    std::vector<unsigned char> source(3 * 64 * 1024 + 123, 0xa5);
    const auto original = source;
    auto enqueue_prefetch = [&](const void * src, size_t size) {
        const auto deadline = clock_type::now() + std::chrono::seconds(10);
        for (;;) {
            const auto status = ggml_spacemit_transfer_prefetch(transfer, src, size, event);
            if (status == GGML_SPACEMIT_TRANSFER_OK) { return; }
            CHECK(status == GGML_SPACEMIT_TRANSFER_BUSY && clock_type::now() < deadline);
            std::this_thread::yield();
        }
    };
    CHECK(ggml_spacemit_transfer_prefetch(nullptr, source.data(), 1, event) == GGML_SPACEMIT_TRANSFER_INVALID);
    CHECK(ggml_spacemit_transfer_prefetch(transfer, nullptr, 1, event) == GGML_SPACEMIT_TRANSFER_INVALID);
    CHECK(ggml_spacemit_transfer_prefetch(transfer, source.data(), 1, nullptr) == GGML_SPACEMIT_TRANSFER_INVALID);
#ifndef SPACEMIT_TRANSFER_TEST_REAL
    const size_t before = warmed_bytes;
    const int copy_count = copies;
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        blocked = true;
        entered = false;
    }
#endif
    enqueue_prefetch(source.data() + 1, source.size() - 2);
#ifndef SPACEMIT_TRANSFER_TEST_REAL
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        CHECK(gate_cv.wait_for(lock, std::chrono::seconds(10), [] { return entered; }));
        CHECK(ggml_spacemit_transfer_query(event) == GGML_SPACEMIT_TRANSFER_PENDING);
        // Submission already returned although the worker is still blocked.
        blocked = false;
    }
    gate_cv.notify_all();
#endif
    CHECK(ggml_spacemit_transfer_wait(event) == GGML_SPACEMIT_TRANSFER_OK);
    CHECK(source == original);
#ifndef SPACEMIT_TRANSFER_TEST_REAL
    CHECK(warmed_bytes - before == source.size() - 2);
    CHECK(copies == copy_count); // warmup never dispatches a payload copy
    copy_failed = true;
    enqueue_prefetch(source.data(), 1);
    CHECK(ggml_spacemit_transfer_wait(event) == GGML_SPACEMIT_TRANSFER_ERROR);
    copy_failed = false;
#endif
    // Exercise cancellation through the warmup dispatch, not just copy jobs.
#ifndef SPACEMIT_TRANSFER_TEST_REAL
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        blocked = true;
        entered = false;
    }
    const size_t before_cancel = warmed_bytes;
    enqueue_prefetch(source.data(), source.size());
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        CHECK(gate_cv.wait_for(lock, std::chrono::seconds(10), [] { return entered; }));
    }
    CHECK(cancel(event) == GGML_SPACEMIT_TRANSFER_OK);
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        blocked = false;
    }
    gate_cv.notify_all();
    CHECK(ggml_spacemit_transfer_wait(event) == GGML_SPACEMIT_TRANSFER_CANCELLED);
    CHECK(warmed_bytes - before_cancel <= 64 * 1024);
#endif
#if defined(__linux__)
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    auto * mapping = static_cast<unsigned char *>(mmap(nullptr, 5 * page, PROT_NONE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    CHECK(mapping != MAP_FAILED);
    CHECK(mprotect(mapping + page, 3 * page, PROT_READ) == 0);
    unsigned char resident[3] = {};
    CHECK(mincore(mapping + page, 3 * page, resident) == 0);
    CHECK(!(resident[0] & 1) && !(resident[1] & 1) && !(resident[2] & 1));
    // Unaligned start, ends exactly at a guard page: cover every page without
    // reading the next page or writing the read-only source mapping.
    enqueue_prefetch(mapping + page + 3, 3 * page - 3);
    CHECK(ggml_spacemit_transfer_wait(event) == GGML_SPACEMIT_TRANSFER_OK);
    CHECK(mincore(mapping + page, 3 * page, resident) == 0);
    CHECK((resident[0] & 1) && (resident[1] & 1) && (resident[2] & 1));
    CHECK(munmap(mapping, 5 * page) == 0);
#endif
    enqueue_prefetch(nullptr, 0);
    CHECK(ggml_spacemit_transfer_wait(event) == GGML_SPACEMIT_TRANSFER_OK);
    // Reuse the warmup event for an ordinary copy.
    unsigned char dst = 0;
    submit(transfer, {&dst, source.data(), 1}, event);
    CHECK(ggml_spacemit_transfer_wait(event) == GGML_SPACEMIT_TRANSFER_OK && dst == 0xa5);
    ggml_spacemit_transfer_event_free(event);
    ggml_spacemit_transfer_free(transfer);
    CHECK(spacemit_transfer_get_proc_address("ggml_spacemit_transfer_prefetch") != nullptr);
    std::puts("PASS async page prefetch/unaligned/zero/failure/source unchanged/event reuse");
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
    const size_t workers = std::min(size_t(4), cpus.size());
#else
    const size_t workers = 2;
    failures();
    bounded_queue();
    cancellation();
#endif
    page_prefetch();
    basic();
    chunked_overlap();
    cancellation_races();
    concurrent_and_shutdown(1);
    concurrent_and_shutdown(workers);
    std::puts("ALL TRANSFER TESTS PASSED");
}
