// SPDX-License-Identifier: MIT
// Exercise the public manager against a controlled asynchronous transfer service.
// Cancellation deliberately remains pending until the test acknowledges it.
#include "llama-spacemit-prefetch.h"
#include "llama-spacemit-expert-cache.h"
#include "llama-model.h"
#include "llama-graph.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-spacemit.h"
#include "ggml-spacemit-transfer.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>


#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)
struct ggml_spacemit_transfer_event {
    ggml_spacemit_transfer_state state = GGML_SPACEMIT_TRANSFER_IDLE;
    ggml_spacemit_transfer_request request{};
    bool cancel = false;
};
struct ggml_spacemit_transfer {};
namespace {
std::mutex service_mutex;
std::vector<ggml_spacemit_transfer_event_t> events;
ggml_spacemit_prefetch_callback callback = nullptr;
void * callback_data = nullptr;
size_t submitted = 0, cancellations = 0, configured_workers = 0;
bool set_callback(ggml_backend_t, ggml_spacemit_prefetch_callback cb, void * data) {
    callback = cb; callback_data = data; return true;
}
ggml_spacemit_transfer_t create(const ggml_spacemit_transfer_config * config, ggml_spacemit_transfer_status * status) {
    configured_workers = config->n_workers;
    CHECK(config->queue_capacity == config->n_workers);
    *status = GGML_SPACEMIT_TRANSFER_OK; return new ggml_spacemit_transfer;
}
void free_transfer(ggml_spacemit_transfer_t service) { delete service; }
ggml_spacemit_transfer_event_t event_create() {
    auto * e = new ggml_spacemit_transfer_event;
    std::lock_guard<std::mutex> lock(service_mutex); events.push_back(e); return e;
}
void event_free(ggml_spacemit_transfer_event_t e) {
    std::lock_guard<std::mutex> lock(service_mutex);
    CHECK(e->state != GGML_SPACEMIT_TRANSFER_PENDING);
    events.erase(std::find(events.begin(), events.end(), e)); delete e;
}
ggml_spacemit_transfer_status submit(ggml_spacemit_transfer_t, const ggml_spacemit_transfer_request * r,
                                    ggml_spacemit_transfer_event_t e) {
    std::lock_guard<std::mutex> lock(service_mutex);
    CHECK(e->state != GGML_SPACEMIT_TRANSFER_PENDING);
    CHECK(r->src && r->size);
    e->request = *r; e->cancel = false; e->state = GGML_SPACEMIT_TRANSFER_PENDING;
    ++submitted; return GGML_SPACEMIT_TRANSFER_OK;
}
ggml_spacemit_transfer_status prefetch(ggml_spacemit_transfer_t service, const void * src, size_t bytes,
                                      ggml_spacemit_transfer_event_t e) {
    const ggml_spacemit_transfer_request r{nullptr, src, bytes}; return submit(service, &r, e);
}
ggml_spacemit_transfer_status cancel(ggml_spacemit_transfer_event_t e) {
    std::lock_guard<std::mutex> lock(service_mutex);
    if (e->state != GGML_SPACEMIT_TRANSFER_PENDING) { return GGML_SPACEMIT_TRANSFER_INVALID; }
    if (!e->cancel) { ++cancellations; }
    e->cancel = true; return GGML_SPACEMIT_TRANSFER_OK;
}
ggml_spacemit_transfer_state query(ggml_spacemit_transfer_event_t e) {
    std::lock_guard<std::mutex> lock(service_mutex); return e->state;
}
void finish(size_t limit = 4) {
    std::lock_guard<std::mutex> lock(service_mutex);
    for (auto * e : events) {
        if (!limit || e->state != GGML_SPACEMIT_TRANSFER_PENDING) { continue; }
        --limit;
        if (e->request.dst) {
            // Even accepted cancellations may still write before acknowledgement.
            if (e->cancel) { std::memset(e->request.dst, 0xa5, e->request.size / 2); }
            else { std::memcpy(e->request.dst, e->request.src, e->request.size); }
        }
        e->state = e->cancel ? GGML_SPACEMIT_TRANSFER_CANCELLED_STATE : GGML_SPACEMIT_TRANSFER_COMPLETE;
    }
}
ggml_spacemit_transfer_status drain(ggml_spacemit_transfer_t) { finish(); return GGML_SPACEMIT_TRANSFER_OK; }
void * proc(ggml_backend_reg_t, const char * name) {
#define PROC(n, f) if (!std::strcmp(name, n)) { return reinterpret_cast<void *>(&f); }
    PROC("ggml_backend_spacemit_set_prefetch_callback", set_callback)
    PROC("ggml_spacemit_transfer_create", create)
    PROC("ggml_spacemit_transfer_free", free_transfer)
    PROC("ggml_spacemit_transfer_event_create", event_create)
    PROC("ggml_spacemit_transfer_event_free", event_free)
    PROC("ggml_spacemit_transfer_submit", submit)
    PROC("ggml_spacemit_transfer_prefetch", prefetch)
    PROC("ggml_spacemit_transfer_cancel", cancel)
    PROC("ggml_spacemit_transfer_query", query)
    PROC("ggml_spacemit_transfer_drain", drain)
#undef PROC
    return nullptr;
}
struct mock_model final : llama_model {
    using llama_model::params;
    mock_model() : llama_model(llama_model_default_params()) {}
    void load_stats(llama_model_loader &) override {}
    void load_hparams(llama_model_loader &) override {}
    void load_vocab(llama_model_loader &) override {}
    bool load_tensors(llama_model_loader &) override { return true; }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override { return nullptr; }
};
template<class F> void until(F predicate, bool pump = false) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!predicate()) {
        CHECK(std::chrono::steady_clock::now() < deadline);
        if (pump) { finish(); }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
size_t pending() {
    std::lock_guard<std::mutex> lock(service_mutex);
    return std::count_if(events.begin(), events.end(), [](auto * e) { return e->state == GGML_SPACEMIT_TRANSFER_PENDING; });
}
size_t submission_count() { std::lock_guard<std::mutex> lock(service_mutex); return submitted; }
size_t cancellation_count() { std::lock_guard<std::mutex> lock(service_mutex); return cancellations; }
void emit(ggml_spacemit_module module, ggml_spacemit_prefetch_phase phase, int layer, ggml_tensor * t = nullptr) {
    CHECK(callback); const ggml_spacemit_prefetch_event e{module, phase, layer, 0, t}; callback(&e, callback_data);
}
} // namespace
int main() {
    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true}); CHECK(ctx);
    auto * cpu = ggml_backend_cpu_init(); CHECK(cpu);
    ggml_backend_reg reg{}; reg.api_version = GGML_BACKEND_API_VERSION; reg.iface.get_proc_address = proc;
    ggml_backend_device device{}; device.reg = &reg;
    auto * fake = new ggml_backend{}; fake->device = &device;
    fake->iface.free = [](ggml_backend_t b) { delete b; };
    std::vector<ggml_backend_ptr> backends; backends.emplace_back(fake);
    ggml_backend_buffer_t memory = nullptr;
    {
        mock_model model; model.layers.resize(2);
        for (auto & layer : model.layers) {
            for (auto ** bank : {&layer.ffn_gate_exps, &layer.ffn_up_exps, &layer.ffn_down_exps}) {
                *bank = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 1033, 5);
                model.can_prefetch.insert(*bank);
            }
        }
        auto * id_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 2);
        auto * ids = ggml_view_2d(ctx, id_storage, 2, 2, id_storage->nb[1], sizeof(int32_t));
        memory = ggml_backend_alloc_ctx_tensors(ctx, cpu); CHECK(memory);
        llama_spacemit_expert_cache::banks banks(2);
        const auto payload = [](size_t layer, size_t bank, int expert) { return 0x30 + 15 * layer + 5 * bank + expert; };
        for (size_t il = 0; il < 2; ++il) {
            auto & layer = model.layers[il];
            for (auto * bank : {layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps}) {
                for (int expert = 0; expert < bank->ne[2]; ++expert) {
                    std::memset(static_cast<unsigned char *>(bank->data) + expert * bank->nb[2],
                            payload(il, banks[il].size(), expert), bank->nb[2]);
                }
                banks[il].push_back(bank);
            }
        }
        model.expert_cache = std::make_unique<llama_spacemit_expert_cache>(banks, .65);
        auto & cache = *model.expert_cache;
        CHECK(model.params.prefetch_threads == 4);
        llama_spacemit_prefetch manager(model, backends); CHECK(callback && configured_workers == 4);
        auto all = [&](size_t il) { for (size_t e = 0; e < 5; ++e) { if (!cache.contains(il, e)) { return false; } } return true; };
        auto set_ids = [&](int32_t a, int32_t b) {
            int32_t values[]{a, b};
            for (int64_t row = 0; row < ids->ne[1]; ++row) {
                std::memcpy(static_cast<char *>(ids->data) + row * ids->nb[1], values, sizeof(values));
            }
        };
        set_ids(1, 3);
        for (int il = 0; il < 2; ++il) {
            emit(GGML_SPACEMIT_MODULE_QKV, GGML_SPACEMIT_PREFETCH_BEGIN, il);
            until([] { return pending() == 4; });
            {
                std::lock_guard<std::mutex> lock(service_mutex);
                const auto * gate = banks[il][0];
                std::vector<bool> seen(3);
                size_t warmed = 0;
                for (auto * e : events) {
                    if (e->state == GGML_SPACEMIT_TRANSFER_PENDING) {
                        CHECK(e->request.size == gate->nb[2]);
                        if (!e->request.dst) {
                            // The lane may have advanced to up/down. At a new
                            // layer boundary an old request can also remain
                            // pending only until its cancellation is acknowledged.
                            bool valid_source = false;
                            bool current_source = false;
                            for (size_t l = 0; l < banks.size(); ++l) {
                                for (const auto * bank : banks[l]) {
                                    for (int expert = 0; expert < bank->ne[2]; ++expert) {
                                        const bool matches = e->request.src ==
                                                static_cast<const unsigned char *>(bank->data) + expert * bank->nb[2];
                                        valid_source |= matches;
                                        current_source |= matches && l == size_t(il);
                                    }
                                }
                            }
                            CHECK(valid_source);
                            CHECK(current_source || e->cancel);
                            ++warmed;
                            continue;
                        }
                        size_t expert = 0;
                        while (expert < seen.size() && e->request.src !=
                                static_cast<const unsigned char *>(gate->data) + expert * gate->nb[2]) {
                            ++expert;
                        }
                        CHECK(expert < seen.size() && !seen[expert]);
                        seen[expert] = true;
                    }
                }
                CHECK(warmed == 1);
                CHECK(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }));
            }
            const size_t cancelled_before = cancellation_count();
            // A second backend fragment in this microbatch must neither
            // cancel in-flight writes nor open another prefetch window.
            emit(GGML_SPACEMIT_MODULE_QKV, GGML_SPACEMIT_PREFETCH_BEGIN, il);
            CHECK(cancellation_count() == cancelled_before && pending() == 4);
            until([&] { return all(il); }, true);
            until([] { return pending() == 0; }, true); // acknowledge the independent warmer too
            emit(GGML_SPACEMIT_MODULE_ROUTER, GGML_SPACEMIT_PREFETCH_READY, il, ids);
            for (size_t b = 0; b < banks[il].size(); ++b) {
                for (int expert : {1, 3}) {
                    const auto * bank = banks[il][b];
                    const auto * data = static_cast<const unsigned char *>(ggml_backend_tensor_get_expert(bank, expert));
                    CHECK(data != static_cast<const unsigned char *>(bank->data) + expert * bank->nb[2]);
                    CHECK(std::all_of(data, data + bank->nb[2], [=](unsigned char v) { return v == payload(il, b, expert); }));
                }
            }
            // CPU experts emit no SpacemiT END: the next QKV must release pins.
        }
        CHECK(!cache.contains(0, 0) && all(1)); // one prefix reused, no extra region consumed
        // Prefill must cancel at router even when no expert finished. Pending
        // destinations stay reserved until the worker acknowledges cancellation.
        emit(GGML_SPACEMIT_MODULE_QKV, GGML_SPACEMIT_PREFETCH_BEGIN, 0);
        until([] { return pending() == 4; });
        const size_t prefill_cancelled_before = cancellation_count();
        emit(GGML_SPACEMIT_MODULE_ROUTER, GGML_SPACEMIT_PREFETCH_READY, 0, ids);
        CHECK(cancellation_count() >= prefill_cancelled_before &&
                cancellation_count() <= prefill_cancelled_before + 4 && pending() == 4);
        {
            std::lock_guard<std::mutex> lock(service_mutex);
            for (const auto * e : events) {
                if (e->state == GGML_SPACEMIT_TRANSFER_PENDING) { CHECK(e->cancel); }
            }
        }
        const size_t prefill_stopped_at = submission_count();
        auto * prefill_bank = model.layers[0].ffn_gate_exps;
        CHECK(ggml_backend_tensor_get_expert(prefill_bank, 1) == static_cast<char *>(prefill_bank->data) + prefill_bank->nb[2]);
        finish();
        until([] { return pending() == 0; });
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        CHECK(submission_count() == prefill_stopped_at && !cache.contains(0, 1));
        emit(GGML_SPACEMIT_MODULE_PREDICTOR, GGML_SPACEMIT_PREFETCH_READY, 0, ids);
        emit(GGML_SPACEMIT_MODULE_QKV, GGML_SPACEMIT_PREFETCH_BEGIN, 0);
        until([] { return pending() == 2; }); // one request per predicted expert
        until([&] {
            if (cache.contains(0, 1)) { return true; }
            finish(1); // only expert 1 progresses; expert 3 remains in flight
            return false;
        });
        until([] { return pending() > 0; });
        set_ids(1, 4); // prediction hit 1; unpredicted expert 4 must use mmap
        emit(GGML_SPACEMIT_MODULE_ROUTER, GGML_SPACEMIT_PREFETCH_READY, 0, ids);
        CHECK(cancellation_count() > 0 && pending() > 0); // cancellation never drains in callback
        const size_t stopped_at = submission_count();
        auto * bank = model.layers[0].ffn_gate_exps;
        CHECK(ggml_backend_tensor_get_expert(bank, 1) != static_cast<char *>(bank->data) + bank->nb[2]);
        CHECK(ggml_backend_tensor_get_expert(bank, 4) == static_cast<char *>(bank->data) + 4 * bank->nb[2]);
        finish(); // late writes finish only into their still-reserved destination
        until([] { return pending() == 0; });
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        CHECK(submission_count() == stopped_at && !cache.contains(0, 3));
        // Again omit EXPERTS END to exercise CPU expert reader lifetime.
        // The next decode window can safely reuse the acknowledged cancellation slot.
        set_ids(0, 1);
        emit(GGML_SPACEMIT_MODULE_PREDICTOR, GGML_SPACEMIT_PREFETCH_READY, 1, ids);
        emit(GGML_SPACEMIT_MODULE_QKV, GGML_SPACEMIT_PREFETCH_BEGIN, 1);
        CHECK(ggml_backend_tensor_get_expert(bank, 1) == static_cast<char *>(bank->data) + bank->nb[2]);
        until([&] { return cache.contains(1, 0) && cache.contains(1, 1); }, true);
        emit(GGML_SPACEMIT_MODULE_ROUTER, GGML_SPACEMIT_PREFETCH_READY, 1, ids);
        auto * next_bank = model.layers[1].ffn_gate_exps;
        CHECK(ggml_backend_tensor_get_expert(next_bank, 0) != next_bank->data);
        // Same-layer next-token QKV must also release CPU readers.
        emit(GGML_SPACEMIT_MODULE_PREDICTOR, GGML_SPACEMIT_PREFETCH_READY, 1, ids);
        emit(GGML_SPACEMIT_MODULE_QKV, GGML_SPACEMIT_PREFETCH_BEGIN, 1);
        CHECK(ggml_backend_tensor_get_expert(next_bank, 0) == next_bank->data);
        emit(GGML_SPACEMIT_MODULE_ROUTER, GGML_SPACEMIT_PREFETCH_READY, 1, ids);
        emit(GGML_SPACEMIT_MODULE_EXPERTS, GGML_SPACEMIT_PREFETCH_END, 1);
        manager.stop(); CHECK(callback == nullptr && events.empty());
        model.params.prefetch_threads = 2;
        llama_spacemit_prefetch two_workers(model, backends);
        CHECK(callback && configured_workers == 2 && events.size() == 2);
        emit(GGML_SPACEMIT_MODULE_QKV, GGML_SPACEMIT_PREFETCH_BEGIN, 0);
        until([] { return pending() == 2; });
        emit(GGML_SPACEMIT_MODULE_ROUTER, GGML_SPACEMIT_PREFETCH_READY, 0, ids);
        const size_t two_stopped_at = submission_count();
        finish();
        until([] { return pending() == 0; });
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        CHECK(submission_count() == two_stopped_at);
        two_workers.stop(); CHECK(callback == nullptr && events.empty());
    }
    ggml_backend_buffer_free(memory); ggml_backend_free(cpu); ggml_free(ctx);
    std::puts("PASS: concurrent prefill pagewarm/copy, shared prefill/decode buffer, prediction, router cancellation, pinned hit, mmap miss, late-write acknowledgement");
}
