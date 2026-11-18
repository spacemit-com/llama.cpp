// Hardware regression test: toy graph exercises event timing and cache replay.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-spacemit.h"
#include "ggml-spacemit-transfer.h"
#include <sched.h>
#include "ggml-impl.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d %s\n", __LINE__, #x); std::abort(); } } while (0)
static void fill(ggml_tensor * t, int seed) {
    std::vector<float> values(ggml_nelements(t));
    for (size_t i=0; i<values.size(); ++i) values[i] = float(int((i*17+seed*31)%101)-50)/128;
    if (t->type == GGML_TYPE_Q4_0) {
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, values.data(), q.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
        ggml_backend_tensor_set(t,q.data(),0,q.size());
    } else ggml_backend_tensor_set(t,values.data(),0,values.size()*sizeof(float));
}
struct State {
    size_t events=0, ready=0, expert_begin=0, attention_begin=0;
    std::vector<int32_t> ids;
    bool selected=false;
    ggml_spacemit_transfer_t mover=nullptr;
    ggml_spacemit_transfer_event_t done=nullptr;
    std::vector<unsigned char> source, destination;
    bool copy_pending=false;
    size_t submitted=0, busy=0, completed=0;
    int submitting_cpu=-1;
};
static void callback(const ggml_spacemit_prefetch_event * e, void * opaque) {
    auto & s=*static_cast<State *>(opaque);
    CHECK(e->layer==3);
    ++s.events;
    if (e->module==GGML_SPACEMIT_MODULE_ATTENTION && e->phase==GGML_SPACEMIT_PREFETCH_BEGIN) {
        CHECK(!s.copy_pending);
        std::fill(s.destination.begin(),s.destination.end(),0);
        ggml_spacemit_transfer_request request{s.destination.data(),s.source.data(),s.source.size()};
        const auto status=ggml_spacemit_transfer_submit(s.mover,&request,s.done);
        CHECK(status==GGML_SPACEMIT_TRANSFER_OK || status==GGML_SPACEMIT_TRANSFER_BUSY);
        s.copy_pending=status==GGML_SPACEMIT_TRANSFER_OK;
        if (s.copy_pending) { ++s.submitted; } else { ++s.busy; }
        s.submitting_cpu=sched_getcpu();
    }
    if (e->module==GGML_SPACEMIT_MODULE_ATTENTION && e->phase==GGML_SPACEMIT_PREFETCH_BEGIN) ++s.attention_begin;
    if (e->phase==GGML_SPACEMIT_PREFETCH_READY) {
        ++s.ready;
        auto * t=e->tensor;
        CHECK(t->type==GGML_TYPE_I32);
        s.ids.clear();
        for (int64_t row=0; row<t->ne[1]; ++row)
            for (int64_t col=0; col<t->ne[0]; ++col) {
                int32_t id;
                std::memcpy(&id, static_cast<const uint8_t *>(t->data)+row*t->nb[1]+col*t->nb[0], sizeof(id));
                CHECK(id>=0 && id<32);
                s.ids.push_back(id);
            }
        s.selected=true;
    }
    if (e->module==GGML_SPACEMIT_MODULE_EXPERTS && e->phase==GGML_SPACEMIT_PREFETCH_BEGIN) {
        CHECK(s.selected);
        if (s.copy_pending) {
            CHECK(ggml_spacemit_transfer_wait(s.done)==GGML_SPACEMIT_TRANSFER_OK);
            CHECK(s.source==s.destination);
            s.copy_pending=false;
            ++s.completed;
        }
        ++s.expert_begin;
    }
}
static void name(ggml_tensor * t, const char * stem) { ggml_format_name(t,"%s-3",stem); }
int main() {
    // Must create background workers on the ordinary control thread, before
    // entering spert/A100 execution. The callback submits only a descriptor.
    State state;
    ggml_spacemit_transfer_status transfer_status;
    state.mover=ggml_spacemit_transfer_create(nullptr,&transfer_status);
    CHECK(state.mover && transfer_status==GGML_SPACEMIT_TRANSFER_OK);
    state.done=ggml_spacemit_transfer_event_create(); CHECK(state.done);
    state.source.resize(64*1024); state.destination.resize(state.source.size());
    for (size_t i=0; i<state.source.size(); ++i) state.source[i]=static_cast<unsigned char>(i*31+7);
    auto backend=ggml_backend_spacemit_init(); CHECK(backend);
    CHECK(!ggml_backend_spacemit_set_prefetch(nullptr,true));
    CHECK(ggml_backend_reg_get_proc_address(ggml_backend_spacemit_reg(),"ggml_backend_set_prefetch"));
    CHECK(!ggml_backend_spacemit_set_prefetch_callback(nullptr,callback,nullptr));
    CHECK(ggml_backend_reg_get_proc_address(ggml_backend_spacemit_reg(),"ggml_backend_spacemit_set_prefetch_callback"));
    auto ctx=ggml_init({16*1024*1024,nullptr,true});
    auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,2);
    auto w0=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_0,32,32);
    auto w1=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_0,32,32);
    auto w2=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_0,32,32);
    auto ew=ggml_new_tensor_3d(ctx,GGML_TYPE_Q4_0,32,32,32);
    auto scale=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,32);
    auto rms=ggml_rms_norm(ctx,x,1e-5f); name(rms,"norm");
    auto norm=ggml_mul(ctx,rms,scale); name(norm,"attn_norm");
    auto q=ggml_mul_mat(ctx,w0,norm); name(q,"Qcur");
    auto k=ggml_mul_mat(ctx,w1,norm); name(k,"Kcur");
    auto v=ggml_mul_mat(ctx,w2,norm); name(v,"Vcur");
    auto attn=ggml_add(ctx,ggml_soft_max(ctx,ggml_add(ctx,q,k)),v); name(attn,"kqv_out");
    auto fn=ggml_rms_norm(ctx,attn,1e-5f); name(fn,"ffn_norm");
    auto logits=ggml_mul_mat(ctx,w0,fn); name(logits,"ffn_moe_logits");
    auto probs=ggml_soft_max(ctx,logits); name(probs,"ffn_moe_probs");
    auto ids=ggml_argsort_top_k(ctx,probs,2); name(ids,"ffn_moe_topk"); name(ids->src[0],"ffn_moe_argsort");
    auto out=ggml_mul_mat_id(ctx,ew,ggml_reshape_3d(ctx,fn,32,1,2),ids); name(out,"ffn_moe_gate");
    auto graph=ggml_new_graph_custom(ctx,128,false);
    ggml_build_forward_expand(graph,out);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend); CHECK(buffer);
    fill(x,1); fill(w0,2); fill(w1,3); fill(w2,4); fill(ew,5); fill(scale,6);
    // Stable UID exercises enable/disable and replay without rebuilding opnodes.
    graph->uid=0x20261117;
    CHECK(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
    std::vector<float> baseline(ggml_nelements(out)), candidate(baseline.size());
    ggml_backend_tensor_get(out,baseline.data(),0,ggml_nbytes(out));
    state.ids.reserve(8);
    CHECK(ggml_backend_spacemit_set_prefetch_callback(backend,callback,&state));
    for (int repeat=0; repeat<2; ++repeat) {
        state.selected=false;
        CHECK(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out,candidate.data(),0,ggml_nbytes(out));
        CHECK(std::memcmp(baseline.data(),candidate.data(),ggml_nbytes(out))==0);
        for (float f:candidate) CHECK(std::isfinite(f));
    }
    CHECK(state.ready==2 && state.expert_begin==2 && state.attention_begin==2);
    // Explicit disable suppresses even an installed callback on the same cached UID.
    const size_t paused_count=state.events;
    CHECK(ggml_backend_spacemit_set_prefetch(backend,false));
    CHECK(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
    CHECK(state.events==paused_count);
    CHECK(ggml_backend_spacemit_set_prefetch(backend,true));
    state.selected=false;
    CHECK(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
    CHECK(state.ready==3 && state.expert_begin==3 && state.attention_begin==3);
    ggml_backend_tensor_get(out,candidate.data(),0,ggml_nbytes(out));
    CHECK(std::memcmp(baseline.data(),candidate.data(),ggml_nbytes(out))==0);
    const size_t event_count=state.events;
    CHECK(ggml_backend_spacemit_set_prefetch_callback(backend,nullptr,nullptr));
    CHECK(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
    CHECK(state.events==event_count);
    // CLI-style planning without a callback changes neither outputs nor events.
    CHECK(ggml_backend_spacemit_set_prefetch(backend,true));
    CHECK(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(out,candidate.data(),0,ggml_nbytes(out));
    CHECK(std::memcmp(baseline.data(),candidate.data(),ggml_nbytes(out))==0);
    CHECK(state.events==event_count);
    CHECK(ggml_backend_spacemit_set_prefetch(backend,false));
    std::printf("PASS: %zu events, READY before expert GEMM, cached replay, explicit pause/resume, planning-only, bitwise outputs\n",event_count);
    CHECK(!state.copy_pending && state.completed==state.submitted && state.submitted>0);
    std::printf("PASS: graph worker-0 CPU %d submitted %zu copies, %zu BUSY, all accepted copies complete\n",
                state.submitting_cpu,state.submitted,state.busy);
    ggml_spacemit_transfer_free(state.mover);
    ggml_spacemit_transfer_event_free(state.done);
    ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(backend);
}
