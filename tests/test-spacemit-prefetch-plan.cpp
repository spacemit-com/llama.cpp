// Host-side graph recognition regression test; no SpacemiT hardware required.
// The two trait queries are intentionally stubbed: tests construct execution
// plans explicitly so fused/unfused boundary mapping can be checked on a host.
#include "spacemit-graph.h"
#include "ime.h"
#include <cstdio>
#include <cstdlib>

bool ggml_spacemit_ffn_f16_eligible(const ggml_tensor *, const ggml_tensor *) { return false; }
const ggml::spacemit::tensor_traits_base * ggml_spacemit_get_tensor_traits(const ggml_tensor *) { return nullptr; }

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

static void name(ggml_tensor * t, const char * stem, int layer = 3) {
    ggml_format_name(t, "%s-%d", stem, layer);
}
static std::vector<spacemit_opnode> plain_plan(ggml_cgraph * graph) {
    std::vector<spacemit_opnode> nodes;
    for (int i = 0; i < graph->n_nodes; ++i) {
        auto * t = graph->nodes[i];
        if (!ggml_op_is_empty(t->op) && !ggml_is_empty(t) && (t->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            nodes.emplace_back(t);
        }
    }
    return nodes;
}
static size_t count(const std::vector<spacemit_prefetch_point> & plan,
                    ggml_spacemit_module module, ggml_spacemit_prefetch_phase phase,
                    const ggml_tensor * expected = nullptr) {
    size_t n = 0;
    for (const auto & point : plan) {
        const auto & events = phase == GGML_SPACEMIT_PREFETCH_BEGIN ? point.before : point.after;
        for (const auto & e : events) {
            if (e.module == module && e.phase == phase && (!expected || e.tensor == expected)) {
                CHECK(e.layer == 3);
                ++n;
            }
        }
    }
    return n;
}
int main() {
    auto * ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 2);
    auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
    auto * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    auto * rms = ggml_rms_norm(ctx, x, 1e-5f); name(rms, "norm");
    auto * norm = ggml_mul(ctx, rms, scale); name(norm, "attn_norm");
    auto * q = ggml_mul_mat(ctx, w, norm); name(q, "Qcur");
    auto * k = ggml_mul_mat(ctx, w, norm); name(k, "Kcur");
    auto * v = ggml_mul_mat(ctx, w, norm); name(v, "Vcur");
    auto * scores = ggml_soft_max(ctx, ggml_add(ctx, q, k));
    auto * out = ggml_add(ctx, scores, v); name(out, "kqv_out");
    auto * ffn_norm = ggml_rms_norm(ctx, out, 1e-5f); name(ffn_norm, "ffn_norm");
    auto * logits = ggml_mul_mat(ctx, w, ffn_norm); name(logits, "ffn_moe_logits");
    auto * probs = ggml_soft_max(ctx, logits); name(probs, "ffn_moe_probs");
    auto * topk = ggml_argsort_top_k(ctx, probs, 2); name(topk, "ffn_moe_topk");
    name(topk->src[0], "ffn_moe_argsort");
    auto * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 32, 32);
    auto * activation = ggml_reshape_3d(ctx, ffn_norm, 32, 1, 2);
    auto * expert = ggml_mul_mat_id(ctx, weights, activation, topk);
    name(expert, "ffn_moe_gate");
    auto * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, expert);
    auto nodes = plain_plan(graph);
    std::vector<spacemit_prefetch_point> plan;
    spacemit_graph_build_prefetch_plan(graph, nodes, plan);
    CHECK(plan.size() == nodes.size());
    CHECK(count(plan, GGML_SPACEMIT_MODULE_QKV, GGML_SPACEMIT_PREFETCH_BEGIN) == 1);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_ATTENTION, GGML_SPACEMIT_PREFETCH_BEGIN) == 1);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_ROUTER, GGML_SPACEMIT_PREFETCH_BEGIN) == 1);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_RMS_NORM, GGML_SPACEMIT_PREFETCH_BEGIN) == 2);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_ROUTER, GGML_SPACEMIT_PREFETCH_READY, topk) == 1);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_EXPERTS, GGML_SPACEMIT_PREFETCH_BEGIN) == 1);
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (const auto & e : plan[i].after) {
            if (e.phase == GGML_SPACEMIT_PREFETCH_READY) {
                CHECK(nodes[i].node == topk->src[0]);
                CHECK(e.tensor == topk);
            }
        }
    }
    std::puts("PASS: named modules, view top-k producer, expert dependency");

    // Collapse RMS+MUL and Q+K to test both fused primary and member mapping.
    std::vector<spacemit_opnode> fused;
    for (const auto & node : nodes) {
        if (node.node == rms) {
            fused.emplace_back(rms, std::vector<ggml_tensor *>{norm}, SPACEMIT_OP_RMS_NORM_MUL);
        } else if (node.node == q) {
            fused.emplace_back(k, std::vector<ggml_tensor *>{q}, SPACEMIT_OP_MUL_MAT_FFN);
        } else if (node.node != norm && node.node != k) {
            fused.push_back(node);
        }
    }
    spacemit_graph_build_prefetch_plan(graph, fused, plan);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_QKV, GGML_SPACEMIT_PREFETCH_BEGIN) == 1);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_RMS_NORM, GGML_SPACEMIT_PREFETCH_END, norm) == 1);
    std::puts("PASS: fused primary/member mapping");

    // Scheduler-copy names preserve semantics, but copied top-k is not locally
    // produced and must not generate a local READY event.
    ggml_set_name(topk, "CPU#ffn_moe_topk-3#0");
    spacemit_graph_build_prefetch_plan(graph, nodes, plan);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_ROUTER, GGML_SPACEMIT_PREFETCH_READY) == 1);
    auto * split = ggml_new_graph_custom(ctx, 16, false);
    ggml_graph_add_node(split, expert);
    std::vector<spacemit_opnode> split_nodes{spacemit_opnode(expert)};
    spacemit_graph_build_prefetch_plan(split, split_nodes, plan);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_EXPERTS, GGML_SPACEMIT_PREFETCH_BEGIN) == 1);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_ROUTER, GGML_SPACEMIT_PREFETCH_READY) == 0);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_ATTENTION, GGML_SPACEMIT_PREFETCH_BEGIN) == 0);
    std::puts("PASS: scheduler copies and partial backend graph");

    // Names without data-flow to the matching layer boundary must not match.
    ggml_set_name(norm, "attn_norm-4");
    ggml_set_name(ffn_norm, "ffn_norm-4");
    spacemit_graph_build_prefetch_plan(graph, nodes, plan);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_QKV, GGML_SPACEMIT_PREFETCH_BEGIN) == 0);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_ATTENTION, GGML_SPACEMIT_PREFETCH_BEGIN) == 0);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_ROUTER, GGML_SPACEMIT_PREFETCH_BEGIN) == 0);
    std::puts("PASS: layer mismatch does not create semantic windows");
    // A scheduler split can end at the first Q projection, with q/kv and
    // RoPE on another backend. Exact weight roles plus a copied norm suffice.
    auto * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
    ggml_set_name(wq, "SPACEMIT0#blk.3.attn_q_a.weight#0");
    auto * norm_copy = ggml_dup(ctx, norm);
    ggml_set_name(norm_copy, "SPACEMIT0#attn_norm-3#0");
    auto * projection = ggml_mul_mat(ctx, wq, norm_copy);
    auto * projection_graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(projection_graph, projection);
    auto projection_split = *projection_graph;
    projection_split.nodes += projection_graph->n_nodes - 1;
    projection_split.n_nodes = 1;
    spacemit_graph_build_prefetch_plan(&projection_split, {spacemit_opnode(projection)}, plan);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_QKV, GGML_SPACEMIT_PREFETCH_BEGIN) == 1);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_QKV, GGML_SPACEMIT_PREFETCH_END) == 1);
    ggml_set_name(norm_copy, "attn_norm-4");
    spacemit_graph_build_prefetch_plan(&projection_split, {spacemit_opnode(projection)}, plan);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_QKV, GGML_SPACEMIT_PREFETCH_BEGIN) == 0);
    ggml_set_name(norm_copy, "attn_norm-3");
    ggml_set_name(wq, "blk.3.attn_q_a.weight.extra");
    spacemit_graph_build_prefetch_plan(&projection_split, {spacemit_opnode(projection)}, plan);
    CHECK(count(plan, GGML_SPACEMIT_MODULE_QKV, GGML_SPACEMIT_PREFETCH_BEGIN) == 0);
    std::puts("PASS: QKV projection in split, layer and exact weight role guards");
    // Predictor is an independent speculative branch expanded before QKV.
    // Its VIEW top-k must signal READY at the producing argsort, separately
    // from the real FFN router, including when graph planning is repeated.
    ggml_set_name(wq, "blk.3.attn_q_a.weight");
    auto * prediction = ggml_argsort_top_k(ctx, q, 2);
    name(prediction, "ffn_predictor_topk");
    name(prediction->src[0], "ffn_predictor_argsort");
    auto * predicted_graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(predicted_graph, prediction);
    ggml_build_forward_expand(predicted_graph, projection);
    auto predicted_nodes = plain_plan(predicted_graph);
    for (int repeat = 0; repeat < 2; ++repeat) {
        spacemit_graph_build_prefetch_plan(predicted_graph, predicted_nodes, plan);
        CHECK(count(plan, GGML_SPACEMIT_MODULE_PREDICTOR, GGML_SPACEMIT_PREFETCH_READY, prediction) == 1);
        CHECK(count(plan, GGML_SPACEMIT_MODULE_ROUTER, GGML_SPACEMIT_PREFETCH_READY) == 0);
        CHECK(count(plan, GGML_SPACEMIT_MODULE_ROUTER, GGML_SPACEMIT_PREFETCH_BEGIN) == 0);
        size_t predictor_at = plan.size(), qkv_at = plan.size();
        for (size_t j = 0; j < plan.size(); ++j) {
            for (const auto & event : plan[j].before) {
                if (event.module == GGML_SPACEMIT_MODULE_QKV) { qkv_at = std::min(qkv_at, j); }
            }
            for (const auto & event : plan[j].after) {
                if (event.module == GGML_SPACEMIT_MODULE_PREDICTOR) {
                    CHECK(predicted_nodes[j].node == prediction->src[0]);
                    predictor_at = j;
                }
            }
        }
        CHECK(predictor_at < qkv_at && qkv_at < plan.size());
    }
    std::puts("PASS: independent predictor top-k READY and repeated planning");
    ggml_free(ctx);
}
