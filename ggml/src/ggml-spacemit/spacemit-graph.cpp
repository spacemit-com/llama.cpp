#include "spacemit-graph.h"
#include "ggml-prefetch-debug.h"
#include "ime.h"

#include <algorithm>
#include <climits>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {

bool spacemit_graph_node_is_compute(ggml_tensor * node) {
    return !ggml_op_is_empty(node->op) && !ggml_is_empty(node) && (node->flags & GGML_TENSOR_FLAG_COMPUTE);
}

//** op fusion helpers
//
// Fusion patterns, matched on the graph when the opnode list is built:
//   1. RMS_NORM + MUL    -> SPACEMIT_OP_RMS_NORM_MUL (one opnode at the rms position)
//   2. CONCAT + SSM_CONV -> SPACEMIT_OP_CONCAT_STATE at the concat position (partial write)
//                           + SPACEMIT_OP_SSM_CONV_CONCAT at the ssm_conv position (reads the
//                           concat inputs directly; the concat only materialized its trailing
//                           columns). Both halves must keep their original node positions: the
//                           graph allocator plans buffer reuse on the node order.
//   3. Shared-input MUL_MAT pair (+ optional split SwiGLU), preserving both projections
//
// Fused opnodes are dispatched by ggml_spacemit_compute_opnode through the
// tensor traits of their primary node (compute_forward_fused).
//
// try_fuse_node contains ALL fusion matching and emission: Pattern 2 consumes
// two nodes at different positions (the concat now, the ssm_conv when the loop
// reaches it — opnode list order is execution order, so the halves keep their
// original node positions); `emitted` carries the pairing across the gap.

// RMS_NORM followed by MUL (rms, weight) that can run as one fused RVV kernel.
static bool rms_norm_mul_fusable(const ggml_tensor * rms, const ggml_tensor * mul) {
    if (mul->op != GGML_OP_MUL || (mul->src[0] != rms && mul->src[1] != rms)) {
        return false;
    }
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    return x && w && x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) &&
           mul->type == GGML_TYPE_F32 && mul->nb[0] == sizeof(float) &&
           w->type == GGML_TYPE_F32 && w->nb[0] == sizeof(float) &&
           w->ne[0] == rms->ne[0] && ggml_are_same_shape(x, mul);
}

// concat(conv_states, transpose(qkv), dim 0) as built by build_conv_state: contiguous F32 rows, the
// second input laid out transposed (column-major) so the fused kernel reads it column by column.
static bool is_transposed_concat_f32(const ggml_tensor * t) {
    if (t->op != GGML_OP_CONCAT || ggml_get_op_params_i32(t, 0) != 0) {
        return false;
    }
    const ggml_tensor * s0 = t->src[0];
    const ggml_tensor * s1 = t->src[1];
    return t->type == GGML_TYPE_F32 && s0->type == GGML_TYPE_F32 && s1->type == GGML_TYPE_F32 &&
           t->nb[0] == sizeof(float) && s0->nb[0] == sizeof(float) && s1->nb[1] == sizeof(float) &&
           s1->nb[0] != sizeof(float) && s0->ne[1] == t->ne[1] && s1->ne[1] == t->ne[1] && s0->ne[2] == t->ne[2] &&
           s1->ne[2] == t->ne[2] && t->ne[3] == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
}

// First column of concat c covered by view v, or -1 if v is not a plain window of whole rows' trailing
// columns (same row/plane strides, starting in row 0).
static int64_t concat_view_first_col(const ggml_tensor * c, const ggml_tensor * v) {
    if (v->type != c->type || v->nb[0] != c->nb[0] || (v->ne[1] > 1 && v->nb[1] != c->nb[1]) ||
        (v->ne[2] > 1 && v->nb[2] != c->nb[2]) || v->ne[3] != 1 || v->view_offs / c->nb[1] != 0) {
        return -1;
    }
    const int64_t col = (int64_t) ((v->view_offs % c->nb[1]) / c->nb[0]);
    if (col + v->ne[0] > c->ne[0] || v->ne[1] > c->ne[1] || v->ne[2] > c->ne[2]) {
        return -1;
    }
    return col;
}

// SSM_CONV op param set by graph_optimize when the concat inputs are kept allocated until the ssm_conv
static constexpr int SSM_CONV_PARAM_INPUTS_KEPT = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;

// concat -> ssm_conv fusion analysis: SSM_CONV reads the two concat inputs directly and CONCAT only
// writes the trailing columns that are read back through views (the conv state update). Returns the
// single ssm_conv consumer (with its graph index) and the first column any view reads back. Fails if
// the concat has any other consumer (or a second ssm_conv) that needs the full concat output.
// With need_kept, also fails unless graph_optimize kept the concat inputs allocated for the ssm_conv.
static bool ssm_conv_concat_fusable(const ggml_cgraph * graph, int ci, bool need_kept,
                                    ggml_tensor ** ssm_out, int * ssm_idx_out, int64_t * col0_out) {
    ggml_tensor * c = graph->nodes[ci];
    if (!is_transposed_concat_f32(c) || c->src[0]->ne[0] != 3) {
        return false;
    }

    ggml_tensor * ssm    = nullptr;
    int           ssm_idx = -1;
    int64_t       col0    = c->ne[0];  // min column read back through views

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * t = graph->nodes[i];

        if (t->view_src == c) {
            const int64_t col = concat_view_first_col(c, t);
            if (col < 0) {
                return false;
            }
            col0 = std::min(col0, col);
        }

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * s = t->src[j];
            if (s == nullptr) {
                continue;
            }
            if (s == c) {
                if (t->op == GGML_OP_SSM_CONV && j == 0 && t->src[1] && t->src[1]->ne[0] == 4 &&
                    (!need_kept || ggml_get_op_params_i32(t, SSM_CONV_PARAM_INPUTS_KEPT) == 1)) {
                    if (ssm != nullptr) {
                        return false;  // more than one ssm_conv consumer
                    }
                    ssm     = t;
                    ssm_idx = i;
                } else if (t->view_src != c) {
                    return false;  // some other consumer needs the full concat output
                }
            } else if (s->view_src == c) {
                const int64_t col = concat_view_first_col(c, s);
                if (col < 0) {
                    return false;
                }
                col0 = std::min(col0, col);
            }
        }
    }

    if (ssm == nullptr || ssm_idx <= ci) {
        return false;
    }

    // The fused kernels re-check their layouts at compute time and abort on a mismatch (there is no
    // fallback once the concat half has materialized only its trailing columns), so the gate must be
    // a superset of forward_concat_dim0_transposed_f32 + forward_ssm_conv_concat_f32: types, element
    // strides and the n_t / d_inner shape agreement between the concat and the ssm_conv output.
    const ggml_tensor * w = ssm->src[1];
    if (ssm->type != GGML_TYPE_F32 || ssm->nb[0] != sizeof(float) ||
        w->type != GGML_TYPE_F32 || w->nb[0] != sizeof(float) ||
        c->src[1]->ne[0] != ssm->ne[1] || c->ne[1] != ssm->ne[0]) {
        return false;
    }

    *ssm_out     = ssm;
    *ssm_idx_out = ssm_idx;
    *col0_out    = col0;
    return true;
}

static bool try_fuse_node(const ggml_cgraph * graph, int & i, std::vector<spacemit_opnode> & nodes,
                          std::unordered_set<ggml_tensor *> & emitted) {
    ggml_tensor * n = graph->nodes[i];

    // Second half of Pattern 2, arriving here: an ssm_conv paired with an earlier concat.
    // Emitted at its own position — opnode list order is execution order, and the graph
    // allocator planned the ssm_conv output buffer on the original node order (see Pattern 2).
    if (emitted.count(n)) {
        nodes.emplace_back(n, std::vector<ggml_tensor *>{}, SPACEMIT_OP_SSM_CONV_CONCAT);
        return true;
    }

    // RMS_NORM + MUL can skip non-compute sentinels inserted by test-backend-ops.
    // Shared-input projections below still require literal adjacency.
    int next_idx = -1;
    for (int j = i + 1; j < graph->n_nodes; ++j) {
        if (spacemit_graph_node_is_compute(graph->nodes[j])) {
            next_idx = j;
            break;
        }
    }
    ggml_tensor * next_node = (next_idx >= 0) ? graph->nodes[next_idx] : nullptr;

    // Pattern 3: adjacent shared-input projections (e.g. FFN gate/up).
    // Keep both outputs and execute at the second projection's position. Require
    // literal adjacency so no intervening node can read or reuse the first output.
    // Other consumers keep reading the original projection outputs.
    if (next_idx == i + 1 && ggml_spacemit_ffn_f16_eligible(n, next_node)) {
        spacemit_opnode node(next_node, {}, SPACEMIT_OP_MUL_MAT_FFN);
        node.add_fused(n, true);
        // SwiGLU consumes the completed projections inside the same execution node.
        // Literal adjacency keeps the allocator's original execution/lifetime order.
        if (next_idx + 1 < graph->n_nodes) {
            ggml_tensor * glu = graph->nodes[next_idx + 1];
            if (spacemit_graph_node_is_compute(glu) && glu->op == GGML_OP_GLU &&
                ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU && ggml_get_op_params_i32(glu, 1) == 0 &&
                ((glu->src[0] == n && glu->src[1] == next_node) ||
                 (glu->src[0] == next_node && glu->src[1] == n)) &&
                ggml_spacemit_get_tensor_traits(glu)) {
                node.opcode = SPACEMIT_OP_MUL_MAT_FFN_SWIGLU;
                node.add_fused(glu, true);
                ++next_idx;
            }
        }
        nodes.push_back(std::move(node));
        i = next_idx;
        return true;
    }

    // Pattern 1: RMS_NORM + MUL
    if (n->op == GGML_OP_RMS_NORM && next_node && next_node->op == GGML_OP_MUL) {
        const int          idxs[2] = { i, next_idx };
        const enum ggml_op ops[2]  = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        if (ggml_can_fuse_ext(graph, idxs, ops, 2) && rms_norm_mul_fusable(n, next_node)) {
            spacemit_opnode node(n, {}, SPACEMIT_OP_RMS_NORM_MUL);
            node.add_fused(next_node);
            nodes.push_back(std::move(node));
            i = next_idx;  // skip past the fused MUL node
            return true;
        }
    }

    // Pattern 2: CONCAT + SSM_CONV. Runs as TWO opnodes at the original node positions: the
    // concat writes only the trailing columns that views read back (the conv state update), and
    // the ssm_conv, at its own position after the state update CPY, reads the concat inputs
    // directly. The ssm_conv must NOT be pulled to the concat position: the graph allocator
    // plans buffer reuse on the original node order, so writing the ssm_conv output early can
    // overwrite tensors that are still live between the two positions (bisected on Qwen3.5 GDN,
    // 2026-10-20: fused-at-concat corrupted both 2B and 4B on K1/K3).
    if (n->op == GGML_OP_CONCAT) {
        ggml_tensor * ssm     = nullptr;
        int           ssm_idx = -1;
        int64_t       col0    = 0;
        if (ssm_conv_concat_fusable(graph, i, true, &ssm, &ssm_idx, &col0)) {
            spacemit_opnode cat(n, {}, SPACEMIT_OP_CONCAT_STATE);
            cat.kernel_params[0] = (int32_t) col0;
            nodes.push_back(std::move(cat));
            // the ssm_conv half is emitted by try_fuse_node when the loop reaches its own
            // node position (see the top of this function) — never here at the concat position
            emitted.insert(ssm);
            return true;
        }
    }

    return false;
}

} // namespace

// Pattern 2 (CONCAT + SSM_CONV) makes the ssm_conv read the concat inputs, so they must stay allocated until
// the ssm_conv has run. The fusion is only taken for ssm_convs marked here, i.e. graphs allocated by the scheduler.
void spacemit_graph_optimize(ggml_cgraph * graph, ggml_backend_graph_optimize_params * params) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i]->op == GGML_OP_SSM_CONV) {
            ggml_set_op_params_i32(graph->nodes[i], SSM_CONV_PARAM_INPUTS_KEPT, 0);
        }
    }
    if (params == nullptr || params->add_alloc_dep == nullptr) {
        return;
    }
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * c = graph->nodes[i];
        ggml_tensor * ssm     = nullptr;
        int           ssm_idx = -1;
        int64_t       col0    = 0;
        if (c->op == GGML_OP_CONCAT && ssm_conv_concat_fusable(graph, i, false, &ssm, &ssm_idx, &col0)) {
            params->add_alloc_dep(params->user_data, c->src[0], ssm);
            params->add_alloc_dep(params->user_data, c->src[1], ssm);
            ggml_set_op_params_i32(ssm, SSM_CONV_PARAM_INPUTS_KEPT, 1);
        }
    }
}

void spacemit_graph_build_opnodes(const ggml_cgraph * graph,
                                  std::vector<spacemit_opnode> & nodes) {
    std::unordered_set<ggml_tensor *> emitted;
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (!spacemit_graph_node_is_compute(node)) {
            continue;
        }
        if (!ggml_prefetch_debug_ideal_enabled() && try_fuse_node(graph, i, nodes, emitted)) {
            continue;
        }
        nodes.emplace_back(node);
    }
}

int spacemit_graph_count_fused_opnodes(const std::vector<spacemit_opnode> & nodes) {
    int count = 0;
    for (const auto & node : nodes) {
        if (!node.fused.empty() || node.opcode != SPACEMIT_OP_INVALID) {
            count++;
        }
    }
    return count;
}

namespace {

struct module_name {
    std::string stem;
    int layer = -1;
};

// Scheduler copies use backend#original-name#copy-index. Preserve the original
// name for both semantic layer suffixes and exact projection weight roles.
static std::string original_tensor_name(const ggml_tensor * tensor) {
    std::string name = tensor->name;
    for (;;) {
        const size_t first = name.find('#');
        const size_t last = name.rfind('#');
        if (first == std::string::npos || first == last) {
            break;
        }
        name = name.substr(first + 1, last - first - 1);
    }
    return name;
}

static module_name parse_module_name(const ggml_tensor * tensor) {
    const std::string name = original_tensor_name(tensor);
    const size_t dash = name.rfind('-');
    if (dash == std::string::npos || dash + 1 == name.size()) {
        return {};
    }
    int layer = 0;
    for (size_t i = dash + 1; i < name.size(); ++i) {
        const int digit = name[i] - '0';
        if (digit < 0 || digit > 9 || layer > (INT_MAX - digit) / 10) {
            return {};
        }
        layer = layer * 10 + digit;
    }
    return { name.substr(0, dash), layer };
}

// A backend split may contain the first projection GEMM but end before the
// named q/kv output (e.g. DeepSeek RoPE on CPU). Anchor that local window to
// the exact projection weight role, still validating its attn_norm dependency.
static int qkv_projection_layer(const ggml_tensor * tensor) {
    if (tensor->op != GGML_OP_MUL_MAT || !tensor->src[0]) { return -1; }
    const std::string name = original_tensor_name(tensor->src[0]);
    if (name.compare(0, 4, "blk.") != 0) { return -1; }
    const size_t dot = name.find('.', 4);
    if (dot == std::string::npos || dot == 4) { return -1; }
    int layer = 0;
    for (size_t i = 4; i < dot; ++i) {
        const int digit = name[i] - '0';
        if (digit < 0 || digit > 9 || layer > (INT_MAX - digit) / 10) { return -1; }
        layer = layer * 10 + digit;
    }
    const std::string role = name.substr(dot + 1);
    if (role == "attn_q.weight" || role == "attn_k.weight" || role == "attn_v.weight" ||
        role == "attn_qkv.weight" || role == "attn_q_a.weight" || role == "attn_kv.weight" ||
        role == "attn_kv_a_mqa.weight") {
        return layer;
    }
    return -1;
}

using opnode_indices = std::unordered_map<const ggml_tensor *, size_t>;

struct module_range {
    size_t first = SIZE_MAX;
    size_t last = 0;
    const ggml_tensor * output = nullptr;
};

// Find a window from dependencies, not from the interval between two names.
// A local projection may anchor an incomplete QKV window when the final named
// output belongs to another split. Only locally mapped operations are included.
static module_range find_module_range(const ggml_tensor * output, int layer,
                                      const char * input_name,
                                      const std::unordered_set<const ggml_tensor *> & local,
                                      const opnode_indices & indices,
                                      bool require_attention) {
    module_range range;
    bool reached_input = false;
    bool has_matmul = false;
    bool has_attention = false;
    std::vector<const ggml_tensor *> pending{output};
    std::unordered_set<const ggml_tensor *> visited;
    while (!pending.empty()) {
        const ggml_tensor * tensor = pending.back();
        pending.pop_back();
        if (!tensor || !visited.insert(tensor).second) {
            continue;
        }
        const auto name = parse_module_name(tensor);
        if (name.layer == layer && name.stem == input_name) {
            reached_input = true;
            continue;
        }
        if ((name.layer >= 0 && name.layer != layer) || !local.count(tensor)) {
            continue;
        }
        const auto it = indices.find(tensor);
        if (it != indices.end()) {
            range.first = std::min(range.first, it->second);
            range.last = std::max(range.last, it->second);
            has_matmul |= tensor->op == GGML_OP_MUL_MAT;
            has_attention |= tensor->op == GGML_OP_FLASH_ATTN_EXT || tensor->op == GGML_OP_SOFT_MAX;
        }
        for (const auto * src : tensor->src) {
            if (src) {
                pending.push_back(src);
            }
        }
    }
    if (!reached_input || !has_matmul || (require_attention && !has_attention)) {
        return {};
    }
    range.output = output;
    return range;
}

static const ggml_tensor * find_topk(const ggml_tensor * ids) {
    std::vector<const ggml_tensor *> pending{ids};
    std::unordered_set<const ggml_tensor *> visited;
    while (!pending.empty()) {
        const ggml_tensor * tensor = pending.back();
        pending.pop_back();
        if (!tensor || !visited.insert(tensor).second) {
            continue;
        }
        const auto name = parse_module_name(tensor);
        if (name.layer >= 0 && name.stem == "ffn_moe_topk" && tensor->type == GGML_TYPE_I32) {
            return tensor;
        }
        // Only trace the index dependency; never search an expert's weights.
        for (const auto * src : tensor->src) {
            if (src) {
                pending.push_back(src);
            }
        }
    }
    return nullptr;
}

static void add_prefetch_event(std::vector<spacemit_prefetch_point> & points,
                               size_t index, ggml_spacemit_module module,
                               ggml_spacemit_prefetch_phase phase, int layer,
                               const ggml_tensor * tensor) {
    auto & events = phase == GGML_SPACEMIT_PREFETCH_BEGIN ? points[index].before : points[index].after;
    events.push_back({ module, phase, layer, index, tensor });
}

} // namespace

void spacemit_graph_build_prefetch_plan(const ggml_cgraph * graph,
                                        const std::vector<spacemit_opnode> & nodes,
                                        std::vector<spacemit_prefetch_point> & points) {
    points.clear();
    points.resize(nodes.size());
    opnode_indices indices;
    for (size_t i = 0; i < nodes.size(); ++i) {
        indices.emplace(nodes[i].node, i);
        for (const auto * tensor : nodes[i].fused) {
            indices.emplace(tensor, i);
        }
    }
    std::unordered_set<const ggml_tensor *> local;
    for (int i = 0; i < graph->n_nodes; ++i) {
        local.insert(graph->nodes[i]);
    }

    std::map<std::pair<ggml_spacemit_module, int>, module_range> ranges;
    const auto merge_range = [&](ggml_spacemit_module module, int layer, const module_range & range) {
        if (range.first == SIZE_MAX) {
            return;
        }
        auto & merged = ranges[{module, layer}];
        merged.first = std::min(merged.first, range.first);
        if (!merged.output || range.last >= merged.last) {
            merged.last = range.last;
            merged.output = range.output;
        }
    };

    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * tensor = graph->nodes[i];
        const int projection_layer = qkv_projection_layer(tensor);
        if (projection_layer >= 0) {
            merge_range(GGML_SPACEMIT_MODULE_QKV, projection_layer,
                        find_module_range(tensor, projection_layer, "attn_norm", local, indices, false));
        }
        const auto name = parse_module_name(tensor);
        if (name.layer < 0) {
            continue;
        }
        const auto & stem = name.stem;
        if (stem == "Qcur" || stem == "Kcur" || stem == "Vcur" || stem == "qkv" ||
            stem == "q" || stem == "kv") {
            merge_range(GGML_SPACEMIT_MODULE_QKV, name.layer,
                        find_module_range(tensor, name.layer, "attn_norm", local, indices, false));
        } else if (stem == "kqv_out" || stem == "attn_out") {
            merge_range(GGML_SPACEMIT_MODULE_ATTENTION, name.layer,
                        find_module_range(tensor, name.layer, "attn_norm", local, indices, true));
        } else if (stem == "ffn_moe_topk" && tensor->type == GGML_TYPE_I32) {
            merge_range(GGML_SPACEMIT_MODULE_ROUTER, name.layer,
                        find_module_range(tensor, name.layer, "ffn_norm", local, indices, false));
            // top-k can be a non-compute VIEW. Signal READY at its actual local
            // producer, after the executor's existing all-worker barrier.
            const ggml_tensor * producer = tensor;
            std::unordered_set<const ggml_tensor *> visited;
            while (producer && !indices.count(producer) && local.count(producer) &&
                   ggml_op_is_empty(producer->op) && visited.insert(producer).second) {
                producer = producer->view_src ? producer->view_src : producer->src[0];
            }
            const auto it = indices.find(producer);
            if (it != indices.end()) {
                add_prefetch_event(points, it->second, GGML_SPACEMIT_MODULE_ROUTER,
                                   GGML_SPACEMIT_PREFETCH_READY, name.layer, tensor);
            }
        }

        const auto it = indices.find(tensor);
        if (it == indices.end()) {
            continue;
        }
        if (tensor->op == GGML_OP_RMS_NORM) {
            size_t last = it->second;
            const ggml_tensor * output = tensor;
            // Include the weight multiply, whether fused or a separate opnode.
            for (int j = i + 1; j < graph->n_nodes; ++j) {
                const ggml_tensor * next = graph->nodes[j];
                if (!spacemit_graph_node_is_compute(graph->nodes[j])) {
                    continue;
                }
                if (next->op == GGML_OP_MUL && (next->src[0] == tensor || next->src[1] == tensor)) {
                    const auto next_it = indices.find(next);
                    if (next_it != indices.end()) {
                        last = next_it->second;
                        output = next;
                    }
                }
                break;
            }
            add_prefetch_event(points, it->second, GGML_SPACEMIT_MODULE_RMS_NORM,
                               GGML_SPACEMIT_PREFETCH_BEGIN, name.layer, tensor);
            add_prefetch_event(points, last, GGML_SPACEMIT_MODULE_RMS_NORM,
                               GGML_SPACEMIT_PREFETCH_END, name.layer, output);
        }
    }

    // Recognize routed expert work from MUL_MAT_ID and its index dependency,
    // including scheduler copies. BEGIN is the place to wait for all weights
    // required by this layer; no worker can begin the GEMM before it returns.
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto add_expert = [&](const ggml_tensor * tensor) {
            if (tensor->op != GGML_OP_MUL_MAT_ID) {
                return;
            }
            const auto * topk = find_topk(tensor->src[2]);
            if (topk) {
                merge_range(GGML_SPACEMIT_MODULE_EXPERTS, parse_module_name(topk).layer, {i, i, tensor});
            }
        };
        add_expert(nodes[i].node);
        for (const auto * tensor : nodes[i].fused) {
            add_expert(tensor);
        }
    }
    for (const auto & item : ranges) {
        const auto & range = item.second;
        add_prefetch_event(points, range.first, item.first.first, GGML_SPACEMIT_PREFETCH_BEGIN,
                           item.first.second, nodes[range.first].node);
        add_prefetch_event(points, range.last, item.first.first, GGML_SPACEMIT_PREFETCH_END,
                           item.first.second, range.output);
    }
    // On one boundary, close windows before announcing available router IDs.
    for (auto & point : points) {
        std::stable_sort(point.after.begin(), point.after.end(),
                         [](const ggml_spacemit_prefetch_event & a, const ggml_spacemit_prefetch_event & b) {
                             return a.phase < b.phase;
                         });
    }
}
