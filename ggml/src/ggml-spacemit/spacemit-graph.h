#ifndef SPACEMIT_GRAPH_H
#define SPACEMIT_GRAPH_H

#include "spacemit-opnode.h"
#include "ggml-spacemit.h"

// Register allocation dependencies before graph allocation.
void spacemit_graph_optimize(ggml_cgraph * graph, ggml_backend_graph_optimize_params * params);

// Build the execution plan, applying fusion patterns internally.
void spacemit_graph_build_opnodes(const ggml_cgraph * graph,
                                  std::vector<spacemit_opnode> & nodes);

int spacemit_graph_count_fused_opnodes(const std::vector<spacemit_opnode> & nodes);

// Events attach to existing opnode boundaries, without consuming graph nodes.
struct spacemit_prefetch_point {
    std::vector<ggml_spacemit_prefetch_event> before;
    std::vector<ggml_spacemit_prefetch_event> after;
};

void spacemit_graph_build_prefetch_plan(const ggml_cgraph * graph,
                                        const std::vector<spacemit_opnode> & nodes,
                                        std::vector<spacemit_prefetch_point> & points);

#endif // SPACEMIT_GRAPH_H
