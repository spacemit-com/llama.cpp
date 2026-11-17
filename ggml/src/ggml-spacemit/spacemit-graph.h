#ifndef SPACEMIT_GRAPH_H
#define SPACEMIT_GRAPH_H

#include "spacemit-opnode.h"

// Register allocation dependencies before graph allocation.
void spacemit_graph_optimize(ggml_cgraph * graph, ggml_backend_graph_optimize_params * params);

// Build the execution plan, applying fusion patterns internally.
void spacemit_graph_build_opnodes(const ggml_cgraph * graph,
                                  std::vector<spacemit_opnode> & nodes);

int spacemit_graph_count_fused_opnodes(const std::vector<spacemit_opnode> & nodes);

#endif // SPACEMIT_GRAPH_H
