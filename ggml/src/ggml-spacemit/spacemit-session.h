#ifndef SPACEMIT_SESSION_H
#define SPACEMIT_SESSION_H

#include "spacemit-opnode.h"
#include "spacemit-context.h"

#include <cstdint>
#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct spacemit_session {
    uint32_t num_cores = 0;
    int64_t  arch_id   = 0;
    int64_t  vlen      = 0;
    int64_t  tcm_size  = 0;
    bool     use_ime1  = false;
    bool     use_ime2  = false;

    std::string name = "SPACEMIT0";

    // Serialize graph_compute calls because the hardware stream is exclusive.
    std::mutex stream_mutex;

    void * workspace = nullptr;
    size_t workspace_size = 0;
    std::vector<ggml::spacemit::context> contexts;

    struct graph_cache_entry {
        uint64_t uid = 0;
        std::vector<spacemit_opnode> nodes;
    } uncached_graph;
    std::array<graph_cache_entry, 4> cached_graphs;
    size_t next_graph_cache = 0;

    spacemit_session() = default;
    ~spacemit_session();

    const char * c_name() const { return name.c_str(); }
};

#endif // SPACEMIT_SESSION_H
