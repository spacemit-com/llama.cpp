#ifndef SPACEMIT_SESSION_H
#define SPACEMIT_SESSION_H

#include "spacemit-graph.h"
#include "spacemit-context.h"

#include <cstdint>
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

    struct {
        uint64_t uid = 0;
        std::vector<spacemit_opnode> nodes;
        bool prefetch_valid = false;
        std::vector<spacemit_prefetch_point> prefetch;
    } cached_graph;

    bool prefetch_enabled = false;
    ggml_spacemit_prefetch_callback prefetch_callback = nullptr;
    void * prefetch_user_data = nullptr;

    spacemit_session() = default;
    ~spacemit_session();

    const char * c_name() const { return name.c_str(); }
};

#endif // SPACEMIT_SESSION_H
