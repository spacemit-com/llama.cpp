#pragma once

#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace qwen3_tts {

class mapped_gguf {
  public:
    explicit mapped_gguf(const std::string & path) : path_(path) {
        fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) {
            throw std::runtime_error("failed to open GGUF: " + path);
        }
        struct stat st{};
        if (fstat(fd_, &st) != 0 || st.st_size <= 0) {
            close(fd_);
            fd_ = -1;
            throw std::runtime_error("failed to stat GGUF: " + path);
        }
        size_    = static_cast<size_t>(st.st_size);
        mapping_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapping_ == MAP_FAILED) {
            mapping_ = nullptr;
            close(fd_);
            fd_ = -1;
            throw std::runtime_error("failed to mmap GGUF: " + path);
        }
        gguf_init_params params{};
        params.no_alloc = true;
        context_        = gguf_init_from_file(path.c_str(), params);
        if (context_ == nullptr) {
            munmap(mapping_, size_);
            mapping_ = nullptr;
            close(fd_);
            fd_ = -1;
            throw std::runtime_error("failed to parse GGUF: " + path);
        }
    }

    mapped_gguf(const mapped_gguf &)             = delete;
    mapped_gguf & operator=(const mapped_gguf &) = delete;

    ~mapped_gguf() {
        if (context_ != nullptr) {
            gguf_free(context_);
        }
        if (mapping_ != nullptr) {
            munmap(mapping_, size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    const void * tensor(const std::string & name, ggml_type type, size_t bytes) const {
        const int64_t id = gguf_find_tensor(context_, name.c_str());
        if (id < 0) {
            throw std::runtime_error("missing tensor '" + name + "' in " + path_);
        }
        if (gguf_get_tensor_type(context_, id) != type || gguf_get_tensor_size(context_, id) != bytes) {
            throw std::runtime_error("unexpected tensor layout for '" + name + "' in " + path_);
        }
        const size_t offset = gguf_get_data_offset(context_) + gguf_get_tensor_offset(context_, id);
        if (offset > size_ || bytes > size_ - offset) {
            throw std::runtime_error("tensor '" + name + "' exceeds GGUF bounds");
        }
        return static_cast<const uint8_t *>(mapping_) + offset;
    }

  private:
    std::string    path_;
    int            fd_      = -1;
    size_t         size_    = 0;
    void *         mapping_ = nullptr;
    gguf_context * context_ = nullptr;
};

}  // namespace qwen3_tts
