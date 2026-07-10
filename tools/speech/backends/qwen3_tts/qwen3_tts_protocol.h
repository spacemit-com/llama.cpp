#pragma once

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace qwen3_tts::protocol {

constexpr uint32_t magic = 0x51545453;
constexpr uint16_t version = 1;
constexpr uint64_t max_payload_size = 512ULL * 1024ULL * 1024ULL;

enum class message_type : uint16_t {
    ready = 1,
    synthesize = 2,
    synthesis = 3,
    error = 4,
    shutdown = 5,
    talker_job = 6,
    talker_frame = 7,
    talker_done = 8,
};

struct message {
    message_type type;
    std::vector<uint8_t> payload;
};

inline void put_u16(uint8_t * dst, uint16_t value) {
    dst[0] = static_cast<uint8_t>(value);
    dst[1] = static_cast<uint8_t>(value >> 8);
}

inline void put_u32(uint8_t * dst, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        dst[i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

inline void put_u64(uint8_t * dst, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        dst[i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

inline uint16_t get_u16(const uint8_t * src) {
    return static_cast<uint16_t>(src[0]) |
           static_cast<uint16_t>(src[1]) << 8;
}

inline uint32_t get_u32(const uint8_t * src) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        value |= static_cast<uint32_t>(src[i]) << (8 * i);
    }
    return value;
}

inline uint64_t get_u64(const uint8_t * src) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(src[i]) << (8 * i);
    }
    return value;
}

inline void write_all(int fd, const void * data, size_t size) {
    const auto * src = static_cast<const uint8_t *>(data);
    while (size > 0) {
        const ssize_t n = write(fd, src, size);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            throw std::runtime_error("Qwen3-TTS protocol write failed");
        }
        src += n;
        size -= static_cast<size_t>(n);
    }
}

inline bool read_all(int fd, void * data, size_t size, bool allow_initial_eof = false) {
    auto * dst = static_cast<uint8_t *>(data);
    size_t total = 0;
    while (total < size) {
        const ssize_t n = read(fd, dst + total, size - total);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n == 0 && allow_initial_eof && total == 0) {
            return false;
        }
        if (n <= 0) {
            throw std::runtime_error("Qwen3-TTS protocol read failed");
        }
        total += static_cast<size_t>(n);
    }
    return true;
}

inline void send(int fd, message_type type, const void * data, size_t size) {
    if (size > max_payload_size) {
        throw std::runtime_error("Qwen3-TTS protocol payload is too large");
    }
    std::array<uint8_t, 16> header{};
    put_u32(header.data(), magic);
    put_u16(header.data() + 4, version);
    put_u16(header.data() + 6, static_cast<uint16_t>(type));
    put_u64(header.data() + 8, size);
    write_all(fd, header.data(), header.size());
    if (size > 0) {
        write_all(fd, data, size);
    }
}

inline void send(int fd, message_type type, const std::vector<uint8_t> & payload) {
    send(fd, type, payload.data(), payload.size());
}

inline void send(int fd, message_type type, const std::string & payload) {
    send(fd, type, payload.data(), payload.size());
}

inline void send(int fd, message_type type) {
    send(fd, type, nullptr, 0);
}

inline bool receive(int fd, message & result) {
    std::array<uint8_t, 16> header{};
    if (!read_all(fd, header.data(), header.size(), true)) {
        return false;
    }
    if (get_u32(header.data()) != magic || get_u16(header.data() + 4) != version) {
        throw std::runtime_error("invalid Qwen3-TTS protocol header");
    }
    const uint64_t size = get_u64(header.data() + 8);
    if (size > max_payload_size) {
        throw std::runtime_error("Qwen3-TTS protocol payload exceeds the limit");
    }
    result.type = static_cast<message_type>(get_u16(header.data() + 6));
    result.payload.resize(static_cast<size_t>(size));
    if (size > 0) {
        read_all(fd, result.payload.data(), result.payload.size());
    }
    return true;
}

inline void append_u32(std::vector<uint8_t> & out, uint32_t value) {
    const size_t offset = out.size();
    out.resize(offset + 4);
    put_u32(out.data() + offset, value);
}

inline void append_u64(std::vector<uint8_t> & out, uint64_t value) {
    const size_t offset = out.size();
    out.resize(offset + 8);
    put_u64(out.data() + offset, value);
}

inline void append_f64(std::vector<uint8_t> & out, double value) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected double size");
    std::memcpy(&bits, &value, sizeof(bits));
    append_u64(out, bits);
}

inline void append_bytes(std::vector<uint8_t> & out, const void * data, size_t size) {
    const size_t offset = out.size();
    out.resize(offset + size);
    if (size > 0) {
        std::memcpy(out.data() + offset, data, size);
    }
}

class reader {
  public:
    explicit reader(const std::vector<uint8_t> & data) : data_(data) {}

    uint32_t u32() {
        const auto * p = bytes(4);
        return get_u32(p);
    }

    uint64_t u64() {
        const auto * p = bytes(8);
        return get_u64(p);
    }

    double f64() {
        const uint64_t bits = u64();
        double value = 0.0;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    const uint8_t * bytes(size_t size) {
        if (size > data_.size() - offset_) {
            throw std::runtime_error("truncated Qwen3-TTS protocol payload");
        }
        const auto * result = data_.data() + offset_;
        offset_ += size;
        return result;
    }

    size_t remaining() const {
        return data_.size() - offset_;
    }

  private:
    const std::vector<uint8_t> & data_;
    size_t offset_ = 0;
};

}  // namespace qwen3_tts::protocol
