#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace sha256 {

// Small self-contained SHA-256 implementation used for deterministic artifact
// and table identities.  `finish` returns "sha256:" followed by the 64
// lowercase hexadecimal digits, which is the identity spelling shared with the
// Python exporter.
class Digest {
public:
    Digest() : state_{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u} {}

    void update(const std::string& text)
    {
        for (const unsigned char byte : text) update_byte(byte);
    }

    std::string finish()
    {
        const std::uint64_t bit_count = total_bytes_ * 8u;
        update_byte(0x80u);
        while (buffer_size_ != 56) update_byte(0u);
        for (int shift = 56; shift >= 0; shift -= 8)
            update_byte(static_cast<unsigned char>(bit_count >> shift));
        std::ostringstream output;
        output << "sha256:" << std::hex << std::setfill('0');
        for (const auto value : state_) output << std::setw(8) << value;
        return output.str();
    }

private:
    static constexpr std::array<std::uint32_t, 64> round_constants = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,
        0x923f82a4u,0xab1c5ed5u,0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
        0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,0xe49b69c1u,0xefbe4786u,
        0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,
        0x06ca6351u,0x14292967u,0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
        0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,0xa2bfe8a1u,0xa81a664bu,
        0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,
        0x5b9cca4fu,0x682e6ff3u,0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
        0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};

    static std::uint32_t rotate_right(std::uint32_t value, int count)
    {
        return (value >> count) | (value << (32-count));
    }

    void update_byte(unsigned char byte)
    {
        buffer_[buffer_size_++] = byte;
        ++total_bytes_;
        if (buffer_size_ == buffer_.size()) {
            transform();
            buffer_size_ = 0;
        }
    }

    void transform()
    {
        std::array<std::uint32_t, 64> schedule{};
        for (int index = 0; index < 16; ++index)
            schedule[index] = (static_cast<std::uint32_t>(buffer_[4*index]) << 24)
                | (static_cast<std::uint32_t>(buffer_[4*index+1]) << 16)
                | (static_cast<std::uint32_t>(buffer_[4*index+2]) << 8)
                | static_cast<std::uint32_t>(buffer_[4*index+3]);
        for (int index = 16; index < 64; ++index) {
            const auto s0 = rotate_right(schedule[index-15], 7)
                ^ rotate_right(schedule[index-15], 18) ^ (schedule[index-15] >> 3);
            const auto s1 = rotate_right(schedule[index-2], 17)
                ^ rotate_right(schedule[index-2], 19) ^ (schedule[index-2] >> 10);
            schedule[index] = schedule[index-16] + s0 + schedule[index-7] + s1;
        }
        auto working = state_;
        for (int index = 0; index < 64; ++index) {
            const auto s1 = rotate_right(working[4], 6)
                ^ rotate_right(working[4], 11) ^ rotate_right(working[4], 25);
            const auto choose = (working[4] & working[5]) ^ ((~working[4]) & working[6]);
            const auto temp1 = working[7] + s1 + choose + round_constants[index] + schedule[index];
            const auto s0 = rotate_right(working[0], 2)
                ^ rotate_right(working[0], 13) ^ rotate_right(working[0], 22);
            const auto majority = (working[0] & working[1])
                ^ (working[0] & working[2]) ^ (working[1] & working[2]);
            const auto temp2 = s0 + majority;
            working = {temp1+temp2, working[0], working[1], working[2],
                working[3]+temp1, working[4], working[5], working[6]};
        }
        for (int index = 0; index < 8; ++index) state_[index] += working[index];
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<unsigned char, 64> buffer_{};
    std::size_t buffer_size_ = 0;
    std::uint64_t total_bytes_ = 0;
};

}  // namespace sha256
