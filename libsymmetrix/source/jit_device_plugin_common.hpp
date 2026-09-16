#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace symmetrix::execution::detail {

class PluginLoaderContext {
public:
    PluginLoaderContext(std::string_view backend, const std::string& path)
        : backend_(backend), path_(path)
    {}

    std::runtime_error error(const std::string& message) const
    {
        return std::runtime_error(
            "Could not load Execution "+std::string(backend_)+" plugin '"
            +path_+"': "+message);
    }

    std::string require_text(const char* value, const char* field) const
    {
        if (value == nullptr || value[0] == '\0')
            throw error(std::string(field)+" is empty");
        return value;
    }

    void require_match(
        const std::string_view expected, const char* actual,
        const char* field) const
    {
        if (!expected.empty() && expected != actual)
            throw error(std::string(field)+" does not match");
    }

    void require_extent(
        const std::int32_t expected, const std::int32_t actual,
        const char* field) const
    {
        if (expected >= 0 && expected != actual)
            throw error(std::string(field)+" does not match");
    }

    std::vector<char> read_binary(const std::string_view artifact_kind) const
    {
        std::ifstream input(path_, std::ios::binary | std::ios::ate);
        if (!input)
            throw error("could not open "+std::string(artifact_kind));
        const auto end = input.tellg();
        if (end <= 0)
            throw error(std::string(artifact_kind)+" is empty");
        if (static_cast<unsigned long long>(end)
            > static_cast<unsigned long long>(
                std::numeric_limits<std::size_t>::max()))
            throw error(std::string(artifact_kind)+" is too large");
        std::vector<char> bytes(static_cast<std::size_t>(end));
        input.seekg(0);
        if (!input.read(
                bytes.data(), static_cast<std::streamsize>(bytes.size())))
            throw error("could not read "+std::string(artifact_kind));
        return bytes;
    }

    std::vector<char> read_elf_binary(
        const std::string_view artifact_kind) const
    {
        auto bytes = read_binary(artifact_kind);
        constexpr std::size_t elf64_header_size = 64;
        if (bytes.size() < elf64_header_size
            || static_cast<unsigned char>(bytes[0]) != 0x7f
            || bytes[1] != 'E' || bytes[2] != 'L' || bytes[3] != 'F'
            || static_cast<unsigned char>(bytes[4]) != 2
            || static_cast<unsigned char>(bytes[5]) != 1)
            throw error(
                std::string(artifact_kind)+" is not a little-endian ELF64 file");

        const auto read_unsigned = [&bytes] (
            const std::size_t offset, const std::size_t width) {
            if (offset > bytes.size() || width > bytes.size()-offset)
                throw std::out_of_range("ELF field is outside the artifact");
            std::uint64_t value = 0;
            for (std::size_t byte=0; byte<width; ++byte)
                value |= static_cast<std::uint64_t>(
                    static_cast<unsigned char>(bytes[offset+byte]))<<(8*byte);
            return value;
        };
        const auto range_is_valid = [&bytes] (
            const std::uint64_t offset, const std::uint64_t extent) {
            return offset <= bytes.size() && extent <= bytes.size()-offset;
        };
        try {
            const auto program_offset = read_unsigned(32, 8);
            const auto section_offset = read_unsigned(40, 8);
            const auto header_size = read_unsigned(52, 2);
            const auto program_entry_size = read_unsigned(54, 2);
            const auto program_count = read_unsigned(56, 2);
            const auto section_entry_size = read_unsigned(58, 2);
            const auto section_count = read_unsigned(60, 2);
            if (header_size != elf64_header_size || program_count == 0xffff
                || section_count == 0 || program_entry_size != 56
                || section_entry_size != 64
                || !range_is_valid(
                    program_offset, program_entry_size*program_count)
                || !range_is_valid(
                    section_offset, section_entry_size*section_count))
                throw error(
                    std::string(artifact_kind)+" has invalid or truncated ELF tables");
            for (std::uint64_t index=0; index<program_count; ++index) {
                const auto base = program_offset+index*program_entry_size;
                const auto offset = read_unsigned(base+8, 8);
                const auto file_size = read_unsigned(base+32, 8);
                if (!range_is_valid(offset, file_size))
                    throw error(
                        std::string(artifact_kind)
                        +" has an invalid or truncated ELF segment");
            }
            for (std::uint64_t index=0; index<section_count; ++index) {
                const auto base = section_offset+index*section_entry_size;
                const auto type = read_unsigned(base+4, 4);
                const auto offset = read_unsigned(base+24, 8);
                const auto size = read_unsigned(base+32, 8);
                if (type != 8 && !range_is_valid(offset, size))
                    throw error(
                        std::string(artifact_kind)
                        +" has an invalid or truncated ELF section");
            }
        } catch (const std::out_of_range&) {
            throw error(
                std::string(artifact_kind)+" has an invalid or truncated ELF header");
        }
        return bytes;
    }

private:
    std::string_view backend_;
    const std::string& path_;
};

inline std::int32_t launch_blocks(
    const std::int64_t work_items,
    const std::int32_t threads_per_block,
    const std::int32_t persistent_blocks)
{
    const std::int64_t required =
        work_items/threads_per_block
        +(work_items%threads_per_block != 0);
    return static_cast<std::int32_t>(
        required < persistent_blocks ? required : persistent_blocks);
}

template<class Descriptor, class Expectation>
void validate_descriptor_common(
    const PluginLoaderContext& context,
    const Descriptor& descriptor,
    const Expectation& expectation,
    const std::uint32_t abi_version,
    const std::size_t descriptor_size,
    const std::uint32_t byte_order,
    const std::string_view abi_tag,
    const std::uint32_t forward_capability,
    const std::uint32_t reverse_capability)
{
    if (descriptor.abi_version != abi_version)
        throw context.error("unsupported ABI version");
    if (descriptor.struct_size < descriptor_size)
        throw context.error("descriptor is smaller than its ABI version");
    if (descriptor.pointer_size != sizeof(void*))
        throw context.error("pointer width does not match this process");
    if (descriptor.byte_order != byte_order)
        throw context.error("byte order does not match this process");
    if (descriptor.reserved != 0)
        throw context.error("reserved descriptor field is nonzero");
    if (descriptor.scalar_kind != expectation.scalar_kind
        || descriptor.scalar_size != expectation.scalar_size)
        throw context.error("scalar kind or width does not match");
    if ((descriptor.capabilities&expectation.required_capabilities)
            != expectation.required_capabilities)
        throw context.error("required launcher capabilities are missing");
    if ((descriptor.capabilities&forward_capability)
            != (descriptor.r1_forward_launch != nullptr
                ? forward_capability : 0u)
        || (descriptor.capabilities&reverse_capability)
            != (descriptor.r1_coordinate_reverse_launch != nullptr
                ? reverse_capability : 0u))
        throw context.error(
            "launcher capability and function pointers disagree");
    if (descriptor.forward_threads_per_block <= 0
        || descriptor.source_threads_per_block <= 0
        || descriptor.edge_threads_per_block <= 0
        || descriptor.forward_threads_per_block
            > expectation.max_threads_per_block
        || descriptor.source_threads_per_block
            > expectation.max_threads_per_block
        || descriptor.edge_threads_per_block
            > expectation.max_threads_per_block)
        throw context.error("thread-block size is invalid for this device");
    if (descriptor.channels <= 0 || descriptor.embedding <= 0
        || descriptor.edge_l_max < 0 || descriptor.source_l_max < 0)
        throw context.error("model extents are invalid");

    if (context.require_text(descriptor.abi_tag, "ABI tag") != abi_tag)
        throw context.error("ABI tag does not match");
    context.require_match(
        expectation.artifact_id,
        context.require_text(descriptor.artifact_id, "artifact id").c_str(),
        "artifact id");
    context.require_match(
        expectation.contract_fingerprint,
        context.require_text(
            descriptor.contract_fingerprint, "contract fingerprint").c_str(),
        "contract fingerprint");
    context.require_match(
        expectation.semantic_fingerprint,
        context.require_text(
            descriptor.semantic_fingerprint, "semantic fingerprint").c_str(),
        "semantic fingerprint");
    context.require_match(
        expectation.structure_fingerprint,
        context.require_text(
            descriptor.structure_fingerprint, "structure fingerprint").c_str(),
        "structure fingerprint");
    context.require_extent(
        expectation.channels, descriptor.channels, "channel count");
    context.require_extent(
        expectation.embedding, descriptor.embedding, "embedding width");
    context.require_extent(
        expectation.edge_l_max, descriptor.edge_l_max, "edge l_max");
    context.require_extent(
        expectation.source_l_max, descriptor.source_l_max, "source l_max");
}

} // namespace symmetrix::execution::detail
