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

    std::string require_text(
        const char* value, const std::string_view field) const
    {
        if (value == nullptr || value[0] == '\0')
            throw error(std::string(field)+" is empty");
        return value;
    }

    void require_match(
        const std::string_view expected, const char* actual,
        const std::string_view field) const
    {
        if (!expected.empty() && expected != actual)
            throw error(std::string(field)+" does not match");
    }

    void require_extent(
        const std::int32_t expected, const std::int32_t actual,
        const std::string_view field) const
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

// The artifact identity envelope every Execution plugin descriptor publishes:
// the ABI layout agreement, the identifying tag, the artifact id, and the
// fingerprint triple the loader checks.  The envelope is the only part of a
// descriptor every loader validates; the payload that follows it (model extents,
// device schedules, interaction and node tables) stays with its family loader.
//
// The interaction count is contract data, so a descriptor that carries it
// publishes it here too: the count the model contract declares and the count
// the generated tables are laid out for are then compared by the one validator
// every loader shares instead of by a constant compiled into each loader.
struct ArtifactEnvelope {
    std::uint32_t abi_version = 0;
    std::uint32_t struct_size = 0;
    std::uint32_t pointer_size = 0;
    std::uint32_t byte_order = 0;
    // ABIs that reserve a descriptor word require it to be zero; the APIs that
    // publish no such word leave `checks_reserved` false.
    bool checks_reserved = false;
    std::uint32_t reserved = 0;
    const char* abi_tag = nullptr;
    const char* artifact_id = nullptr;
    // The family's own name for the fingerprint that identifies the generated
    // revision.  Which published vocabulary a descriptor uses is a property of
    // its ABI; the loader names the spelling it reports in `identity_label`.
    const char* identity_fingerprint = nullptr;
    const char* semantic_fingerprint = nullptr;
    const char* structure_fingerprint = nullptr;
    std::uint32_t interaction_count = 0;
    bool publishes_interaction_count = false;
};

// The identity one envelope must announce.  An empty expectation member is a
// wildcard, but a descriptor member is always required to name itself: an ABI
// that publishes no such member clears the matching `publishes_*` flag instead.
struct ArtifactEnvelopeExpectation {
    std::uint32_t abi_version = 0;
    std::size_t struct_size = 0;
    std::uint32_t byte_order = 0;
    std::string_view abi_tag;
    std::string_view artifact_id;
    std::string_view identity_fingerprint;
    std::string_view semantic_fingerprint;
    std::string_view structure_fingerprint;
    std::string_view identity_label = "contract fingerprint";
    bool publishes_identity_fingerprint = true;
    bool publishes_semantic_fingerprint = true;
    // Unlike the wildcard string members, an interaction count is compared
    // exactly whenever the expectation declares it: the contract states a
    // count, never a range.
    std::uint32_t interaction_count = 0;
    bool publishes_interaction_count = false;
};

inline void validate_artifact_envelope(
    const PluginLoaderContext& context,
    const ArtifactEnvelope& envelope,
    const ArtifactEnvelopeExpectation& expectation)
{
    if (envelope.abi_version != expectation.abi_version)
        throw context.error("unsupported ABI version");
    if (envelope.struct_size < expectation.struct_size)
        throw context.error("descriptor is smaller than its ABI version");
    if (envelope.pointer_size != sizeof(void*))
        throw context.error("pointer width does not match this process");
    if (envelope.byte_order != expectation.byte_order)
        throw context.error("byte order does not match this process");
    if (envelope.checks_reserved && envelope.reserved != 0)
        throw context.error("reserved descriptor field is nonzero");

    if (context.require_text(envelope.abi_tag, "ABI tag")
            != expectation.abi_tag)
        throw context.error("ABI tag does not match");
    context.require_match(
        expectation.artifact_id,
        context.require_text(envelope.artifact_id, "artifact id").c_str(),
        "artifact id");
    if (expectation.publishes_identity_fingerprint)
        context.require_match(
            expectation.identity_fingerprint,
            context.require_text(
                envelope.identity_fingerprint,
                expectation.identity_label).c_str(),
            expectation.identity_label);
    if (expectation.publishes_semantic_fingerprint)
        context.require_match(
            expectation.semantic_fingerprint,
            context.require_text(
                envelope.semantic_fingerprint, "semantic fingerprint").c_str(),
            "semantic fingerprint");
    context.require_match(
        expectation.structure_fingerprint,
        context.require_text(
            envelope.structure_fingerprint, "structure fingerprint").c_str(),
        "structure fingerprint");
    if (expectation.publishes_interaction_count) {
        if (!envelope.publishes_interaction_count)
            throw context.error(
                "descriptor does not publish the contract interaction count");
        if (envelope.interaction_count != expectation.interaction_count)
            throw context.error(
                "descriptor interaction count "
                +std::to_string(envelope.interaction_count)
                +" does not match the contract interaction count "
                +std::to_string(expectation.interaction_count));
    } else if (envelope.publishes_interaction_count) {
        throw context.error(
            "descriptor publishes an interaction count the contract does not "
            "declare");
    }
}

// The interaction count a contract declares must be the count its fixed-size
// tables admit.  Generalizing beyond the admitted tables needs the storage,
// codegen, and orchestration the ABI does not carry, so the loader names both
// counts instead of reading past a table or silently ignoring extra
// interactions.  The counts are signed so a malformed negative declaration is
// reported as declared rather than as an unsigned wrap.
inline std::string admitted_interaction_count_error(
    const std::int64_t declared,
    const std::int64_t admitted,
    const std::string_view tables)
{
    return "interaction count "+std::to_string(declared)
        +" does not match the "+std::to_string(admitted)
        +" interactions admitted by "+std::string(tables);
}

// A capability the descriptor advertises must be the capability it exports: a
// launcher that claims an entry point and leaves the pointer null, or the other
// way round, is rejected.  Which capabilities are required is caller policy and
// stays with the family loader.
inline void require_capability_agreement(
    const PluginLoaderContext& context,
    const std::uint32_t capabilities,
    const std::uint32_t capability,
    const bool has_function,
    const std::string_view name)
{
    if (((capabilities&capability) != 0) != has_function)
        throw context.error(
            std::string(name)+" capability and function pointers disagree");
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
    const ArtifactEnvelope envelope{
        .abi_version = descriptor.abi_version,
        .struct_size = descriptor.struct_size,
        .pointer_size = descriptor.pointer_size,
        .byte_order = descriptor.byte_order,
        .checks_reserved = true,
        .reserved = descriptor.reserved,
        .abi_tag = descriptor.abi_tag,
        .artifact_id = descriptor.artifact_id,
        .identity_fingerprint = descriptor.contract_fingerprint,
        .semantic_fingerprint = descriptor.semantic_fingerprint,
        .structure_fingerprint = descriptor.structure_fingerprint,
    };
    const ArtifactEnvelopeExpectation identity{
        .abi_version = abi_version,
        .struct_size = descriptor_size,
        .byte_order = byte_order,
        .abi_tag = abi_tag,
        .artifact_id = expectation.artifact_id,
        .identity_fingerprint = expectation.contract_fingerprint,
        .semantic_fingerprint = expectation.semantic_fingerprint,
        .structure_fingerprint = expectation.structure_fingerprint,
    };
    validate_artifact_envelope(context, envelope, identity);

    if (descriptor.scalar_kind != expectation.scalar_kind
        || descriptor.scalar_size != expectation.scalar_size)
        throw context.error("scalar kind or width does not match");
    if ((descriptor.capabilities&expectation.required_capabilities)
            != expectation.required_capabilities)
        throw context.error("required launcher capabilities are missing");
    require_capability_agreement(
        context, descriptor.capabilities, forward_capability,
        descriptor.r1_forward_launch != nullptr, "launcher");
    require_capability_agreement(
        context, descriptor.capabilities, reverse_capability,
        descriptor.r1_coordinate_reverse_launch != nullptr, "launcher");
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
