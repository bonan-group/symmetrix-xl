#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace symmetrix::execution {

// Identity a generated CUDA R1 artifact publishes.  The codegen artifact id
// and the semantic/structure fingerprints are hashes over the Python codegen
// layer layouts, so only the code generator can produce them; the JIT
// manifest records the subset the adapter published.  Every field left empty
// here is filled from the manifest at `manifest_path`, which defaults to
// "manifest.json" next to the plugin.
//
// `contract_fingerprint` is optional because the evaluator always derives the
// expectation's value from the model's canonical contract fingerprint; when it
// is supplied it must equal that fingerprint, or the artifact is rejected as
// generated from a different contract.  `target_architecture` is the canonical
// artifact architecture ("sm120"), `target_compute_capability` its numeric
// form (filled from the active device when absent), and `max_threads_per_block`
// the artifact's published thread-block extent, which must equal the generated
// descriptor's value rather than the device limit.
struct CudaR1ArtifactIdentity {
    std::string artifact_id;
    std::string contract_fingerprint;
    std::string semantic_fingerprint;
    std::string structure_fingerprint;
    std::string target_architecture;
    std::string manifest_path;
    std::int32_t target_compute_capability = -1;
    std::int32_t max_threads_per_block = -1;
    std::string target_policy;
    std::string precision_policy;
    std::string schedule_qualification;
};

// The artifact publishes the canonical architecture spelling ("sm120") while
// the device environment reports the CUDA spelling ("sm_120").
inline std::string canonical_cuda_target_architecture(const std::string& architecture)
{
    std::string canonical;
    canonical.reserve(architecture.size());
    for (const char character : architecture)
        if (character != '_')
            canonical.push_back(character);
    return canonical;
}

// Fill every identity field `target` left empty from `source`; an explicitly
// supplied field is authoritative and is never overwritten.
inline void merge_cuda_r1_identity(
    CudaR1ArtifactIdentity& target, const CudaR1ArtifactIdentity& source)
{
    if (target.artifact_id.empty())
        target.artifact_id = source.artifact_id;
    if (target.contract_fingerprint.empty())
        target.contract_fingerprint = source.contract_fingerprint;
    if (target.semantic_fingerprint.empty())
        target.semantic_fingerprint = source.semantic_fingerprint;
    if (target.structure_fingerprint.empty())
        target.structure_fingerprint = source.structure_fingerprint;
    if (target.target_architecture.empty())
        target.target_architecture = source.target_architecture;
    if (target.target_compute_capability <= 0)
        target.target_compute_capability = source.target_compute_capability;
    if (target.max_threads_per_block <= 0)
        target.max_threads_per_block = source.max_threads_per_block;
#define SYMMETRIX_MERGE_CUDA_TEXT(field) if (target.field.empty()) target.field = source.field
    SYMMETRIX_MERGE_CUDA_TEXT(target_policy);
    SYMMETRIX_MERGE_CUDA_TEXT(precision_policy);
    SYMMETRIX_MERGE_CUDA_TEXT(schedule_qualification);
#undef SYMMETRIX_MERGE_CUDA_TEXT
}

inline std::vector<std::string> cuda_r1_identity_mismatches(
    const CudaR1ArtifactIdentity& requested,
    const CudaR1ArtifactIdentity& published)
{
    std::vector<std::string> mismatches;
#define SYMMETRIX_COMPARE_CUDA_TEXT(field) \
    if (!requested.field.empty() && !published.field.empty() \
        && requested.field != published.field) mismatches.emplace_back(#field)
    SYMMETRIX_COMPARE_CUDA_TEXT(artifact_id);
    SYMMETRIX_COMPARE_CUDA_TEXT(contract_fingerprint);
    SYMMETRIX_COMPARE_CUDA_TEXT(semantic_fingerprint);
    SYMMETRIX_COMPARE_CUDA_TEXT(structure_fingerprint);
    SYMMETRIX_COMPARE_CUDA_TEXT(target_architecture);
    SYMMETRIX_COMPARE_CUDA_TEXT(target_policy);
    SYMMETRIX_COMPARE_CUDA_TEXT(precision_policy);
    SYMMETRIX_COMPARE_CUDA_TEXT(schedule_qualification);
#undef SYMMETRIX_COMPARE_CUDA_TEXT
#define SYMMETRIX_COMPARE_CUDA_EXTENT(field) \
    if (requested.field > 0 && published.field > 0 \
        && requested.field != published.field) mismatches.emplace_back(#field)
    SYMMETRIX_COMPARE_CUDA_EXTENT(target_compute_capability);
    SYMMETRIX_COMPARE_CUDA_EXTENT(max_threads_per_block);
#undef SYMMETRIX_COMPARE_CUDA_EXTENT
    return mismatches;
}

namespace detail {

inline const nlohmann::json* cuda_r1_manifest_member(
    const nlohmann::json& object, const char* key)
{
    if (!object.is_object())
        return nullptr;
    const auto found = object.find(key);
    return found == object.end() ? nullptr : &*found;
}

inline std::string cuda_r1_manifest_string(
    const nlohmann::json& object, const char* key)
{
    if (const auto* value = cuda_r1_manifest_member(object, key);
        value != nullptr && value->is_string())
        return value->get<std::string>();
    return {};
}

inline std::int32_t cuda_r1_manifest_extent(
    const nlohmann::json& object, const char* key)
{
    if (const auto* value = cuda_r1_manifest_member(object, key);
        value != nullptr && value->is_number_integer())
        return value->get<std::int32_t>();
    return -1;
}

// Read the identity one published manifest object records.  The CUDA adapter
// records the bound contract identity, and the code generator's metadata block
// is the only place the codegen artifact id and the semantic/structure
// fingerprints appear; both keys and both spellings of the target architecture
// are accepted.
inline CudaR1ArtifactIdentity cuda_r1_manifest_identity(
    const nlohmann::json& object)
{
    CudaR1ArtifactIdentity identity;
    identity.artifact_id = cuda_r1_manifest_string(object, "artifact_id");
    identity.contract_fingerprint =
        cuda_r1_manifest_string(object, "contract_fingerprint");
    identity.semantic_fingerprint =
        cuda_r1_manifest_string(object, "semantic_fingerprint");
    identity.structure_fingerprint =
        cuda_r1_manifest_string(object, "structure_fingerprint");
    identity.target_architecture =
        cuda_r1_manifest_string(object, "target_architecture");
    if (identity.target_architecture.empty())
        identity.target_architecture = cuda_r1_manifest_string(object, "architecture");
    identity.target_compute_capability =
        cuda_r1_manifest_extent(object, "target_compute_capability");
    if (identity.target_compute_capability <= 0) {
        const auto capability =
            cuda_r1_manifest_extent(object, "compute_capability_code");
        identity.target_compute_capability = capability;
    }
    identity.max_threads_per_block =
        cuda_r1_manifest_extent(object, "max_threads_per_block");
#define SYMMETRIX_READ_CUDA_TEXT(field) identity.field = cuda_r1_manifest_string(object, #field)
    SYMMETRIX_READ_CUDA_TEXT(target_policy);
    SYMMETRIX_READ_CUDA_TEXT(precision_policy);
    SYMMETRIX_READ_CUDA_TEXT(schedule_qualification);
#undef SYMMETRIX_READ_CUDA_TEXT
    return identity;
}

}  // namespace detail

// Collect the identity the JIT manifest published for this artifact.  Returns
// false with a diagnostic when the manifest cannot be read or records no build
// identity at all.  `request_key` names the adapter's identity block below the
// manifest's build request (for example the architecture-specific receiver or
// the JIT plugin family).
inline bool read_cuda_r1_manifest_identity(
    const std::string& manifest_path,
    const char* request_key,
    CudaR1ArtifactIdentity& identity,
    std::string& diagnostic)
{
    std::ifstream manifest_stream(manifest_path);
    if (!manifest_stream) {
        diagnostic = "the JIT manifest is missing: " + manifest_path;
        return false;
    }
    nlohmann::json manifest;
    try {
        manifest_stream >> manifest;
    } catch (const std::exception& error) {
        diagnostic = "the JIT manifest is invalid JSON ("
            + std::string(error.what()) + "): " + manifest_path;
        return false;
    }
    const auto* key_inputs =
        detail::cuda_r1_manifest_member(manifest, "key_inputs");
    const auto* build = key_inputs == nullptr
        ? nullptr
        : detail::cuda_r1_manifest_member(*key_inputs, "build");
    if (build == nullptr) {
        diagnostic =
            "the JIT manifest does not record its build identity: " + manifest_path;
        return false;
    }
    if (const auto* metadata = detail::cuda_r1_manifest_member(*build, "metadata"))
        merge_cuda_r1_identity(
            identity, detail::cuda_r1_manifest_identity(*metadata));
    if (const auto* request = detail::cuda_r1_manifest_member(*build, "request"))
        if (const auto* adapter =
                detail::cuda_r1_manifest_member(*request, request_key))
            merge_cuda_r1_identity(
                identity, detail::cuda_r1_manifest_identity(*adapter));
    return true;
}

// Fields a validated expectation requires that the artifact identity does not
// publish yet.
inline std::vector<std::string> missing_cuda_r1_identity_fields(
    const CudaR1ArtifactIdentity& identity)
{
    std::vector<std::string> missing;
    if (identity.artifact_id.empty())
        missing.emplace_back("artifact_id");
    if (identity.semantic_fingerprint.empty())
        missing.emplace_back("semantic_fingerprint");
    if (identity.structure_fingerprint.empty())
        missing.emplace_back("structure_fingerprint");
    if (identity.target_architecture.empty())
        missing.emplace_back("target_architecture");
    if (identity.target_compute_capability <= 0)
        missing.emplace_back("target_compute_capability");
    if (identity.max_threads_per_block <= 0)
        missing.emplace_back("max_threads_per_block");
#define SYMMETRIX_MISSING_CUDA_TEXT(field) if (identity.field.empty()) missing.emplace_back(#field)
    SYMMETRIX_MISSING_CUDA_TEXT(target_policy);
    SYMMETRIX_MISSING_CUDA_TEXT(precision_policy);
    SYMMETRIX_MISSING_CUDA_TEXT(schedule_qualification);
#undef SYMMETRIX_MISSING_CUDA_TEXT
    return missing;
}

inline std::string join_cuda_r1_fields(const std::vector<std::string>& fields)
{
    std::string joined;
    for (const auto& field : fields) {
        if (!joined.empty())
            joined += ", ";
        joined += field;
    }
    return joined;
}

}  // namespace symmetrix::execution
