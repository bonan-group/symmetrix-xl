#pragma once

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace symmetrix::execution {

// Path of the JIT cache manifest a publication writes beside its artifact.
inline std::string jit_manifest_path(const std::string& artifact_path)
{
    const std::size_t separator = artifact_path.find_last_of("/\\");
    if (separator == std::string::npos)
        return "manifest.json";
    return artifact_path.substr(0, separator + 1) + "manifest.json";
}

namespace detail {

inline const nlohmann::json* jit_manifest_member(
    const nlohmann::json& object, const char* key)
{
    if (!object.is_object())
        return nullptr;
    const auto found = object.find(key);
    return found == object.end() ? nullptr : &*found;
}

inline std::string jit_manifest_text(
    const nlohmann::json& object, const char* key)
{
    if (const auto* value = jit_manifest_member(object, key);
        value != nullptr && value->is_string())
        return value->get<std::string>();
    return {};
}

// The artifact identity block one publication records.  The host publication
// keeps the caller's build directly under `key_inputs.build`; the CUDA and HIP
// adapters nest it under `key_inputs.build.request`.  Both spellings name the
// same codegen block under `contract`.
inline const nlohmann::json* jit_manifest_identity(
    const nlohmann::json& manifest)
{
    const auto* key_inputs = jit_manifest_member(manifest, "key_inputs");
    const auto* build = key_inputs == nullptr
        ? nullptr : jit_manifest_member(*key_inputs, "build");
    if (build == nullptr)
        return nullptr;
    if (const auto* contract = jit_manifest_member(*build, "contract"))
        return contract;
    const auto* request = jit_manifest_member(*build, "request");
    return request == nullptr
        ? nullptr : jit_manifest_member(*request, "contract");
}

}  // namespace detail

// Read the artifact id the Python generator published for the artifact beside
// `artifact_path`.  The identity is reconstructed from the model contract
// nowhere: the manifest a validated publication wrote is the one record of it.
// Throws when the manifest or its identity block cannot be read, because an
// artifact that has neither did not come from a validated publication.
inline std::string jit_manifest_artifact_id(const std::string& artifact_path)
{
    const auto manifest_path = jit_manifest_path(artifact_path);
    std::ifstream manifest_stream(manifest_path);
    if (!manifest_stream)
        throw std::runtime_error(
            "the JIT manifest is missing: " + manifest_path);
    nlohmann::json manifest;
    try {
        manifest_stream >> manifest;
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "the JIT manifest is invalid JSON (" + std::string(error.what())
            + "): " + manifest_path);
    }
    const auto* identity = detail::jit_manifest_identity(manifest);
    if (identity == nullptr)
        throw std::runtime_error(
            "the JIT manifest does not record its artifact identity: "
            + manifest_path);
    const auto artifact_id = detail::jit_manifest_text(*identity, "artifact_id");
    if (artifact_id.empty())
        throw std::runtime_error(
            "the JIT manifest records no artifact id: " + manifest_path);
    return artifact_id;
}

}  // namespace symmetrix::execution
