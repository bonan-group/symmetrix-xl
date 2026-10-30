#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "sha256.hpp"

namespace symmetrix::runtime {

struct OpenMPRuntimeSnapshot {
    std::string loaded_runtime_path;
    std::vector<std::string> loaded_openmp_libraries;
};

inline std::string normalized_runtime_sha256(const std::string& value)
{
    if (value.size() != 64)
        return {};
    std::string result = value;
    for (char& digit : result) {
        if (digit >= 'A' && digit <= 'F')
            digit += 'a'-'A';
        else if (!((digit >= '0' && digit <= '9')
                || (digit >= 'a' && digit <= 'f')))
            return {};
    }
    return result;
}

inline std::string runtime_file_sha256(const std::string& path)
{
    if (path.empty())
        return {};
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
        return {};
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return {};
    sha256::Digest digest;
    std::array<char, 64*1024> buffer{};
    while (stream) {
        stream.read(buffer.data(), buffer.size());
        const auto count = stream.gcount();
        if (count > 0)
            digest.update(std::string(
                buffer.data(), static_cast<std::size_t>(count)));
    }
    if (stream.bad() || !stream.eof())
        return {};
    return digest.finish().substr(7);
}

inline bool is_auditwheel_copy_of_expected_runtime(const std::string& path,
    const std::string& expected_path, const std::string& expected_sha256)
{
    const auto expected_hash = normalized_runtime_sha256(expected_sha256);
    if (expected_hash.empty())
        return false;
    auto filename = std::filesystem::path(path).filename().string();
    const auto so = filename.find(".so");
    const auto dash = filename.rfind('-', so);
    if (so == std::string::npos || dash == std::string::npos
        || so-dash-1 != 8)
        return false;
    auto prefix = filename.substr(dash+1, 8);
    std::transform(prefix.begin(), prefix.end(), prefix.begin(),
        [] (const unsigned char value) { return std::tolower(value); });
    if (!std::all_of(prefix.begin(), prefix.end(), [] (const unsigned char value) {
            return std::isxdigit(value);
        }))
        return false;
    auto expected_filename = std::filesystem::path(
        expected_path).filename().string();
    const auto expected_so = expected_filename.find(".so");
    if (expected_so == std::string::npos)
        return false;
    auto loaded_stem = filename.substr(0, dash);
    auto expected_stem = expected_filename.substr(0, expected_so);
    std::transform(loaded_stem.begin(), loaded_stem.end(), loaded_stem.begin(),
        [] (const unsigned char value) { return std::tolower(value); });
    std::transform(expected_stem.begin(), expected_stem.end(), expected_stem.begin(),
        [] (const unsigned char value) { return std::tolower(value); });
    if (loaded_stem != expected_stem)
        return false;
    return expected_hash.rfind(prefix, 0) == 0;
}

inline std::vector<std::string> openmp_runtime_issues(
    const OpenMPRuntimeSnapshot& snapshot, bool openmp_enabled,
    const std::string& expected_path, const std::string& expected_sha256)
{
    std::vector<std::string> issues;
    if (!openmp_enabled)
        return issues;
    const auto expected_hash = normalized_runtime_sha256(expected_sha256);
    if (snapshot.loaded_runtime_path.empty())
        issues.emplace_back("the loaded OpenMP runtime symbol owner is unavailable");
    else if (expected_hash.empty())
        issues.emplace_back("the build-time OpenMP runtime SHA-256 is missing or invalid");
    else {
        const auto loaded_hash = runtime_file_sha256(snapshot.loaded_runtime_path);
        if (loaded_hash.empty())
            issues.emplace_back("the loaded OpenMP runtime file could not be hashed");
        else if (loaded_hash != expected_hash
                && !is_auditwheel_copy_of_expected_runtime(
                    snapshot.loaded_runtime_path, expected_path, expected_hash))
            issues.emplace_back(
                "loaded OpenMP runtime '"+snapshot.loaded_runtime_path
                +"' does not match the build-time runtime SHA-256");
    }
    if (snapshot.loaded_openmp_libraries.size() > 1)
        issues.emplace_back(
            "multiple OpenMP runtimes are loaded in the process");
    return issues;
}

}  // namespace symmetrix::runtime
