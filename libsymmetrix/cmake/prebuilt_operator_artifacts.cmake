include_guard(GLOBAL)

set(_SYMMETRIX_PREBUILT_OPERATOR_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(symmetrix_add_prebuilt_operator_artifacts)
    if(NOT DEFINED SYMMETRIX_PREBUILT_OPERATOR_ARTIFACTS)
        set(SYMMETRIX_PREBUILT_OPERATOR_ARTIFACTS "off")
    endif()
    string(TOLOWER "${SYMMETRIX_PREBUILT_OPERATOR_ARTIFACTS}"
        _symmetrix_prebuilt_mode)
    if(NOT _symmetrix_prebuilt_mode MATCHES "^(off|r0|full)$")
        message(FATAL_ERROR
            "SYMMETRIX_PREBUILT_OPERATOR_ARTIFACTS must be off, r0, or full; "
            "got '${SYMMETRIX_PREBUILT_OPERATOR_ARTIFACTS}'")
    endif()
    if(NOT DEFINED SYMMETRIX_PREBUILT_OAM_ARTIFACTS)
        set(SYMMETRIX_PREBUILT_OAM_ARTIFACTS "off")
    endif()
    string(TOLOWER "${SYMMETRIX_PREBUILT_OAM_ARTIFACTS}"
        _symmetrix_prebuilt_oam_mode)
    if(NOT _symmetrix_prebuilt_oam_mode MATCHES "^(off|known)$")
        message(FATAL_ERROR
            "SYMMETRIX_PREBUILT_OAM_ARTIFACTS must be off or known; got "
            "'${SYMMETRIX_PREBUILT_OAM_ARTIFACTS}'")
    endif()
    if(_symmetrix_prebuilt_mode STREQUAL "off"
            AND _symmetrix_prebuilt_oam_mode STREQUAL "off")
        set(SYMMETRIX_PREBUILT_OPERATOR_CATALOG_HEADER "" CACHE INTERNAL
            "Generated installed-operator provenance header" FORCE)
        message(STATUS "Symmetrix: prebuilt operator artifacts disabled")
        return()
    endif()

    if(NOT DEFINED Python_EXECUTABLE OR NOT EXISTS "${Python_EXECUTABLE}")
        find_package(Python COMPONENTS Interpreter REQUIRED)
    endif()
    if(NOT DEFINED SYMMETRIX_PREBUILT_OPERATOR_TOOL)
        get_filename_component(_symmetrix_repository_root
            "${_SYMMETRIX_PREBUILT_OPERATOR_CMAKE_DIR}/../.." ABSOLUTE)
        set(SYMMETRIX_PREBUILT_OPERATOR_TOOL
            "${_symmetrix_repository_root}/tools/_symmetrix_build/prebuilt_operator_artifacts.py")
    endif()
    if(NOT EXISTS "${SYMMETRIX_PREBUILT_OPERATOR_TOOL}")
        message(FATAL_ERROR
            "Symmetrix prebuilt operator build adapter is unavailable: "
            "${SYMMETRIX_PREBUILT_OPERATOR_TOOL}")
    endif()

    get_filename_component(_symmetrix_tool_package
        "${SYMMETRIX_PREBUILT_OPERATOR_TOOL}" DIRECTORY)
    get_filename_component(_symmetrix_tools_root
        "${_symmetrix_tool_package}" DIRECTORY)
    get_filename_component(_symmetrix_repository_root
        "${_symmetrix_tools_root}" DIRECTORY)
    set(_symmetrix_operator_source
        "${_symmetrix_repository_root}/symmetrix/source/symmetrix")
    if(NOT IS_DIRECTORY "${_symmetrix_operator_source}")
        set(_symmetrix_operator_source
            "${_symmetrix_repository_root}/source/symmetrix")
    endif()
    if(NOT DEFINED SYMMETRIX_PREBUILT_OPERATOR_MANIFEST)
        set(SYMMETRIX_PREBUILT_OPERATOR_MANIFEST
            "${_symmetrix_operator_source}/foundation_operator_manifest.json")
    endif()
    if(NOT EXISTS "${SYMMETRIX_PREBUILT_OPERATOR_MANIFEST}")
        message(FATAL_ERROR
            "Symmetrix production operator manifest is unavailable: "
            "${SYMMETRIX_PREBUILT_OPERATOR_MANIFEST}")
    endif()
    if(NOT DEFINED SYMMETRIX_PREBUILT_OAM_MANIFEST)
        set(SYMMETRIX_PREBUILT_OAM_MANIFEST
            "${_symmetrix_operator_source}/foundation_oam_architectures.json")
    endif()
    if(_symmetrix_prebuilt_oam_mode STREQUAL "known"
            AND NOT EXISTS "${SYMMETRIX_PREBUILT_OAM_MANIFEST}")
        message(FATAL_ERROR
            "Symmetrix OAM architecture manifest is unavailable: "
            "${SYMMETRIX_PREBUILT_OAM_MANIFEST}")
    endif()
    string(LENGTH "${SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256}"
        _symmetrix_source_hash_length)
    if(NOT _symmetrix_source_hash_length EQUAL 64
            OR NOT SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256 MATCHES "^[0-9a-f]+$")
        message(FATAL_ERROR
            "SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256 must be a complete SHA-256 "
            "when prebuilt operator artifacts are enabled")
    endif()
    string(LENGTH "${SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256}"
        _symmetrix_native_source_hash_length)
    if(NOT _symmetrix_native_source_hash_length EQUAL 64
            OR NOT SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256
                MATCHES "^[0-9a-f]+$")
        message(FATAL_ERROR
            "SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256 must be a complete "
            "SHA-256 when prebuilt operator artifacts are enabled")
    endif()

    if(NOT DEFINED SYMMETRIX_PREBUILT_OPERATOR_BACKEND)
        if(Kokkos_ENABLE_CUDA)
            set(SYMMETRIX_PREBUILT_OPERATOR_BACKEND cuda)
        elseif(Kokkos_ENABLE_HIP)
            set(SYMMETRIX_PREBUILT_OPERATOR_BACKEND hip)
        else()
            set(SYMMETRIX_PREBUILT_OPERATOR_BACKEND host)
        endif()
    endif()
    string(TOLOWER "${SYMMETRIX_PREBUILT_OPERATOR_BACKEND}"
        _symmetrix_artifact_backend)
    if(NOT _symmetrix_artifact_backend MATCHES "^(host|cuda|hip)$")
        message(FATAL_ERROR
            "SYMMETRIX_PREBUILT_OPERATOR_BACKEND must be host, cuda, or hip")
    endif()
    if(NOT DEFINED SYMMETRIX_PREBUILT_OPERATOR_TARGET)
        if(_symmetrix_artifact_backend STREQUAL "host")
            set(SYMMETRIX_PREBUILT_OPERATOR_TARGET "${SYMMETRIX_HOST_ARCH}")
        else()
            set(SYMMETRIX_PREBUILT_OPERATOR_TARGET
                "${SYMMETRIX_DEVICE_COMPILER_TARGET}")
        endif()
    endif()
    if(NOT SYMMETRIX_PREBUILT_OPERATOR_TARGET)
        message(FATAL_ERROR
            "The prebuilt operator target must be explicit for "
            "${_symmetrix_artifact_backend}")
    endif()

    if(_symmetrix_artifact_backend STREQUAL "cuda")
        if(NOT CMAKE_CUDA_COMPILER OR NOT EXISTS "${CMAKE_CUDA_COMPILER}")
            message(FATAL_ERROR
                "CUDA compiler is unavailable for prebuilt operator artifacts: "
                "${CMAKE_CUDA_COMPILER}")
        endif()
        if(NOT SYMMETRIX_PREBUILT_OPERATOR_TARGET MATCHES "^sm_([0-9]+)$")
            message(FATAL_ERROR
                "CUDA prebuilt operator target must use the configured sm_NN "
                "compiler spelling; got '${SYMMETRIX_PREBUILT_OPERATOR_TARGET}'")
        endif()
        set(_symmetrix_cuda_digits "${CMAKE_MATCH_1}")
    elseif(_symmetrix_artifact_backend STREQUAL "hip")
        if(NOT CMAKE_CXX_COMPILER OR NOT EXISTS "${CMAKE_CXX_COMPILER}")
            message(FATAL_ERROR
                "HIP compiler is unavailable for prebuilt operator artifacts: "
                "${CMAKE_CXX_COMPILER}")
        endif()
        if(NOT SYMMETRIX_PREBUILT_OPERATOR_TARGET MATCHES "^gfx[0-9a-f]+(:.*)?$")
            message(FATAL_ERROR
                "HIP prebuilt operator target must use a configured gfx target; "
                "got '${SYMMETRIX_PREBUILT_OPERATOR_TARGET}'")
        endif()
    elseif(NOT CMAKE_CXX_COMPILER OR NOT EXISTS "${CMAKE_CXX_COMPILER}")
        message(FATAL_ERROR
            "CXX compiler is unavailable for prebuilt operator artifacts: "
            "${CMAKE_CXX_COMPILER}")
    endif()

    if(_symmetrix_artifact_backend STREQUAL "host")
        if(CMAKE_CONFIGURATION_TYPES)
            message(FATAL_ERROR
                "Host prebuilt operator artifacts require a single-config generator "
                "so one effective compile target can be identity-bound")
        endif()
        set(_symmetrix_effective_host_flags "")
        set(_symmetrix_cxx_flag_strings "${CMAKE_CXX_FLAGS}")
        if(CMAKE_BUILD_TYPE)
            string(TOUPPER "${CMAKE_BUILD_TYPE}" _symmetrix_build_type_upper)
            list(APPEND _symmetrix_cxx_flag_strings
                "${CMAKE_CXX_FLAGS_${_symmetrix_build_type_upper}}")
        endif()
        foreach(_symmetrix_flag_string IN LISTS _symmetrix_cxx_flag_strings)
            if(_symmetrix_flag_string)
                separate_arguments(_symmetrix_split_flags UNIX_COMMAND
                    "${_symmetrix_flag_string}")
                list(APPEND _symmetrix_effective_host_flags
                    ${_symmetrix_split_flags})
            endif()
        endforeach()
        get_directory_property(_symmetrix_directory_compile_options
            COMPILE_OPTIONS)
        foreach(_symmetrix_option IN LISTS _symmetrix_directory_compile_options)
            set(_symmetrix_resolved_option "")
            set(_symmetrix_option_applies TRUE)
            if(_symmetrix_option MATCHES
                    "^\\$<\\$<COMPILE_LANGUAGE:CXX>:(.*)>$")
                set(_symmetrix_resolved_option "${CMAKE_MATCH_1}")
            elseif(_symmetrix_option MATCHES
                    "^\\$<\\$<COMPILE_LANG_AND_ID:CXX,([^>]*)>:(.*)>$")
                string(REPLACE "," ";" _symmetrix_option_compilers
                    "${CMAKE_MATCH_1}")
                if(CMAKE_CXX_COMPILER_ID IN_LIST _symmetrix_option_compilers)
                    set(_symmetrix_resolved_option "${CMAKE_MATCH_2}")
                else()
                    set(_symmetrix_option_applies FALSE)
                endif()
            elseif(NOT _symmetrix_option MATCHES "^\\$<")
                set(_symmetrix_resolved_option "${_symmetrix_option}")
            elseif(_symmetrix_option MATCHES
                    "(-m[a-zA-Z]|target-feature|target-cpu|--target|-arch)")
                message(FATAL_ERROR
                    "Cannot verify ISA-affecting host compile option: "
                    "${_symmetrix_option}")
            else()
                set(_symmetrix_option_applies FALSE)
            endif()
            if(_symmetrix_option_applies
                    AND _symmetrix_resolved_option MATCHES "\\$<")
                if(_symmetrix_resolved_option MATCHES
                        "(-m[a-zA-Z]|target-feature|target-cpu|--target|-arch)")
                    message(FATAL_ERROR
                        "Cannot verify ISA-affecting host compile option: "
                        "${_symmetrix_option}")
                endif()
                set(_symmetrix_option_applies FALSE)
            endif()
            if(_symmetrix_option_applies)
                if(_symmetrix_resolved_option MATCHES "^SHELL:(.*)$")
                    separate_arguments(_symmetrix_shell_flags UNIX_COMMAND
                        "${CMAKE_MATCH_1}")
                    list(APPEND _symmetrix_effective_host_flags
                        ${_symmetrix_shell_flags})
                else()
                    list(APPEND _symmetrix_effective_host_flags
                        "${_symmetrix_resolved_option}")
                endif()
            endif()
        endforeach()
        set(_symmetrix_host_target_arguments
            host-target
            --declared-target "${SYMMETRIX_PREBUILT_OPERATOR_TARGET}"
            --host-arch "${SYMMETRIX_HOST_ARCH}")
        foreach(_symmetrix_flag IN LISTS _symmetrix_effective_host_flags)
            list(APPEND _symmetrix_host_target_arguments
                "--flag=${_symmetrix_flag}")
        endforeach()
        set(_symmetrix_effective_host_target_file
            "${CMAKE_CURRENT_BINARY_DIR}/prebuilt-effective-host-target.txt")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env PYTHONDONTWRITEBYTECODE=1
                "${Python_EXECUTABLE}" "${SYMMETRIX_PREBUILT_OPERATOR_TOOL}"
                ${_symmetrix_host_target_arguments}
                --output "${_symmetrix_effective_host_target_file}"
            RESULT_VARIABLE _symmetrix_host_target_status
            ERROR_VARIABLE _symmetrix_host_target_error)
        if(NOT _symmetrix_host_target_status EQUAL 0)
            message(FATAL_ERROR
                "Could not derive the effective prebuilt host target: "
                "${_symmetrix_host_target_error}")
        endif()
        file(READ "${_symmetrix_effective_host_target_file}"
            SYMMETRIX_PREBUILT_OPERATOR_TARGET)
        string(STRIP "${SYMMETRIX_PREBUILT_OPERATOR_TARGET}"
            SYMMETRIX_PREBUILT_OPERATOR_TARGET)
    endif()
    if(_symmetrix_artifact_backend STREQUAL "cuda")
        set(_symmetrix_artifact_compiler "${CMAKE_CUDA_COMPILER}")
        set(_symmetrix_artifact_compiler_id "${CMAKE_CUDA_COMPILER_ID}")
        set(_symmetrix_artifact_compiler_version
            "${CMAKE_CUDA_COMPILER_VERSION}")
        # Kokkos enables CUDA below this directory, so CMake's compiler path is
        # cached globally while its detected ID/version can remain scoped to
        # the Kokkos subdirectory.  Query nvcc (or nvcc_wrapper) directly when
        # configuring prebuilt artifacts from the parent scope.
        if(NOT _symmetrix_artifact_compiler_id
                OR NOT _symmetrix_artifact_compiler_version)
            execute_process(
                COMMAND "${_symmetrix_artifact_compiler}" --version
                RESULT_VARIABLE _symmetrix_cuda_version_status
                OUTPUT_VARIABLE _symmetrix_cuda_version_stdout
                ERROR_VARIABLE _symmetrix_cuda_version_stderr)
            string(CONCAT _symmetrix_cuda_version_output
                "${_symmetrix_cuda_version_stdout}"
                "${_symmetrix_cuda_version_stderr}")
            if(NOT _symmetrix_cuda_version_status EQUAL 0)
                message(FATAL_ERROR
                    "CUDA compiler identity query failed for prebuilt operator "
                    "artifacts: ${_symmetrix_cuda_version_output}")
            endif()
            if(NOT _symmetrix_cuda_version_output
                    MATCHES "V([0-9]+\\.[0-9]+\\.[0-9]+)")
                message(FATAL_ERROR
                    "CUDA compiler identity is unavailable for prebuilt operator "
                    "artifacts; unable to parse '${_symmetrix_cuda_version_output}'")
            endif()
            set(_symmetrix_artifact_compiler_id NVIDIA)
            set(_symmetrix_artifact_compiler_version "${CMAKE_MATCH_1}")
        endif()
    else()
        set(_symmetrix_artifact_compiler "${CMAKE_CXX_COMPILER}")
        set(_symmetrix_artifact_compiler_id "${CMAKE_CXX_COMPILER_ID}")
        set(_symmetrix_artifact_compiler_version
            "${CMAKE_CXX_COMPILER_VERSION}")
    endif()
    if(NOT _symmetrix_artifact_compiler_id
            OR NOT _symmetrix_artifact_compiler_version)
        message(FATAL_ERROR
            "Compiler identity is unavailable for ${_symmetrix_artifact_backend} "
            "prebuilt operator artifacts")
    endif()
    if(_symmetrix_artifact_backend STREQUAL "cuda" AND SYMMETRIX_FAST_MATH)
        set(_symmetrix_r0_floating_point_policy cuda-fast-math)
    else()
        set(_symmetrix_r0_floating_point_policy strict)
    endif()
    if(_symmetrix_artifact_backend STREQUAL "host" AND SYMMETRIX_FAST_MATH)
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
            message(FATAL_ERROR
                "SYMMETRIX_FAST_MATH host prebuilts require a supported GNU, "
                "Clang, or AppleClang C++ compiler; detected "
                "'${CMAKE_CXX_COMPILER_ID}'")
        endif()
        set(_symmetrix_oam_floating_point_policy fast)
    else()
        set(_symmetrix_oam_floating_point_policy strict)
    endif()

    set(_symmetrix_generation_dependencies
        "${SYMMETRIX_PREBUILT_OPERATOR_TOOL}"
        "${_symmetrix_tool_package}/generated_operator_codegen.py"
        "${_symmetrix_operator_source}/foundation_operator_manifest.py"
        "${SYMMETRIX_PREBUILT_OPERATOR_MANIFEST}"
        "${_symmetrix_operator_source}/foundation_m0_topologies.json"
        "${_symmetrix_operator_source}/generated_operator_codegen.py"
        "${_symmetrix_operator_source}/generated_operator_core.py"
        "${_symmetrix_operator_source}/generated_operator_version.py"
        "${_symmetrix_operator_source}/jit_codegen_common.py"
        "${_symmetrix_operator_source}/jit.py"
        "${_symmetrix_operator_source}/mace_execution_contract.py"
        "${_symmetrix_operator_source}/mace_jit_codegen.py"
        "${_symmetrix_operator_source}/mace_jit_host_plugin_abi.h"
        "${_symmetrix_operator_source}/mace_jit_cuda_plugin_abi.h"
        "${_symmetrix_operator_source}/mace_jit_hip_plugin_abi.h"
        "${_symmetrix_operator_source}/mace_jit_m0_host_plugin_abi.h"
        "${_symmetrix_operator_source}/mace_jit_m1_host_plugin_abi.h"
        "${_symmetrix_operator_source}/mace_jit_r0_host_plugin_abi.h"
        "${_SYMMETRIX_PREBUILT_OPERATOR_CMAKE_DIR}/../source/jit_generation_version.hpp")
    if(_symmetrix_prebuilt_oam_mode STREQUAL "known")
        list(APPEND _symmetrix_generation_dependencies
            "${SYMMETRIX_PREBUILT_OAM_MANIFEST}"
            "${_symmetrix_operator_source}/oam_architecture.py"
            "${_symmetrix_operator_source}/oam_receiver_codegen.py"
            "${_symmetrix_operator_source}/oam_cuda_codegen.py"
            "${_symmetrix_operator_source}/oam_jit_codegen.py"
            "${_symmetrix_operator_source}/oam_jit_host_plugin_abi.h"
            "${_symmetrix_operator_source}/oam_cuda_plugin_abi.h"
            "${_symmetrix_operator_source}/nequip_schema.py")
    endif()
    foreach(_symmetrix_dependency IN LISTS _symmetrix_generation_dependencies)
        if(NOT EXISTS "${_symmetrix_dependency}")
            message(FATAL_ERROR
                "Prebuilt operator generation dependency is unavailable: "
                "${_symmetrix_dependency}")
        endif()
    endforeach()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${_symmetrix_generation_dependencies})

    set(_symmetrix_artifact_directory
        "${CMAKE_CURRENT_BINARY_DIR}/generated-operator-artifacts")
    file(MAKE_DIRECTORY "${_symmetrix_artifact_directory}")
    file(SHA256 "${SYMMETRIX_PREBUILT_OPERATOR_MANIFEST}"
        _symmetrix_manifest_sha256)
    if(_symmetrix_prebuilt_oam_mode STREQUAL "known")
        file(SHA256 "${SYMMETRIX_PREBUILT_OAM_MANIFEST}"
            _symmetrix_oam_manifest_sha256)
    else()
        set(_symmetrix_oam_manifest_sha256 "off")
    endif()
    file(SHA256
        "${_SYMMETRIX_PREBUILT_OPERATOR_CMAKE_DIR}/../source/jit_generation_version.hpp"
        _symmetrix_generation_version_header_sha256)
    set(_symmetrix_build_inputs
        "${_symmetrix_artifact_directory}/build-inputs.txt")
    string(CONCAT _symmetrix_build_inputs_content
        "mode=${_symmetrix_prebuilt_mode}\n"
        "oam_mode=${_symmetrix_prebuilt_oam_mode}\n"
        "backend=${_symmetrix_artifact_backend}\n"
        "target=${SYMMETRIX_PREBUILT_OPERATOR_TARGET}\n"
        "source_content_sha256=${SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256}\n"
        "native_source_content_sha256=${SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256}\n"
        "manifest=${SYMMETRIX_PREBUILT_OPERATOR_MANIFEST}\n"
        "oam_manifest=${SYMMETRIX_PREBUILT_OAM_MANIFEST}\n"
        "manifest_sha256=${_symmetrix_manifest_sha256}\n"
        "oam_manifest_sha256=${_symmetrix_oam_manifest_sha256}\n"
        "generation_version_header_sha256=${_symmetrix_generation_version_header_sha256}\n"
        "python_executable=${Python_EXECUTABLE}\n"
        "python_version=${Python_VERSION}\n"
        "cxx_compiler=${CMAKE_CXX_COMPILER}\n"
        "cxx_compiler_id=${CMAKE_CXX_COMPILER_ID}\n"
        "cxx_compiler_version=${CMAKE_CXX_COMPILER_VERSION}\n"
        "cuda_compiler=${CMAKE_CUDA_COMPILER}\n"
        "cuda_compiler_id=${_symmetrix_artifact_compiler_id}\n"
        "cuda_compiler_version=${_symmetrix_artifact_compiler_version}\n"
        "artifact_compiler=${_symmetrix_artifact_compiler}\n"
        "compiler_id=${_symmetrix_artifact_compiler_id}\n"
        "compiler_version=${_symmetrix_artifact_compiler_version}\n"
        "host_arch=${SYMMETRIX_HOST_ARCH}\n"
        "openmp=${Kokkos_ENABLE_OPENMP}\n")
    string(APPEND _symmetrix_build_inputs_content
        "r0_floating_point_policy=${_symmetrix_r0_floating_point_policy}\n"
        "oam_host_floating_point_policy=${_symmetrix_oam_floating_point_policy}\n")
    file(CONFIGURE
        OUTPUT "${_symmetrix_build_inputs}"
        CONTENT "${_symmetrix_build_inputs_content}"
        @ONLY)
    list(APPEND _symmetrix_generation_dependencies
        "${_symmetrix_build_inputs}"
        "${_symmetrix_artifact_compiler}")
    set(_symmetrix_plan "${_symmetrix_artifact_directory}/plan.json")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env PYTHONDONTWRITEBYTECODE=1
            "${Python_EXECUTABLE}" "${SYMMETRIX_PREBUILT_OPERATOR_TOOL}"
            plan
            --manifest "${SYMMETRIX_PREBUILT_OPERATOR_MANIFEST}"
            --oam-manifest "${SYMMETRIX_PREBUILT_OAM_MANIFEST}"
            --mode "${_symmetrix_prebuilt_mode}"
            --oam-mode "${_symmetrix_prebuilt_oam_mode}"
            --backend "${_symmetrix_artifact_backend}"
            --target "${SYMMETRIX_PREBUILT_OPERATOR_TARGET}"
            --source-content-sha256 "${SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256}"
            --floating-point-policy "${_symmetrix_r0_floating_point_policy}"
            --oam-floating-point-policy
                "${_symmetrix_oam_floating_point_policy}"
            --output "${_symmetrix_plan}"
        RESULT_VARIABLE _symmetrix_plan_status
        ERROR_VARIABLE _symmetrix_plan_error)
    if(NOT _symmetrix_plan_status EQUAL 0)
        message(FATAL_ERROR
            "Could not plan prebuilt operator artifacts: ${_symmetrix_plan_error}")
    endif()
    file(READ "${_symmetrix_plan}" _symmetrix_plan_json)
    string(JSON _symmetrix_artifact_count LENGTH
        "${_symmetrix_plan_json}" artifacts)
    if(_symmetrix_prebuilt_mode STREQUAL "full")
        set(_symmetrix_expected_artifact_count 32)
    elseif(_symmetrix_prebuilt_mode STREQUAL "r0")
        set(_symmetrix_expected_artifact_count 6)
    else()
        set(_symmetrix_expected_artifact_count 0)
    endif()
    if(_symmetrix_prebuilt_oam_mode STREQUAL "known")
        math(EXPR _symmetrix_expected_artifact_count
            "${_symmetrix_expected_artifact_count}+8")
    endif()
    if(NOT _symmetrix_artifact_count EQUAL _symmetrix_expected_artifact_count)
        message(FATAL_ERROR
            "Verified ${_symmetrix_prebuilt_mode}/${_symmetrix_prebuilt_oam_mode} "
            "mode must plan exactly "
            "${_symmetrix_expected_artifact_count} "
            "artifacts for one backend target; planned ${_symmetrix_artifact_count}")
    endif()

    add_custom_target(symmetrix_prebuilt_operator_artifacts ALL)
    set(_symmetrix_catalog
        "${_symmetrix_artifact_directory}/catalog.json")
    set(_symmetrix_catalog_arguments "")
    set(_symmetrix_catalog_dependencies "")
    set(_symmetrix_catalog_binary_targets "")
    set(_symmetrix_r0_install_files "")
    set(_symmetrix_r1_install_files "")
    set(_symmetrix_m0_install_files "")
    set(_symmetrix_m1_install_files "")
    if(SYMMETRIX_PYTHON_INSTALL_PACKAGE)
        set(_symmetrix_install_root
            "${SYMMETRIX_PYTHON_INSTALL_PACKAGE}/_operator_artifacts")
    endif()
    math(EXPR _symmetrix_artifact_last "${_symmetrix_artifact_count}-1")
    foreach(_symmetrix_index RANGE 0 ${_symmetrix_artifact_last})
        foreach(_symmetrix_field IN ITEMS
                artifact_id build_lane operator precision source_extension stem)
            string(JSON _symmetrix_${_symmetrix_field} GET
                "${_symmetrix_plan_json}" artifacts ${_symmetrix_index}
                ${_symmetrix_field})
        endforeach()
        string(TOLOWER "${_symmetrix_operator}" _symmetrix_operator_directory)
        set(_symmetrix_generated_source
            "${_symmetrix_artifact_directory}/${_symmetrix_stem}.${_symmetrix_source_extension}")
        set(_symmetrix_generated_metadata
            "${_symmetrix_artifact_directory}/${_symmetrix_stem}.json")
        set(_symmetrix_build_started
            "${_symmetrix_artifact_directory}/${_symmetrix_stem}.started-ns")
        set(_symmetrix_generated_outputs
            "${_symmetrix_generated_source}"
            "${_symmetrix_generated_metadata}")
        set(_symmetrix_oam_render_arguments "")
        if(_symmetrix_operator STREQUAL "OAM")
            string(JSON _symmetrix_architecture_id GET
                "${_symmetrix_plan_json}" artifacts ${_symmetrix_index}
                architecture_id)
            set(_symmetrix_companion_manifest
                "${_symmetrix_artifact_directory}/${_symmetrix_stem}.manifest.json")
            list(APPEND _symmetrix_generated_outputs
                "${_symmetrix_companion_manifest}")
            list(APPEND _symmetrix_oam_render_arguments
                --oam-manifest "${SYMMETRIX_PREBUILT_OAM_MANIFEST}"
                --architecture-id "${_symmetrix_architecture_id}"
                --companion-manifest "${_symmetrix_companion_manifest}")
            set(_symmetrix_artifact_floating_point_policy
                "${_symmetrix_oam_floating_point_policy}")
        else()
            set(_symmetrix_artifact_floating_point_policy
                "${_symmetrix_r0_floating_point_policy}")
        endif()
        add_custom_command(
            OUTPUT ${_symmetrix_generated_outputs}
            COMMAND "${CMAKE_COMMAND}" -E env PYTHONDONTWRITEBYTECODE=1
                "${Python_EXECUTABLE}"
                "${SYMMETRIX_PREBUILT_OPERATOR_TOOL}" timestamp
                --output "${_symmetrix_build_started}"
            COMMAND "${CMAKE_COMMAND}" -E env PYTHONDONTWRITEBYTECODE=1
                "${Python_EXECUTABLE}"
                "${SYMMETRIX_PREBUILT_OPERATOR_TOOL}" render
                --manifest "${SYMMETRIX_PREBUILT_OPERATOR_MANIFEST}"
                ${_symmetrix_oam_render_arguments}
                --operator "${_symmetrix_operator}"
                --artifact-id "${_symmetrix_artifact_id}"
                --backend "${_symmetrix_artifact_backend}"
                --target "${SYMMETRIX_PREBUILT_OPERATOR_TARGET}"
                --precision "${_symmetrix_precision}"
                --source-content-sha256
                    "${SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256}"
                --floating-point-policy
                    "${_symmetrix_artifact_floating_point_policy}"
                --compiler-id "${_symmetrix_artifact_compiler_id}"
                --compiler-version "${_symmetrix_artifact_compiler_version}"
                --source "${_symmetrix_generated_source}"
                --metadata "${_symmetrix_generated_metadata}"
            DEPENDS ${_symmetrix_generation_dependencies}
            COMMENT "Generating verified ${_symmetrix_stem} operator"
            VERBATIM)

        string(MAKE_C_IDENTIFIER "${_symmetrix_stem}"
            _symmetrix_artifact_token)
        set(_symmetrix_generate_target
            "symmetrix_generate_${_symmetrix_artifact_token}")
        add_custom_target(${_symmetrix_generate_target}
            DEPENDS
                ${_symmetrix_generated_outputs})

        if(_symmetrix_artifact_backend STREQUAL "host")
            set(_symmetrix_binary_target
                "symmetrix_prebuilt_${_symmetrix_artifact_token}")
            add_library(${_symmetrix_binary_target} MODULE
                "${_symmetrix_generated_source}")
            add_dependencies(${_symmetrix_binary_target}
                ${_symmetrix_generate_target})
            set_target_properties(${_symmetrix_binary_target} PROPERTIES
                CXX_STANDARD 20
                CXX_STANDARD_REQUIRED ON
                # These modules only use system runtime libraries.  Inheriting
                # the owning extension's BLAS/toolkit build RPATH makes
                # auditwheel rewrite their bytes after the catalog hashes them.
                SKIP_BUILD_RPATH ON
                LIBRARY_OUTPUT_DIRECTORY "${_symmetrix_artifact_directory}"
                OUTPUT_NAME "${_symmetrix_stem}"
                PREFIX ""
                POSITION_INDEPENDENT_CODE ON)
            if(NOT SYMMETRIX_HOST_ARCH STREQUAL ""
                    AND NOT SYMMETRIX_HOST_ARCH STREQUAL "none")
                target_compile_options(${_symmetrix_binary_target} PRIVATE
                    "-march=${SYMMETRIX_HOST_ARCH}")
            endif()
            if(SYMMETRIX_FAST_MATH)
                target_compile_options(${_symmetrix_binary_target} PRIVATE
                    "$<$<COMPILE_LANGUAGE:CXX>:-ffast-math>")
            endif()
            if(_symmetrix_operator STREQUAL "R1"
                    OR _symmetrix_operator STREQUAL "OAM")
                target_compile_options(${_symmetrix_binary_target} PRIVATE
                    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wno-pragma-once-outside-header>")
            endif()
            if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
                # Generated host modules contain only omp simd directives.
                # Enabling the full OpenMP runtime makes every independently
                # loaded artifact own libgomp and lets auditwheel rewrite its
                # catalog-bound bytes.  The native extension remains the sole
                # owner of worker parallelism and the OpenMP runtime.
                target_compile_options(${_symmetrix_binary_target} PRIVATE
                    -fopenmp-simd)
            elseif(TARGET OpenMP::OpenMP_CXX)
                target_link_libraries(${_symmetrix_binary_target} PRIVATE
                    OpenMP::OpenMP_CXX)
            elseif(Kokkos_ENABLE_OPENMP)
                message(FATAL_ERROR
                    "The owning build enables OpenMP, but OpenMP::OpenMP_CXX "
                    "is unavailable for CPU prebuilt operator artifacts")
            endif()
            add_custom_command(TARGET ${_symmetrix_binary_target} PRE_LINK
                COMMAND "${CMAKE_COMMAND}" -E env PYTHONDONTWRITEBYTECODE=1
                    "${Python_EXECUTABLE}"
                    "${SYMMETRIX_PREBUILT_OPERATOR_TOOL}" timestamp
                    --output "${_symmetrix_build_started}"
                    --preserve-if-newer-than "${_symmetrix_catalog}"
                COMMENT "Timing verified ${_symmetrix_stem} operator link"
                VERBATIM)
            add_dependencies(symmetrix_prebuilt_operator_artifacts
                ${_symmetrix_binary_target})
            if(_symmetrix_operator STREQUAL "OAM")
                set(_symmetrix_installed_binary
                    "oam/${_symmetrix_artifact_id}/plugin${CMAKE_SHARED_MODULE_SUFFIX}")
                set(_symmetrix_installed_manifest
                    "oam/${_symmetrix_artifact_id}/manifest.json")
                list(APPEND _symmetrix_catalog_arguments
                    --artifact
                    "${_symmetrix_generated_metadata}"
                    "$<TARGET_FILE:${_symmetrix_binary_target}>"
                    "${_symmetrix_installed_binary}"
                    "${_symmetrix_build_started}"
                    "${_symmetrix_companion_manifest}"
                    "${_symmetrix_installed_manifest}")
            else()
                list(APPEND _symmetrix_catalog_arguments
                    --artifact
                    "${_symmetrix_generated_metadata}"
                    "$<TARGET_FILE:${_symmetrix_binary_target}>"
                    "${_symmetrix_operator_directory}/${_symmetrix_stem}${CMAKE_SHARED_MODULE_SUFFIX}"
                    "${_symmetrix_build_started}")
            endif()
            list(APPEND _symmetrix_catalog_dependencies
                "${_symmetrix_generated_metadata}"
                ${_symmetrix_binary_target}
                "$<TARGET_FILE:${_symmetrix_binary_target}>")
            if(_symmetrix_operator STREQUAL "OAM")
                list(APPEND _symmetrix_catalog_dependencies
                    "${_symmetrix_companion_manifest}")
            endif()
            list(APPEND _symmetrix_catalog_binary_targets
                ${_symmetrix_binary_target})
            # Install the exact file that the catalog hashes.  install(TARGETS)
            # participates in CMake's --strip step and would silently mutate
            # release-wheel bytes after the checksum was recorded.
            if(_symmetrix_operator STREQUAL "OAM")
                if(SYMMETRIX_PYTHON_INSTALL_PACKAGE)
                    install(FILES "$<TARGET_FILE:${_symmetrix_binary_target}>"
                        DESTINATION
                        "${_symmetrix_install_root}/oam/${_symmetrix_artifact_id}"
                        RENAME "plugin${CMAKE_SHARED_MODULE_SUFFIX}")
                    install(FILES "${_symmetrix_companion_manifest}"
                        DESTINATION
                        "${_symmetrix_install_root}/oam/${_symmetrix_artifact_id}"
                        RENAME manifest.json)
                endif()
            else()
                list(APPEND _symmetrix_${_symmetrix_operator_directory}_install_files
                    "$<TARGET_FILE:${_symmetrix_binary_target}>")
            endif()
            set(_symmetrix_lane_target ${_symmetrix_binary_target})
        elseif(_symmetrix_artifact_backend STREQUAL "cuda")
            set(_symmetrix_cuda_math_options "")
            set(_symmetrix_cuda_module_options "")
            if(_symmetrix_operator STREQUAL "R0" AND SYMMETRIX_FAST_MATH)
                set(_symmetrix_cuda_math_options --use_fast_math)
            endif()
            if(_symmetrix_operator STREQUAL "OAM")
                set(_symmetrix_cuda_module_options
                    -DSYMMETRIX_OAM_CUDA_MODULE=1)
            endif()
            set(_symmetrix_binary
                "${_symmetrix_artifact_directory}/${_symmetrix_stem}.cubin")
            add_custom_command(
                OUTPUT "${_symmetrix_binary}"
                COMMAND "${CMAKE_COMMAND}" -E env PYTHONDONTWRITEBYTECODE=1
                    "${Python_EXECUTABLE}"
                    "${SYMMETRIX_PREBUILT_OPERATOR_TOOL}" timestamp
                    --output "${_symmetrix_build_started}"
                    --preserve-if-newer-than "${_symmetrix_catalog}"
                COMMAND "${CMAKE_CUDA_COMPILER}"
                    -std=c++20 -O3 ${_symmetrix_cuda_math_options}
                    ${_symmetrix_cuda_module_options} --cubin
                    "--generate-code=arch=compute_${_symmetrix_cuda_digits},code=sm_${_symmetrix_cuda_digits}"
                    "${_symmetrix_generated_source}"
                    -o "${_symmetrix_binary}"
                DEPENDS
                    "${_symmetrix_generated_source}"
                    "${_symmetrix_generated_metadata}"
                COMMENT "Compiling verified ${_symmetrix_stem} CUDA cubin"
                VERBATIM)
            add_custom_target(symmetrix_compile_${_symmetrix_artifact_token}
                DEPENDS "${_symmetrix_binary}")
            add_dependencies(symmetrix_compile_${_symmetrix_artifact_token}
                ${_symmetrix_generate_target})
            add_dependencies(symmetrix_prebuilt_operator_artifacts
                symmetrix_compile_${_symmetrix_artifact_token})
            if(_symmetrix_operator STREQUAL "OAM")
                set(_symmetrix_installed_binary
                    "oam/${_symmetrix_artifact_id}/plugin.cubin")
                set(_symmetrix_installed_manifest
                    "oam/${_symmetrix_artifact_id}/manifest.json")
                list(APPEND _symmetrix_catalog_arguments
                    --artifact
                    "${_symmetrix_generated_metadata}"
                    "${_symmetrix_binary}"
                    "${_symmetrix_installed_binary}"
                    "${_symmetrix_build_started}"
                    "${_symmetrix_companion_manifest}"
                    "${_symmetrix_installed_manifest}")
            else()
                list(APPEND _symmetrix_catalog_arguments
                    --artifact
                    "${_symmetrix_generated_metadata}"
                    "${_symmetrix_binary}"
                    "${_symmetrix_operator_directory}/${_symmetrix_stem}.cubin"
                    "${_symmetrix_build_started}")
            endif()
            list(APPEND _symmetrix_catalog_dependencies
                "${_symmetrix_generated_metadata}"
                "${_symmetrix_binary}")
            if(_symmetrix_operator STREQUAL "OAM")
                list(APPEND _symmetrix_catalog_dependencies
                    "${_symmetrix_companion_manifest}")
            endif()
            list(APPEND _symmetrix_catalog_binary_targets
                symmetrix_compile_${_symmetrix_artifact_token})
            if(_symmetrix_operator STREQUAL "OAM")
                if(SYMMETRIX_PYTHON_INSTALL_PACKAGE)
                    install(FILES "${_symmetrix_binary}"
                        DESTINATION
                        "${_symmetrix_install_root}/oam/${_symmetrix_artifact_id}"
                        RENAME plugin.cubin)
                    install(FILES "${_symmetrix_companion_manifest}"
                        DESTINATION
                        "${_symmetrix_install_root}/oam/${_symmetrix_artifact_id}"
                        RENAME manifest.json)
                endif()
            else()
                list(APPEND _symmetrix_${_symmetrix_operator_directory}_install_files
                    "${_symmetrix_binary}")
            endif()
            set(_symmetrix_lane_target
                symmetrix_compile_${_symmetrix_artifact_token})
        else()
            set(_symmetrix_binary
                "${_symmetrix_artifact_directory}/${_symmetrix_stem}.hsaco")
            add_custom_command(
                OUTPUT "${_symmetrix_binary}"
                COMMAND "${CMAKE_COMMAND}" -E env PYTHONDONTWRITEBYTECODE=1
                    "${Python_EXECUTABLE}"
                    "${SYMMETRIX_PREBUILT_OPERATOR_TOOL}" timestamp
                    --output "${_symmetrix_build_started}"
                    --preserve-if-newer-than "${_symmetrix_catalog}"
                COMMAND "${CMAKE_CXX_COMPILER}"
                    -x hip -std=c++20 -O3 --genco
                    --no-gpu-bundle-output
                    -include hip/hip_runtime.h
                    "--offload-arch=${SYMMETRIX_PREBUILT_OPERATOR_TARGET}"
                    "${_symmetrix_generated_source}"
                    -o "${_symmetrix_binary}"
                DEPENDS
                    "${_symmetrix_generated_source}"
                    "${_symmetrix_generated_metadata}"
                COMMENT "Compiling verified ${_symmetrix_stem} HIP code object"
                VERBATIM)
            add_custom_target(symmetrix_compile_${_symmetrix_artifact_token}
                DEPENDS "${_symmetrix_binary}")
            add_dependencies(symmetrix_compile_${_symmetrix_artifact_token}
                ${_symmetrix_generate_target})
            add_dependencies(symmetrix_prebuilt_operator_artifacts
                symmetrix_compile_${_symmetrix_artifact_token})
            list(APPEND _symmetrix_catalog_arguments
                --artifact
                "${_symmetrix_generated_metadata}"
                "${_symmetrix_binary}"
                "${_symmetrix_operator_directory}/${_symmetrix_stem}.hsaco"
                "${_symmetrix_build_started}")
            list(APPEND _symmetrix_catalog_dependencies
                "${_symmetrix_generated_metadata}"
                "${_symmetrix_binary}")
            list(APPEND _symmetrix_catalog_binary_targets
                symmetrix_compile_${_symmetrix_artifact_token})
            list(APPEND _symmetrix_${_symmetrix_operator_directory}_install_files
                "${_symmetrix_binary}")
            set(_symmetrix_lane_target
                symmetrix_compile_${_symmetrix_artifact_token})
        endif()
        set(_symmetrix_lane_tail_variable
            "_symmetrix_lane_${_symmetrix_build_lane}_tail")
        if(DEFINED ${_symmetrix_lane_tail_variable})
            add_dependencies(${_symmetrix_lane_target}
                ${${_symmetrix_lane_tail_variable}})
        endif()
        set(${_symmetrix_lane_tail_variable} ${_symmetrix_lane_target})
    endforeach()

    set(_symmetrix_catalog_header
        "${CMAKE_BINARY_DIR}/generated/prebuilt_operator_build_info.hpp")
    if(_symmetrix_prebuilt_mode STREQUAL "off")
        set(_symmetrix_catalog_mode oam)
    else()
        set(_symmetrix_catalog_mode "${_symmetrix_prebuilt_mode}")
    endif()
    add_custom_command(
        OUTPUT "${_symmetrix_catalog}" "${_symmetrix_catalog_header}"
        COMMAND "${CMAKE_COMMAND}" -E env PYTHONDONTWRITEBYTECODE=1
            "${Python_EXECUTABLE}"
            "${SYMMETRIX_PREBUILT_OPERATOR_TOOL}" catalog
            ${_symmetrix_catalog_arguments}
            --mode "${_symmetrix_catalog_mode}"
            --backend "${_symmetrix_artifact_backend}"
            --target "${SYMMETRIX_PREBUILT_OPERATOR_TARGET}"
            --source-content-sha256
                "${SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256}"
            --native-source-content-sha256
                "${SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256}"
            --compiler-id "${_symmetrix_artifact_compiler_id}"
            --compiler-version "${_symmetrix_artifact_compiler_version}"
            --output "${_symmetrix_catalog}"
            --header "${_symmetrix_catalog_header}"
        DEPENDS
            ${_symmetrix_catalog_dependencies}
            "${SYMMETRIX_PREBUILT_OPERATOR_TOOL}"
        COMMENT "Cataloging installed generated operator artifacts"
        VERBATIM)
    add_custom_target(symmetrix_prebuilt_operator_catalog
        DEPENDS "${_symmetrix_catalog}" "${_symmetrix_catalog_header}")
    add_dependencies(symmetrix_prebuilt_operator_catalog
        ${_symmetrix_catalog_binary_targets})
    add_dependencies(symmetrix_prebuilt_operator_artifacts
        symmetrix_prebuilt_operator_catalog)
    set(SYMMETRIX_PREBUILT_OPERATOR_CATALOG_HEADER
        "${_symmetrix_catalog_header}" PARENT_SCOPE)
    set(SYMMETRIX_PREBUILT_OPERATOR_CATALOG_HEADER
        "${_symmetrix_catalog_header}" CACHE INTERNAL
        "Generated installed-operator provenance header" FORCE)

    if(SYMMETRIX_PYTHON_INSTALL_PACKAGE)
        install(FILES "${_symmetrix_catalog}"
            DESTINATION "${_symmetrix_install_root}")
        foreach(_symmetrix_operator_directory IN ITEMS r1 r0 m0 m1)
            if(_symmetrix_${_symmetrix_operator_directory}_install_files)
                install(FILES
                    ${_symmetrix_${_symmetrix_operator_directory}_install_files}
                    DESTINATION
                    "${_symmetrix_install_root}/${_symmetrix_operator_directory}")
            endif()
        endforeach()
    endif()
    message(STATUS
        "Symmetrix: ${_symmetrix_prebuilt_mode} prebuilt operator mode plans "
        "${_symmetrix_artifact_count} artifacts for "
        "${_symmetrix_artifact_backend}/${SYMMETRIX_PREBUILT_OPERATOR_TARGET}")
endfunction()
