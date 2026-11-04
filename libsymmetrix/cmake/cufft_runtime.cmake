# Stage the pinned provider without modifying its source submodule.
function(symmetrix_prepare_cufft_runtime)
    # Git applies the staging patch; declare it here instead of relying on
    # another subproject's discovery order.
    find_package(Git QUIET)
    if(NOT GIT_FOUND OR NOT GIT_EXECUTABLE)
        message(FATAL_ERROR
            "Git is required to stage the cuFFT runtime provider but was not found.")
    endif()
    set(base "${SYMMETRIX_KOKKOSFFT_SOURCE_DIR}")
    set(staged "${CMAKE_CURRENT_BINARY_DIR}/external/kokkosfft-runtime-source")
    set(patch "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/kokkosfft-2.0.0-cufft-runtime.patch")
    file(SHA256 "${patch}" fingerprint)
    file(GLOB_RECURSE inputs CONFIGURE_DEPENDS RELATIVE "${base}"
        "${base}/*.cmake" "${base}/*CMakeLists.txt" "${base}/*.hpp" "${base}/*.h")
    list(SORT inputs)
    foreach(relative IN LISTS inputs)
        file(SHA256 "${base}/${relative}" checksum)
        string(APPEND fingerprint "${relative}:${checksum}\n")
    endforeach()
    set(stamp "${staged}/.symmetrix-patch-fingerprint")
    set(current "")
    if(EXISTS "${stamp}")
        file(READ "${stamp}" current)
    endif()
    if(NOT current STREQUAL fingerprint)
        file(REMOVE_RECURSE "${staged}")
        file(MAKE_DIRECTORY "${staged}")
        file(COPY "${base}/" DESTINATION "${staged}" PATTERN ".git" EXCLUDE)
        execute_process(COMMAND "${CMAKE_COMMAND}" -E env
            "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}/external"
            "${GIT_EXECUTABLE}" apply --check "${patch}"
            WORKING_DIRECTORY "${staged}" COMMAND_ERROR_IS_FATAL ANY)
        execute_process(COMMAND "${CMAKE_COMMAND}" -E env
            "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}/external"
            "${GIT_EXECUTABLE}" apply "${patch}"
            WORKING_DIRECTORY "${staged}" COMMAND_ERROR_IS_FATAL ANY)
        file(WRITE "${stamp}" "${fingerprint}")
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env
        "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}/external"
        "${GIT_EXECUTABLE}" apply --reverse --check "${patch}"
        WORKING_DIRECTORY "${staged}" COMMAND_ERROR_IS_FATAL ANY)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${patch}")
    foreach(relative IN LISTS inputs)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${base}/${relative}")
    endforeach()
    set(SYMMETRIX_KOKKOSFFT_SOURCE_DIR "${staged}" PARENT_SCOPE)
endfunction()
