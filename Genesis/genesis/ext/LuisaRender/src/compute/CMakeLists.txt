cmake_minimum_required(VERSION 3.26...3.29)
cmake_policy(VERSION 3.26)

set(CMAKE_POLICY_DEFAULT_CMP0042 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0068 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(LuisaCompute LANGUAGES C CXX VERSION 0.4.0)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(LUISA_COMPUTE_MASTER_PROJECT ON)
else ()
    set(LUISA_COMPUTE_MASTER_PROJECT OFF)
endif ()

if (NOT SKBUILD AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/options.cmake")
    message(STATUS "Loading bootstrap options from \"scripts/options.cmake\"")
    include(scripts/options.cmake)
else ()
    option(LUISA_COMPUTE_ENABLE_DSL "Enable C++ DSL" ON)
    option(LUISA_COMPUTE_ENABLE_TENSOR "Enable C++ DSL tensor extension" ON)
    option(LUISA_COMPUTE_ENABLE_DX "Enable DirectX backend" ON)
    option(LUISA_COMPUTE_ENABLE_METAL "Enable Metal backend" ON)
    option(LUISA_COMPUTE_ENABLE_CUDA "Enable CUDA backend" ON)
    option(LUISA_COMPUTE_ENABLE_CUDA_EXT_LCUB "Enable CUDA extension: LCUB" OFF)
    option(LUISA_COMPUTE_ENABLE_VULKAN "Enable Vulkan backend" OFF)
    option(LUISA_COMPUTE_ENABLE_CPU "Enable CPU backend" ON)
    option(LUISA_COMPUTE_ENABLE_FALLBACK "Enable Fallback backend" ON)
    option(LUISA_COMPUTE_ENABLE_REMOTE "Enable Remote backend" ON)
    option(LUISA_COMPUTE_ENABLE_GUI "Enable GUI support" ON)
    option(LUISA_COMPUTE_ENABLE_CLANG_CXX "Enable ClangTooling-based C++ shading language" OFF)
    option(LUISA_COMPUTE_BUILD_TESTS "Build tests for LuisaCompute" ${LUISA_COMPUTE_MASTER_PROJECT})
    option(LUISA_COMPUTE_EMBREE_ZIP_PATH "Path to embree zip file. Use this if embree fails to download" "")
endif ()

option(LUISA_COMPUTE_USE_SYSTEM_STL "Use system STL instead of EASTL" OFF)

option(LUISA_COMPUTE_ENABLE_SANITIZERS "Enable sanitizers" OFF)
option(LUISA_COMPUTE_ENABLE_UNITY_BUILD "Enable unity build (which might accelerate the compilation but could break VS Code's clangd indexing)" OFF)
option(LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES "Check dependencies of backends (and disable the unsatisfied ones instead of reporting a configuration error)" ON)

if (UNIX AND NOT APPLE)
    option(LUISA_COMPUTE_ENABLE_WAYLAND "Enable Wayland support in GUI and Vulkan swapchains" OFF)
endif ()

# validate the options
include(scripts/validate_options.cmake)

# Open Image Denoise
option(LUISA_COMPUTE_DOWNLOAD_OIDN "Download OpenImageDenoise for denoiser extension" OFF)

# nvCOMP
if (LUISA_COMPUTE_ENABLE_CUDA)
    option(LUISA_COMPUTE_DOWNLOAD_NVCOMP "Download the nvCOMP library for CUDA GPU decompression" OFF)
endif ()

# basic set-up for compilers
include(scripts/setup_compilation.cmake)
include(scripts/setup_output_dirs.cmake)

# rpath
set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_SKIP_BUILD_RPATH OFF)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

if (APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../bin;@loader_path/../lib")
elseif (UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../bin;$ORIGIN/../lib")
endif ()

if (NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "dist")
endif ()

if (SKBUILD)
    set(CMAKE_INSTALL_INCLUDEDIR ${SKBUILD_NULL_DIR})
    set(CMAKE_INSTALL_DOCDIR ${SKBUILD_NULL_DIR})
    if (WIN32)
        set(CMAKE_INSTALL_LIBDIR ${SKBUILD_NULL_DIR})
    else ()
        set(CMAKE_INSTALL_LIBDIR ${SKBUILD_PLATLIB_DIR}/luisa/dylibs)
    endif ()
    set(CMAKE_INSTALL_BINDIR ${SKBUILD_PLATLIB_DIR}/luisa/dylibs)
else ()
    include(GNUInstallDirs)
endif ()

install(FILES README.md LICENSE
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/)

# include
file(GLOB_RECURSE LUISA_COMPUTE_HEADERS CONFIGURE_DEPENDS
        "include/luisa/*.h"
        "include/luisa/*.hpp")
add_library(luisa-compute-include INTERFACE ${LUISA_COMPUTE_HEADERS})
target_include_directories(luisa-compute-include INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(luisa-compute-include INTERFACE c_std_11 cxx_std_20)

if (LUISA_COMPUTE_ENABLE_RUST)
    target_compile_definitions(luisa-compute-include INTERFACE LUISA_ENABLE_IR=1)
endif ()

if (UNIX AND NOT APPLE AND LUISA_COMPUTE_ENABLE_WAYLAND)
    target_compile_definitions(luisa-compute-include INTERFACE LUISA_ENABLE_WAYLAND=1)
endif ()

# add platform macros
if (WIN32)
    target_compile_definitions(luisa-compute-include INTERFACE LUISA_PLATFORM_WINDOWS=1)
elseif (UNIX)
    target_compile_definitions(luisa-compute-include INTERFACE LUISA_PLATFORM_UNIX=1)
    if (APPLE)
        target_compile_definitions(luisa-compute-include INTERFACE LUISA_PLATFORM_APPLE=1)
    endif ()
endif ()

# sanitizers
if (LUISA_COMPUTE_ENABLE_SANITIZERS)
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(LUISA_COMPUTE_SANITIZER_FLAGS "/fsanitize=address")
        target_compile_options(luisa-compute-include INTERFACE ${LUISA_COMPUTE_SANITIZER_FLAGS})
    else ()
        set(LUISA_COMPUTE_SANITIZER_FLAGS "-fsanitize=address" "-fsanitize=leak" "-fsanitize=undefined")
        target_compile_options(luisa-compute-include INTERFACE ${LUISA_COMPUTE_SANITIZER_FLAGS})
        target_link_options(luisa-compute-include INTERFACE ${LUISA_COMPUTE_SANITIZER_FLAGS})
    endif ()
    mark_as_advanced(LUISA_COMPUTE_SANITIZER_FLAGS)
endif ()

if (LUISA_COMPUTE_ENABLE_DSL)
    target_compile_definitions(luisa-compute-include INTERFACE LUISA_ENABLE_DSL=1)
endif ()

install(TARGETS luisa-compute-include
        EXPORT LuisaComputeTargets)

install(DIRECTORY include/luisa
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Global definition to avoid MSVC ABI break.
# See https://github.com/microsoft/STL/releases/tag/vs-2022-17.10 for details.
if (WIN32)
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    target_compile_definitions(luisa-compute-include INTERFACE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif ()

add_subdirectory(src)
