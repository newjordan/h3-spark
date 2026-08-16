cmake_minimum_required(VERSION 3.24)
project(h3_swiglu_nvfp4_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(H3_TORCH_CUDA_ARCH_LIST "12.1a" CACHE STRING
    "Torch-style CUDA target list for the H3 native kernels")
# TorchConfig runs its own architecture setup during find_package(Torch) and
# otherwise probes the live GPU (GB10 reports 12.1). Pin its spelling to the
# family-compatible target used by the extension so configuration is stable
# even when performed on a headless build machine.
set(TORCH_CUDA_ARCH_LIST "${H3_TORCH_CUDA_ARCH_LIST}" CACHE STRING
    "CUDA architecture list consumed by TorchConfig" FORCE)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)

# Torch's CMake package intentionally exposes the C++ libraries but does not
# consistently define a torch_python target. Resolve it from the selected
# interpreter's torch installation instead of baking in a workspace path.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / 'lib')"
    RESULT_VARIABLE H3_TORCH_QUERY_STATUS
    OUTPUT_VARIABLE H3_TORCH_LIBRARY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT H3_TORCH_QUERY_STATUS EQUAL 0 OR NOT IS_DIRECTORY "${H3_TORCH_LIBRARY_DIR}")
    message(FATAL_ERROR "Could not resolve the selected Python interpreter's torch/lib directory")
endif()
find_library(H3_TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS "${H3_TORCH_LIBRARY_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
)

add_library(h3_swiglu_nvfp4_native_v1 MODULE swiglu_nvfp4_cuda.cu)
target_include_directories(h3_swiglu_nvfp4_native_v1 PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(h3_swiglu_nvfp4_native_v1 PRIVATE
    ${TORCH_LIBRARIES}
    "${H3_TORCH_PYTHON_LIBRARY}"
    Python::Module
    cuda
)
target_compile_definitions(h3_swiglu_nvfp4_native_v1 PRIVATE
    TORCH_EXTENSION_NAME=h3_swiglu_nvfp4_native_v1
)
target_compile_options(h3_swiglu_nvfp4_native_v1 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-U__CUDA_NO_BFLOAT16_CONVERSIONS__>
)
set_target_properties(h3_swiglu_nvfp4_native_v1 PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    # TorchConfig already adds the pinned architecture flag. Suppress CMake's
    # independently detected default to avoid duplicate cubin generation.
    CUDA_ARCHITECTURES OFF
    BUILD_RPATH "${H3_TORCH_LIBRARY_DIR}"
)

add_library(h3_swiglu_sat_cert MODULE swiglu_sat_cert.cu)
target_include_directories(h3_swiglu_sat_cert PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(h3_swiglu_sat_cert PRIVATE
    ${TORCH_LIBRARIES}
    "${H3_TORCH_PYTHON_LIBRARY}"
    Python::Module
    cuda
)
target_compile_definitions(h3_swiglu_sat_cert PRIVATE
    TORCH_EXTENSION_NAME=h3_swiglu_sat_cert
)
target_compile_options(h3_swiglu_sat_cert PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-U__CUDA_NO_BFLOAT16_CONVERSIONS__>
)
set_target_properties(h3_swiglu_sat_cert PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    CUDA_ARCHITECTURES OFF
    BUILD_RPATH "${H3_TORCH_LIBRARY_DIR}"
)
