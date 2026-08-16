cmake_minimum_required(VERSION 3.24)
project(h3_gate_rms_adaln_nvfp4_static_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(H3_TORCH_CUDA_ARCH_LIST "12.0" CACHE STRING "Torch-style CUDA target")
set(TORCH_CUDA_ARCH_LIST "${H3_TORCH_CUDA_ARCH_LIST}" CACHE STRING
    "CUDA architecture list consumed by TorchConfig" FORCE)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / 'lib')"
    RESULT_VARIABLE H3_TORCH_QUERY_STATUS
    OUTPUT_VARIABLE H3_TORCH_LIBRARY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT H3_TORCH_QUERY_STATUS EQUAL 0 OR NOT IS_DIRECTORY "${H3_TORCH_LIBRARY_DIR}")
    message(FATAL_ERROR "Could not resolve torch/lib for the selected Python")
endif()
find_library(H3_TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS "${H3_TORCH_LIBRARY_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
)

add_library(h3_gate_rms_adaln_nvfp4_static_native_v1 MODULE
    gate_rms_adaln_nvfp4_static_cuda.cu)
target_include_directories(h3_gate_rms_adaln_nvfp4_static_native_v1 PRIVATE
    ${TORCH_INCLUDE_DIRS})
target_link_libraries(h3_gate_rms_adaln_nvfp4_static_native_v1 PRIVATE
    ${TORCH_LIBRARIES}
    "${H3_TORCH_PYTHON_LIBRARY}"
    Python::Module
)
target_compile_definitions(h3_gate_rms_adaln_nvfp4_static_native_v1 PRIVATE
    TORCH_EXTENSION_NAME=h3_gate_rms_adaln_nvfp4_static_native_v1)
target_compile_options(h3_gate_rms_adaln_nvfp4_static_native_v1 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-U__CUDA_NO_BFLOAT16_CONVERSIONS__;--ptxas-options=-v>
)
set_target_properties(h3_gate_rms_adaln_nvfp4_static_native_v1 PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    CUDA_ARCHITECTURES OFF
    BUILD_RPATH "${H3_TORCH_LIBRARY_DIR}"
)
