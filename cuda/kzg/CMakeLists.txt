cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_C_STANDARD_REQUIRED TRUE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# find_program(_nvidia_smi "nvidia-smi")

# if(_nvidia_smi)
#   set(DETECT_GPU_COUNT_NVIDIA_SMI 0)

#   # execute nvidia-smi -L to get a short list of GPUs available
#   exec_program(${_nvidia_smi_path} ARGS -L
#     OUTPUT_VARIABLE _nvidia_smi_out
#     RETURN_VALUE _nvidia_smi_ret)

#   # process the stdout of nvidia-smi
#   if(_nvidia_smi_ret EQUAL 0)
#     # convert string with newlines to list of strings
#     string(REGEX REPLACE "\n" ";" _nvidia_smi_out "${_nvidia_smi_out}")

#     foreach(_line ${_nvidia_smi_out})
#       if(_line MATCHES "^GPU [0-9]+:")
#         math(EXPR DETECT_GPU_COUNT_NVIDIA_SMI "${DETECT_GPU_COUNT_NVIDIA_SMI}+1")

#         # the UUID is not very useful for the user, remove it
#         string(REGEX REPLACE " \\(UUID:.*\\)" "" _gpu_info "${_line}")

#         if(NOT _gpu_info STREQUAL "")
#           list(APPEND DETECT_GPU_INFO "${_gpu_info}")
#         endif()
#       endif()
#     endforeach()

#     check_num_gpu_info(${DETECT_GPU_COUNT_NVIDIA_SMI} DETECT_GPU_INFO)
#     set(DETECT_GPU_COUNT ${DETECT_GPU_COUNT_NVIDIA_SMI})
#   endif()
# endif()

# ##
# if(DETECT_GPU_COUNT GREATER 0)
#   set(CMAKE_CUDA_ARCHITECTURES native) # do native
# else()
#   # no GPUs found, like on Github CI runners
#   set(CMAKE_CUDA_ARCHITECTURES 50) # some safe value
# endif()

include(CheckLanguage)
check_language(CUDA)

# set(CMAKE_CXX_COMPILER "clang-cl")
# set(CMAKE_C_COMPILER "clang-cl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80 -gencode arch=compute_70,code=sm_70 -t0")

project(rust_kzg_cuda LANGUAGES CUDA CXX C ASM_MASM)

set(CMAKE_AR lib.exe)
set(CMAKE_ASM_MASM_CREATE_STATIC_LIBRARY "<CMAKE_AR> /OUT:<TARGET> <LINK_FLAGS> <OBJECTS>")

include_directories(../external/sppark)
include_directories(../external/blst/src)
include_directories(../external/blst/bindings)

file(GLOB BLST_ASM_FILES ${PROJECT_SOURCE_DIR}/../external/blst/build/win64/*-x86_64.asm)
message(${BLST_ASM_FILES})
add_library(blst STATIC ${BLST_ASM_FILES} ../external/blst/src/server.c)
set_target_properties(blst PROPERTIES LINKER_LANGUAGE C)
target_compile_definitions(blst PUBLIC __BLST_PORTABLE__)

add_library(sppark ../external/sppark/rust/src/lib.cpp ../external/sppark/util/all_gpus.cpp)
target_compile_definitions(sppark PUBLIC __BLST_PORTABLE__ TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE)
set_target_properties(sppark
        PROPERTIES
                CUDA_RUNTIME_LIBRARY Shared
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(rust_kzg_cuda_example main.cu)
set_target_properties(rust_kzg_cuda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(rust_kzg_cuda_example PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# add_executable(test_exec test.c)
target_link_libraries(rust_kzg_cuda_example PUBLIC sppark blst)
