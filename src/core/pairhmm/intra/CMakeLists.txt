cmake_minimum_required(VERSION 3.10)

# PairHMM Intra 模块
project(PairHMMIntra CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# 头文件路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# AVX2 编译标志
set(AVX2_FLAGS "-mavx2;-mfma")

# AVX512 编译标志
set(AVX512_FLAGS "-mavx512f;-mavx512dq;-mavx512bw;-mavx512vl")

# ============================================================================
# 编译所有版本的库
# ============================================================================
set(PAIRHMM_AVX2_SOURCES
    pairhmm_impl.cpp
    pairhmm_api.cpp
)

set(PAIRHMM_AVX512_SOURCES
    pairhmm_impl.cpp
    pairhmm_api.cpp
)

# 编译 AVX2 版本
message(STATUS "Building AVX2 version")
add_library(pairhmm_intra_avx2 STATIC ${PAIRHMM_AVX2_SOURCES})
target_compile_options(pairhmm_intra_avx2 PRIVATE ${AVX2_FLAGS})
target_compile_definitions(pairhmm_intra_avx2 PRIVATE -D__AVX2__)

install(TARGETS pairhmm_intra_avx2
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

# 编译 AVX512 版本
message(STATUS "Building AVX512 version")
add_library(pairhmm_intra_avx512 STATIC ${PAIRHMM_AVX512_SOURCES})
target_compile_options(pairhmm_intra_avx512 PRIVATE ${AVX512_FLAGS})
target_compile_definitions(pairhmm_intra_avx512 PRIVATE -D__AVX512F__)

install(TARGETS pairhmm_intra_avx512
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

# 安装头文件
install(FILES 
    pairhmm_api.h
    common.h
    context.h
    simd_traits.h
    DESTINATION include/pairhmm/intra)
