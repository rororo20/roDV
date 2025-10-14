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

# ============================================================================
# CPU 指令集检测
# ============================================================================
message(STATUS "Detecting CPU instruction set support...")

# 创建测试文件检测 AVX2
file(WRITE "${CMAKE_BINARY_DIR}/test_avx2.cpp"
"#include <immintrin.h>
int main() {
    __m256 a = _mm256_set1_ps(1.0f);
    __m256 b = _mm256_set1_ps(2.0f);
    __m256 c = _mm256_add_ps(a, b);
    return 0;
}")

# 创建测试文件检测 AVX512
file(WRITE "${CMAKE_BINARY_DIR}/test_avx512.cpp"
"#include <immintrin.h>
int main() {
    __m512 a = _mm512_set1_ps(1.0f);
    __m512 b = _mm512_set1_ps(2.0f);
    __m512 c = _mm512_add_ps(a, b);
    return 0;
}")

# 检测 AVX2 支持
try_compile(HAS_AVX2
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/test_avx2.cpp
    COMPILE_DEFINITIONS "-mavx2 -mfma"
)

# 检测 AVX512 支持
try_compile(HAS_AVX512
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/test_avx512.cpp
    COMPILE_DEFINITIONS "-mavx512f -mavx512dq -mavx512bw -mavx512vl"
)

message(STATUS "AVX2 support: ${HAS_AVX2}")
message(STATUS "AVX512 support: ${HAS_AVX512}")

# AVX2 编译标志
set(AVX2_FLAGS "-mavx2;-mfma")

# AVX512 编译标志
set(AVX512_FLAGS "-mavx512f;-mavx512dq;-mavx512bw;-mavx512vl")

# ============================================================================
# 根据 CPU 支持编译对应的库
# ============================================================================
set(PAIRHMM_AVX2_SOURCES
    pairhmm_avx2.cpp
    pairhmm_impl.cpp
)

set(PAIRHMM_AVX512_SOURCES
    pairhmm_avx512.cpp
    pairhmm_impl.cpp
)

# 始终编译 AVX2（现代 CPU 都支持）
if(HAS_AVX2)
    message(STATUS "Building AVX2 version")
    add_library(pairhmm_intra_avx2 STATIC ${PAIRHMM_AVX2_SOURCES})
    target_compile_options(pairhmm_intra_avx2 PRIVATE ${AVX2_FLAGS})
    target_compile_definitions(pairhmm_intra_avx2 PRIVATE -D__AVX2__)
    
    install(TARGETS pairhmm_intra_avx2
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib)
endif()

# 仅在支持时编译 AVX512
if(HAS_AVX512)
    message(STATUS "Building AVX512 version")
    add_library(pairhmm_intra_avx512 STATIC ${PAIRHMM_AVX512_SOURCES})
    target_compile_options(pairhmm_intra_avx512 PRIVATE ${AVX512_FLAGS})
    target_compile_definitions(pairhmm_intra_avx512 PRIVATE -D__AVX512F__)
    
    install(TARGETS pairhmm_intra_avx512
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib)
else()
    message(STATUS "AVX512 not supported, skipping AVX512 version")
endif()

# 安装头文件
install(FILES 
    pairhmm_api.h
    common.h
    context.h
    simd_traits.h
    DESTINATION include/pairhmm/intra)
