cmake_minimum_required(VERSION 3.10)

# Core 模块
project(RoDVCore CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加子模块（先构建 pairhmm 提供的库）
add_subdirectory(pairhmm)

# 构建调度库
add_library(pairhmm_schedule STATIC
    pairhmm_schedule.cpp
)
target_compile_options(pairhmm_schedule PRIVATE -mavx2 -mfma)

target_include_directories(pairhmm_schedule PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/pairhmm
    ${CMAKE_CURRENT_SOURCE_DIR}/pairhmm/common
    ${CMAKE_CURRENT_SOURCE_DIR}/pairhmm/intra
    ${CMAKE_CURRENT_SOURCE_DIR}/pairhmm/inter
)

target_compile_options(pairhmm_schedule PRIVATE -Wall -Wextra)

# 链接 inter/intra 库（如存在）
target_link_libraries(pairhmm_schedule PUBLIC
    pairhmm_intra_avx2
    pairhmm_inter_avx2
    pairhmm_intra_avx512
    pairhmm_inter_avx512
)

# 继续添加测试子模块
add_subdirectory(test)

