cmake_minimum_required(VERSION 3.16)
project(rodv LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)

# 测试配置
enable_testing()
find_package(GTest REQUIRED)

find_path(HTSLIB_INCLUDE_DIR
    NAMES htslib/sam.h      # 指定要查找的具体文件路径
    HINTS /usr/local/include /opt/homebrew/include  # Homebrew 的可能路径
    /usr/include             # 系统路径
    PATH_SUFFIXES htslib     # 额外的搜索子目录（可选）
)


# 手动查找库文件
find_library(HTSLIB_LIBRARY
             NAMES hts
             HINTS /usr/local/lib /opt/homebrew/lib)

if (HTSLIB_INCLUDE_DIR AND HTSLIB_LIBRARY)
    message(STATUS "Found HTSlib: ${HTSLIB_LIBRARY}")
    message(STATUS "Found HTSlib include: ${HTSLIB_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "HTSlib not found!")
endif()

add_subdirectory(iostream)
add_subdirectory(core)

# 主程序可执行文件
add_executable(rodv_main rodv_main.cpp)

target_link_libraries(rodv_main 
    PRIVATE 
    iostream 
    ${HTSLIB_LIBRARY})

target_include_directories(rodv_main PRIVATE ${HTSLIB_INCLUDE_DIR})
