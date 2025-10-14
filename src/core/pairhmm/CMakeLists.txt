cmake_minimum_required(VERSION 3.10)

# PairHMM 模块
project(PairHMM CXX)

# 添加子模块
add_subdirectory(intra)

# 可选：如果存在 inter 模块
# add_subdirectory(inter)
