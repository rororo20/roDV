cmake_minimum_required(VERSION 3.10)

find_package(Threads REQUIRED)

add_library(make_examples STATIC
    snp_counter.cpp
    snp_region.cpp
    realigner.cpp
    apply_alignment.cpp
    cigar_norm.cpp
    pileup.cpp
    runtime/reference_cache.cpp
    runtime/read_dispatch.cpp
    runtime/hts_read_source.cpp
    runtime/region_worker.cpp
    runtime/example_writer.cpp
    runtime/pipeline.cpp)
target_include_directories(make_examples PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HTSLIB_INCLUDE_DIR})
target_link_libraries(make_examples PUBLIC Threads::Threads ${HTSLIB_LIBRARY})

add_subdirectory(test)
