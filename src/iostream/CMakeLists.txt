

file(GLOB ALL_SRC "*.cpp")

add_library(iostream ${ALL_SRC})

target_link_libraries(iostream PRIVATE ${HTSLIB_LIBRARY})

target_include_directories(iostream PRIVATE ${HTSLIB_INCLUDE_DIR})

install(TARGETS iostream DESTINATION lib)