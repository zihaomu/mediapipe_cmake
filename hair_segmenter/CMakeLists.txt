#CMake File for hair_segmenter

set(SUB_DIR ${CMAKE_SOURCE_DIR}/hair_segmenter)
message(STATUS "SEGMENTER_DIR = ${SUB_DIR}")

include_directories(
        ${CMAKE_SOURCE_DIR}/hair_segmenter/include
        ${SUB_DIR}/include
        ${SUB_DIR}/src
        ${CMAKE_SOURCE_DIR}/utils
        )

file(GLOB SRC
        ${SUB_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/hair_segmenter/src/*.cpp
        ${CMAKE_SOURCE_DIR}/utils/*.cpp
        )

add_library(hair_segmenter SHARED ${SRC})

target_link_libraries(hair_segmenter
        ${OpenCV_LIBS}
        )

add_executable(demo_hair_segmenter ${SUB_DIR}/demo/hair_segmenter_demo.cpp)
target_link_libraries(demo_hair_segmenter hair_segmenter)
