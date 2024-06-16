#CMake File for selfie segmenter

set(SUB_DIR ${CMAKE_SOURCE_DIR}/selfie_segmenter)
message(STATUS "SEGMENTER_DIR = ${SUB_DIR}")

include_directories(
        ${CMAKE_SOURCE_DIR}/selfie_segmenter/include
        ${SUB_DIR}/include
        ${SUB_DIR}/src
        ${CMAKE_SOURCE_DIR}/utils
        )

file(GLOB SRC
        ${SUB_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/selfie_segmenter/src/*.cpp
        ${CMAKE_SOURCE_DIR}/utils/*.cpp
        )

add_library(selfie_segmenter SHARED ${SRC})

target_link_libraries(selfie_segmenter
        ${OpenCV_LIBS}
        )

add_executable(demo_selfie_segmenter ${SUB_DIR}/demo/selfie_segmenter_demo.cpp)
target_link_libraries(demo_selfie_segmenter selfie_segmenter)
