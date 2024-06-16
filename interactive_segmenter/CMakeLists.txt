#CMake File for hand detector

set(SUB_DIR ${CMAKE_SOURCE_DIR}/interactive_segmenter)
message(STATUS "INTERACTIVE_SEGMENTER_DIR = ${SUB_DIR}")

include_directories(
        ${CMAKE_SOURCE_DIR}/interactive_segmenter/include
        ${SUB_DIR}/include
        ${SUB_DIR}/src
        ${CMAKE_SOURCE_DIR}/utils
        )

file(GLOB SRC
        ${SUB_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/interactive_segmenter/src/*.cpp
        ${CMAKE_SOURCE_DIR}/utils/*.cpp
        )

add_library(interactive_segmenter SHARED ${SRC})

target_link_libraries(interactive_segmenter
        ${OpenCV_LIBS}
        )

add_executable(demo_interative_segmenter ${SUB_DIR}/demo/interactive_segmenter_demo.cpp)
target_link_libraries(demo_interative_segmenter interactive_segmenter)
