#CMake File for iris tracker

set(SUB_DIR ${CMAKE_SOURCE_DIR}/iris_tracker)
message(STATUS "SEGMENTER_DIR = ${SUB_DIR}")

include_directories(
        ${CMAKE_SOURCE_DIR}/iris_tracker/include
        ${CMAKE_SOURCE_DIR}/face_landmarker/include
        ${SUB_DIR}/include
        ${SUB_DIR}/src
        ${CMAKE_SOURCE_DIR}/utils
        )

file(GLOB SRC
        ${SUB_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/iris_tracker/src/*.cpp
        ${CMAKE_SOURCE_DIR}/utils/*.cpp
        )

add_library(iris_tracker SHARED ${SRC})

target_link_libraries(iris_tracker
        vision_face_landmarker
        ${OpenCV_LIBS}
        )

add_executable(demo_iris_tracker ${SUB_DIR}/demo/iris_tracker_demo.cpp)
target_link_libraries(demo_iris_tracker iris_tracker)
