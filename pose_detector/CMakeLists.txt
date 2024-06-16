#CMake File for hand detector

set(SUB_DIR ${CMAKE_SOURCE_DIR}/pose_detector)
message(STATUS "SUB_DIR = ${SUB_DIR}")

include_directories(
        ${SUB_DIR}/include
        ${CMAKE_SOURCE_DIR}/utils
)
file(GLOB SRC
        ${SUB_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/utils/*.cpp
        )

add_library(vision_pose_detector SHARED ${SRC})

target_link_libraries(vision_pose_detector
        ${OpenCV_LIBS}
        )

add_executable(demo_pose_detector ${SUB_DIR}/demo/pose_detector_demo.cpp)
target_link_libraries(demo_pose_detector vision_pose_detector)
