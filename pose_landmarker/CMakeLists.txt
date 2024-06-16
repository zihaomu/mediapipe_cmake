#CMake File for pose landmarker

set(SUB_DIR ${CMAKE_SOURCE_DIR}/pose_landmarker)

include_directories(
        ${CMAKE_SOURCE_DIR}/pose_detector/include
        ${SUB_DIR}/include
        ${SUB_DIR}/interface
        ${SUB_DIR}/src
        ${CMAKE_SOURCE_DIR}/utils
        )

file(GLOB SRC
        ${SUB_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/pose_detector/src/*.cpp # TODO remove this code!
        ${SUB_DIR}/interface/*.cpp
        ${CMAKE_SOURCE_DIR}/utils/*.cpp
        )

add_library(vision_pose_landmarker SHARED ${SRC} ${SUB_DIR}/interface/pose_landmark.def)

target_link_libraries(vision_pose_landmarker
#        ${vision_pose_detector}
        ${OpenCV_LIBS}
        )

add_executable(demo_pose_landmarker ${SUB_DIR}/demo/pose_landmark_demo.cpp)
target_link_libraries(demo_pose_landmarker vision_pose_landmarker)
