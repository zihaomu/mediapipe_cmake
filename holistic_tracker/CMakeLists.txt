#CMake File for holistic tracker

set(SUB_DIR ${CMAKE_SOURCE_DIR}/holistic_tracker)
message(STATUS "SEGMENTER_DIR = ${SUB_DIR}")

include_directories(
        ${CMAKE_SOURCE_DIR}/pose_landmarker/include
        ${CMAKE_SOURCE_DIR}/face_landmarker/src
        ${CMAKE_SOURCE_DIR}/hand_landmarker/src
        ${SUB_DIR}/include
        ${SUB_DIR}/src
        ${CMAKE_SOURCE_DIR}/utils
        )

file(GLOB SRC
        ${SUB_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/holistic_tracker/src/*.cpp
        ${CMAKE_SOURCE_DIR}/face_landmarker/src/*.cpp
        ${CMAKE_SOURCE_DIR}/hand_landmarker/src/*.cpp
        ${CMAKE_SOURCE_DIR}/utils/*.cpp
        )

add_library(holistic_tracker SHARED ${SRC})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#        message(STATUS "Build type is Release")
        add_definitions(-DHOLISTIC_DEBUG)
endif()

target_link_libraries(holistic_tracker
        vision_pose_landmarker
        vision_face_landmarker
        vision_hand_landmarker
        ${OpenCV_LIBS}
        )

add_executable(demo_holistic_tracker ${SUB_DIR}/demo/holistic_tracker_demo.cpp)
target_link_libraries(demo_holistic_tracker holistic_tracker)
