#CMake File for hand detector

set(SUB_DIR ${CMAKE_SOURCE_DIR}/hand_landmarker)
message(STATUS "HAND_DETECTOR_DIR = ${HAND_DETECTOR_DIR}")

include_directories(
        ${CMAKE_SOURCE_DIR}/hand_detector/include
        ${SUB_DIR}/include
        ${SUB_DIR}/src
        ${CMAKE_SOURCE_DIR}/utils
        )

file(GLOB SRC
        ${SUB_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/hand_detector/src/*.cpp # TODO remove this code!
        ${CMAKE_SOURCE_DIR}/utils/*.cpp
        )

add_library(vision_hand_landmarker SHARED ${SRC})

target_link_libraries(vision_hand_landmarker
        ${vision_hand_detector}
        ${OpenCV_LIBS}
        )

add_executable(demo_handlandmarker ${SUB_DIR}/demo/hand_landmark_demo.cpp)
target_link_libraries(demo_handlandmarker vision_hand_landmarker)
