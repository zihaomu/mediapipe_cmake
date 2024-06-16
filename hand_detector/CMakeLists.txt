#CMake File for hand detector

set(SUB_DIR ${CMAKE_SOURCE_DIR}/hand_detector)
message(STATUS "SUB_DIR = ${SUB_DIR}")

include_directories(
        ${SUB_DIR}/include
        ${CMAKE_SOURCE_DIR}/utils
        )
file(GLOB SRC
        ${SUB_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/utils/*.cpp
        )

add_library(vision_hand_detector SHARED ${SRC})

target_link_libraries(vision_hand_detector
        ${OpenCV_LIBS}
        )

add_executable(demo_hand_detector ${SUB_DIR}/demo/hand_detector_demo.cpp)
target_link_libraries(demo_hand_detector vision_hand_detector)
