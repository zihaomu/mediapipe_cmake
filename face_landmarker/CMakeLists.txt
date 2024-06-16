#CMake File for face landmarker
unset(SUB_PROJECT)
set(SUB_PROJECT "face_landmarker")

set(SUB_DIR ${CMAKE_SOURCE_DIR}/${SUB_PROJECT})
message(STATUS "Face_Landmarker_DIR = ${SUB_DIR}")

include_directories(
        ${CMAKE_SOURCE_DIR}/face_detector/include
        ${SUB_DIR}/include
        ${SUB_DIR}/src
        ${CMAKE_SOURCE_DIR}/utils
)

file(GLOB SRC
        ${SUB_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/face_detector/src/*.cpp # TODO remove this code!
        ${CMAKE_SOURCE_DIR}/utils/*.cpp
        ${CMAKE_SOURCE_DIR}/face_detector/src/face_geometry/*.cpp
)

add_library(vision_${SUB_PROJECT} SHARED ${SRC})

target_link_libraries(vision_${SUB_PROJECT}
        vision_face_detector
        ${OpenCV_LIBS}
)

add_executable(demo_${SUB_PROJECT} ${SUB_DIR}/demo/${SUB_PROJECT}_demo.cpp)
target_link_libraries(demo_${SUB_PROJECT} vision_${SUB_PROJECT})
