#CMake File for face detector

unset(SUB_PROJECT)
set(SUB_PROJECT "face_detector")

set(SUB_DIR ${CMAKE_SOURCE_DIR}/${SUB_PROJECT})
message(STATUS "SUB_DIR of ${SUB_PROJECT} = ${SUB_DIR}")

include_directories(
        ${SUB_DIR}/include
        ${CMAKE_SOURCE_DIR}/utils
)
file(GLOB SRC
        ${SUB_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/utils/*.cpp
)

add_library(vision_${SUB_PROJECT} SHARED ${SRC})

target_link_libraries(vision_${SUB_PROJECT}
        ${OpenCV_LIBS}
)

add_executable(demo_${SUB_PROJECT} ${SUB_DIR}/demo/${SUB_PROJECT}_demo.cpp)
target_link_libraries(demo_${SUB_PROJECT} vision_${SUB_PROJECT})
