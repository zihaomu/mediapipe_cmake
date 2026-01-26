cmake_minimum_required(VERSION 3.10)
project(MPP_cmake)

set(CMAKE_CXX_STANDARD 14)
option(BUILD_DEMO "IF BUILD LIB WITH DEMO" ON)

if (BUILD_DEMO)
    message(STATUS "BUILD_DEMO is ON")
else ()
    message(STATUS "BUILD_DEMO is OFF")
endif ()

# set OpenCV_lite build DIR
# NOTE: please set the OpenCV_DIR to the opencv_lite build directory
set(OpenCV_DIR "/home/mpp_project/opencv_lite/build") # change to your opencv_lite build directory

# for android platform, we need assign the platform based lib, at leaset the "arm64-v8a" and "armeabi-v7a"
# Please orgnize the folder like the following:
# lib:
#   - arm64-v8a, it should contains: libMNN.so libopencv_core.so libopencv_dnn.so libopencv_imgproc.so
#   - armeabi-v7a, it should contains: libMNN.so libopencv_core.so libopencv_dnn.so libopencv_imgproc.so
#   - include/opencv2
if (ANDROID)
    include_directories("${OpenCV_DIR}/include")
    set(OpenCV_DIR_Android "")
    if (ANDROID_ABI STREQUAL "arm64-v8a")
        set(OpenCV_DIR_Android "${OpenCV_DIR}/arm64-v8a")
    elseif (ANDROID_ABI STREQUAL "armeabi-v7a")
        set(OpenCV_DIR_Android "${OpenCV_DIR}/armeabi-v7a")
    endif ()

    set(OpenCV_LIBS "")
    set(LIB_MNN "${OpenCV_DIR_Android}/libMNN.so")
    set(LIB_opencv_core "${OpenCV_DIR_Android}/libopencv_core.so")
    set(LIB_opencv_dnn "${OpenCV_DIR_Android}/libopencv_dnn.so")
    set(LIB_opencv_imgproc "${OpenCV_DIR_Android}/libopencv_imgproc.so")
    set(LIB_ANDROID_LOG "")
    if(ANDROID)
        find_library(log-lib log)
        message(STATUS "Found android log lib at ${log-lib}!")
    endif()

    if (NOT LIB_opencv_core)
        message(SEND_ERROR "Cound not found opencv_core from ${OpenCV_DIR_Android}")
    endif ()

    list(APPEND OpenCV_LIBS ${LIB_MNN})
    list(APPEND OpenCV_LIBS ${LIB_opencv_core})
    list(APPEND OpenCV_LIBS ${LIB_opencv_dnn})
    list(APPEND OpenCV_LIBS ${LIB_opencv_imgproc})
    list(APPEND OpenCV_LIBS ${log-lib})
else ()
    find_package(OpenCV REQUIRED)
    if (OpenCV_FOUND)
        message(STATUS "Found OpenCV Lib at ${OpenCV_DIR}!")
        include_directories( ${OpenCV_INCLUDE_DIRS})
    else ()
        message(SEND_ERROR "Cound nont found OpenCV lib at ${OpenCV_DIR}")
    endif ()
endif ()


#mpp_vision face detector
include(${CMAKE_SOURCE_DIR}/face_detector/CMakeLists.txt)

#mpp_vision face landmarker
include(${CMAKE_SOURCE_DIR}/face_landmarker/CMakeLists.txt)

#mpp_vision hand detector
include(${CMAKE_SOURCE_DIR}/hand_detector/CMakeLists.txt)

#mpp_vision hand landmark
include(${CMAKE_SOURCE_DIR}/hand_landmarker/CMakeLists.txt)

#mpp_vision pose detector
include(${CMAKE_SOURCE_DIR}/pose_detector/CMakeLists.txt)

#mpp_vision pose landmark
include(${CMAKE_SOURCE_DIR}/pose_landmarker/CMakeLists.txt)

#mpp_vision interactive_segmenter
include(${CMAKE_SOURCE_DIR}/interactive_segmenter/CMakeLists.txt)

#mpp_vision hair_segmenter
include(${CMAKE_SOURCE_DIR}/hair_segmenter/CMakeLists.txt)

#mpp_vision selfie_segmenter
include(${CMAKE_SOURCE_DIR}/selfie_segmenter/CMakeLists.txt)

#mpp_vision iris_tracker
include(${CMAKE_SOURCE_DIR}/iris_tracker/CMakeLists.txt)

#mpp_vision holistic tracker
include(${CMAKE_SOURCE_DIR}/holistic_tracker/CMakeLists.txt)

