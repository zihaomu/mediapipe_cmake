#!/bin/bash

NDK_PATH="/Android/sdk/ndk/26.1.10909125" # set your NDK path here
CMAKE_TOOLCHAIN_PATH="${NDK_PATH}/build/cmake/android.toolchain.cmake"

cmake ../ \
-DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_PATH \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_NDK=$NDK_PATH  \
-DCMAKE_CXX_FLAGS=-std=c++11 \
-DANDROID_ABI="armeabi-v7a" \
-DDENABLE_CXX11=ON \
-DANDROID_TOOLCHAIN=clang \
-DANDROID_NATIVE_API_LEVEL=android-14 ..

make -j4

