#!/bin/bash

NDK_PATH="/Users/moo/Library/Android/sdk/ndk/26.1.10909125"

# NDK_PATH="/Users/mzh/Library/Android/sdk/ndk/25.2.9519653"
CMAKE_TOOLCHAIN_PATH="${NDK_PATH}/build/cmake/android.toolchain.cmake"

cmake ../ \
-DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_PATH \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_NDK=$NDK_PATH  \
-DCMAKE_CXX_FLAGS=-std=c++11 \
-DANDROID_ABI="arm64-v8a" \
-DDENABLE_CXX11=ON \
-DBUILD_DEMO=OFF \
-DANDROID_TOOLCHAIN=clang \
-DANDROID_NATIVE_API_LEVEL=android-21 ..

make -j4
