# mediapipe_cmake
Try to reproduce google mediapipe project with **CMake** and [OpenCV_lite](https://github.com/zihaomu/opencv_lite) + [modified MNN](https://github.com/zihaomu/MNN/tree/mediapipe_cmake).
And I will try my best to achieve the same performance as the native mediapipe.

Try to combine the opencv_lite project and mnn project, and run every thing with script.

# How to use this project?
### 1st : clone this project and opencv_lite.
```
git clone https://github.com/zihaomu/opencv_lite.git
git clone https://github.com/zihaomu/mediapipe_cmake.git
```

### 2nd: build opencv_lite from scratch.
I have sliced the MNN source code and integrate it to opencv_lite source code.
So, just build opencv_lite from scratch.
```
cd opencv_lite
mkdir build
cmake ..
make -j4
```

### 3rd: set the right `opencv_lite` build path at `CMakeList.txt`.
Please take a look the 10th line code of `mediapipe_cmake/CMakeLists.txt`.
And build the right target demo, and it should work fine.
