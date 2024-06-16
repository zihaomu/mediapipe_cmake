## Hair Segmenter
Since the `selfie_segmetation.tflite` has one built-in op which is not supported by opencv_lite MNN backend(`Convolution2DTransposeBias`).
This demo can only be run at opencv_lite with tflite backend.

[How to build opencv_lite with tflite backend?](Link here.)

TODO: Try to add the built-in implementation to MNN source code.