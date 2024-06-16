## Hair Segmenter
Since the `hair_segmentation.tflite` has three built-in op,(`MaxPoolingWithArgmax2D`, `MaxUnpooling2D`, `Convolution2DTransposeBias`).
This demo can only be run at opencv_lite with tflite backend.

[How to build opencv_lite with tflite backend?](Link here.)

TODO: Try to add the built-in implementation to MNN source code.