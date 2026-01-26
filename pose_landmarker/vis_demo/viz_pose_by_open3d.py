#!/usr/bin/env python3
"""
Pose Landmark C API Demo with OpenCV (2D) and Open3D (3D) visualization.
Shows 33-pose keypoints from the C API using a sample video or webcam.
"""

from time import sleep
import open3d as o3d
import numpy as np
import cv2
import ctypes
from ctypes import c_int, c_float, c_char_p, POINTER, Structure


# Constants matching C API
MPP_POSE_LANDMARK_MAX_POINTS = 33
MPP_IMAGE_TYPE_BGR = 1
MPP_NO_FLIP = 0
MPP_ROTATION_0 = -1

ROOT_PATH = '/home/moo/work/my_lab/mpp_project/mediapiep_cmake_private/'

# Pose connections (from pose_landmark.h)
POSE_CONNECTIONS = [
    (0, 4), (4, 5), (5, 6), (6, 8), (0, 1), (1, 2), (2, 3), (3, 7),
    (10, 9), (12, 11), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (18, 20), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 24), (11, 23), (24, 23), (24, 26), (23, 25), (26, 28), (25, 27),
    (28, 30), (27, 29), (30, 32), (29, 31), (28, 32), (27, 31)
]


class PoseLandmarkResult(Structure):
    _fields_ = [
        ("poseNum", c_int),
        ("rect", c_int * 4),
        ("score", c_float),
        ("points", c_float * (MPP_POSE_LANDMARK_MAX_POINTS * 3)),  # x, y, z
    ]


# Load shared library
lib = ctypes.CDLL(ROOT_PATH + 'build/libvision_pose_landmarker.so')

# C API signatures
lib.initPoseLandmarker.argtypes = []
lib.initPoseLandmarker.restype = c_int

lib.loadModelPoseDetectFromFile.argtypes = [c_char_p, c_int]
lib.loadModelPoseDetectFromFile.restype = c_int

lib.loadModelPoseLandmarkFromFile.argtypes = [c_char_p, c_int]
lib.loadModelPoseLandmarkFromFile.restype = c_int

lib.runPoseLandmark.argtypes = [c_char_p, c_int, c_int, c_int, c_int, c_int, c_int, POINTER(PoseLandmarkResult)]
lib.runPoseLandmark.restype = c_int

lib.releasePoseLandmark.argtypes = []
lib.releasePoseLandmark.restype = c_int


def draw_pose_opencv(img, result: PoseLandmarkResult):
    """Draw 2D keypoints and skeleton on the OpenCV frame."""
    if result.poseNum != 1:
        return img

    h, w = img.shape[:2]

    # Bounding box
    x, y, bw, bh = result.rect
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

    # Draw skeleton
    for a, b in POSE_CONNECTIONS:
        xa, ya = result.points[a * 3] * w, result.points[a * 3 + 1] * h
        xb, yb = result.points[b * 3] * w, result.points[b * 3 + 1] * h
        cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 255), 2)

    # Draw keypoints
    for k in range(MPP_POSE_LANDMARK_MAX_POINTS):
        px = int(result.points[k * 3] * w)
        py = int(result.points[k * 3 + 1] * h)
        cv2.circle(img, (px, py), 3, (255, 0, 255), -1)

    return img


def create_open3d_visualizer():
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Pose 3D Landmarks - Drag to Rotate", width=960, height=720)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.08, 0.08])
    opt.point_size = 12.0
    opt.line_width = 4.0

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)
    vis.add_geometry(coord_frame)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    line_set = o3d.geometry.LineSet()
    vis.add_geometry(line_set)

    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 1])   # Look towards +Z
    ctr.set_up([0, -1, 0])     # Y-axis points down (image coordinates)
    ctr.set_zoom(0.35)

    return vis, pcd, line_set


VIEW_INIT_DONE = False


def update_open3d_visualization(vis, pcd, line_set, result: PoseLandmarkResult, frame_w: int, frame_h: int):
    if result.poseNum != 1:
        pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        line_set.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        line_set.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        vis.update_geometry(pcd)
        vis.update_geometry(line_set)
        vis.poll_events()
        vis.update_renderer()
        return

    points = []
    colors = []
    lines = []
    line_colors = []

    # Scale to pixel space; flip Z to face camera
    z_offset = -100.0  # Pull points towards camera
    for k in range(MPP_POSE_LANDMARK_MAX_POINTS):
        x = result.points[k * 3] * frame_w
        y = result.points[k * 3 + 1] * frame_h
        z = -result.points[k * 3 + 2] * frame_w  # Flip Z sign

        # Only print first few for debugging
        print(f"Point {k}: ({result.points[k * 3]:.3f}, {result.points[k * 3 + 1]:.3f}, {result.points[k * 3 + 2]:.3f})")
        points.append([x, y, z])
        colors.append([0.9, 0.3, 0.8] if k == 0 else [0.3, 0.7, 1.0])

    for a, b in POSE_CONNECTIONS:
        lines.append([a, b])
        line_colors.append([0.2, 0.8, 0.4])

    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.array(line_colors))

    global VIEW_INIT_DONE
    if not VIEW_INIT_DONE:
        center = np.array(points).mean(axis=0)
        ctr = vis.get_view_control()
        ctr.set_lookat(center.tolist())
        ctr.set_front([0, 0, 1])    # Look towards +Z
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.50)
        VIEW_INIT_DONE = True

    vis.update_geometry(pcd)
    vis.update_geometry(line_set)
    vis.poll_events()
    vis.update_renderer()


def main():
    print("Initializing Pose Landmarker C API...")
    if lib.initPoseLandmarker() != 0:
        print("Failed to init landmarker")
        return

    detect_model = (ROOT_PATH + "pose_detector/models/pose_detection.mnn").encode('utf-8')
    landmark_model = (ROOT_PATH + "pose_landmarker/models/pose_landmark_full_sim.mnn").encode('utf-8')

    if lib.loadModelPoseDetectFromFile(detect_model, 0) != 0:
        print("Failed to load detect model")
        return
    if lib.loadModelPoseLandmarkFromFile(landmark_model, 0) != 0:
        print("Failed to load landmark model")
        return

    video_path = ROOT_PATH + "data/video/dance_2.mp4"
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Fallback to webcam (0)")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Failed to open video or webcam")
            return

    vis, pcd, line_set = create_open3d_visualizer()
    result = PoseLandmarkResult()

    try:
        count = 0
        while True:
            ret, frame = cap.read()

            # ret, frame = cap.read()

            max_width, max_height = 960, 720
            h_frame, w_frame = frame.shape[:2]
            scale = min(max_width / w_frame, max_height / h_frame)
            new_w, new_h = int(w_frame * scale), int(h_frame * scale)
            frame = cv2.resize(frame, (new_w, new_h))

            if not ret:
                break

            h, w = frame.shape[:2]
            frame_data = frame.ctypes.data_as(c_char_p)
            err = lib.runPoseLandmark(frame_data, w, h, frame.strides[0], MPP_NO_FLIP, MPP_ROTATION_0,
                                      MPP_IMAGE_TYPE_BGR, ctypes.byref(result))
            if err != 0:
                print(f"runPoseLandmark error: {err}")
                break

            display = frame.copy()
            draw_pose_opencv(display, result)
            cv2.imshow("Pose Landmarks - OpenCV", display)

            update_open3d_visualization(vis, pcd, line_set, result, w, h)

            key = cv2.waitKey(30) & 0xFF

            if count == 0:
                sleep(10)  # Slow down first frame for viewing

            if key == ord('q') or key == 27:
                break
            count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        vis.destroy_window()
        lib.releasePoseLandmark()


if __name__ == "__main__":
    main()

