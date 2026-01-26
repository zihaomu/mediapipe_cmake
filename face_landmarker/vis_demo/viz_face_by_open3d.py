#!/usr/bin/env python3
"""
Face Landmark C API Demo with OpenCV and Open3D Visualization
Uses the C API to detect face landmarks from webcam and displays results in two windows
"""

from time import sleep
import open3d as o3d
import numpy as np
import cv2
import ctypes
from ctypes import c_int, c_float, c_char_p, POINTER, Structure

# Constants matching C API
MPP_FACE_LANDMARK_MAX_POINTS = 478
MPP_FACE_IMAGE_TYPE_BGR = 1
MPP_FACE_NO_FLIP = 0
MPP_FACE_ROTATION_0 = -1

ROOT_PATH = '/home/moo/work/my_lab/mpp_project/mediapiep_cmake_private/'


# Define C API structures
class FaceLandmarkResult(Structure):
    _fields_ = [
        ("landmark_count", c_int),
        ("rect", c_float * 4),  # [x, y, w, h]
        ("score", c_float),
        ("radians", c_float),
        ("points", c_float * (MPP_FACE_LANDMARK_MAX_POINTS * 3))  # x,y,z,...
    ]


# Load shared library
lib = ctypes.CDLL(ROOT_PATH + 'build/libvision_face_landmarker.so')

# Define function signatures
lib.initFaceLandmarker.argtypes = [c_int, c_int]
lib.initFaceLandmarker.restype = c_int

lib.loadModelFaceDetectFromFile.argtypes = [c_char_p, c_int]
lib.loadModelFaceDetectFromFile.restype = c_int

lib.loadModelFaceLandmarkFromFile.argtypes = [c_char_p, c_int]
lib.loadModelFaceLandmarkFromFile.restype = c_int

lib.runFaceLandmarkVideo.argtypes = [c_char_p, c_int, c_int, c_int, c_int, c_int, c_int, 
                                      POINTER(FaceLandmarkResult), c_int]
lib.runFaceLandmarkVideo.restype = c_int

lib.releaseFaceLandmarker.argtypes = []
lib.releaseFaceLandmarker.restype = c_int

lib.getFaceLandmarkDimension.argtypes = []
lib.getFaceLandmarkDimension.restype = c_int


def draw_landmarks_opencv(img, faces, face_count):
    """Draw face landmarks on OpenCV image"""
    h, w = img.shape[:2]
    
    for i in range(face_count):
        face = faces[i]
        
        # Draw bounding box
        x, y, bw, bh = face.rect
        cv2.rectangle(img, (int(x), int(y)), (int(x + bw), int(y + bh)), (255, 0, 0), 2)
        
        # Draw score
        score_text = f"Score: {face.score:.2f}"
        cv2.putText(img, score_text, (int(x), int(y) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw landmarks
        for k in range(face.landmark_count):
            px = face.points[k * 3] * w
            py = face.points[k * 3 + 1] * h
            cv2.circle(img, (int(px), int(py)), 1, (255, 0, 255), -1)
    
    return img


def create_open3d_visualizer():
    """Create and configure Open3D visualizer"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Face 3D Landmarks - Drag to Rotate", width=800, height=600)
    
    # Configure rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 5.0
    
    # Add initial coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coord_frame)
    
    # Create empty point cloud and add it
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    # Set initial camera view with proper world coordinates
    # Y-axis points up, Z-axis points towards viewer
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 1])   # Look from +Z towards origin
    ctr.set_up([0, 1, 0])      # Y-axis points up (standard world coordinates)
    ctr.set_zoom(0.6)
    
    return vis, pcd, coord_frame


def update_open3d_visualization(vis, pcd, faces, face_count, img_width, img_height):
    """Update Open3D point cloud with face landmarks without resetting camera"""
    
    if face_count == 0:
        # Clear points but keep geometry
        pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        return
    
    # Collect all landmarks from all faces
    all_points = []
    all_colors = []
    
    colors_per_face = [
        [1, 0, 0],  # Red for first face
        [0, 1, 0],  # Green for second face
        [0, 0, 1],  # Blue for third face
    ]
    
    for i in range(face_count):
        face = faces[i]
        color = colors_per_face[i % len(colors_per_face)]
        
        for k in range(face.landmark_count):
            # Convert normalized coordinates to world coordinates
            # Keep proper aspect ratio: z depth is much smaller than x,y range
            x = (face.points[k * 3] - 0.5) * 2.0          # X: [-1, 1] left to right
            y = (0.5 - face.points[k * 3 + 1]) * 1.5      # Y: [-1, 1] bottom to top
            z = -face.points[k * 3 + 2] * 2             # Z: scaled down to match face depth
            
            all_points.append([x, y, z])
            all_colors.append(color)
    
    if len(all_points) > 0:
        # Update existing point cloud instead of creating new one
        pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))
        
        # Update geometry (this preserves camera view)
        vis.update_geometry(pcd)
    
    vis.poll_events()
    vis.update_renderer()


def main():
    # Initialize C API
    print("Initializing Face Landmarker C API...")
    ret = lib.initFaceLandmarker(2, 0)  # max 2 faces, CPU
    if ret != 0:
        print(f"Failed to initialize landmarker: {ret}")
        return
    
    # Load models
    detect_model = ROOT_PATH + "face_detector/models/face_detection_short_range.mnn"
    landmark_model = ROOT_PATH + "face_landmarker/models/face_landmark478.mnn"
    
    print("Loading detection model...")
    ret = lib.loadModelFaceDetectFromFile(detect_model.encode('utf-8'), 0)
    if ret != 0:
        print(f"Failed to load detection model: {ret}")
        lib.releaseFaceLandmarker()
        return
    
    print("Loading landmark model...")
    ret = lib.loadModelFaceLandmarkFromFile(landmark_model.encode('utf-8'), 0)
    if ret != 0:
        print(f"Failed to load landmark model: {ret}")
        lib.releaseFaceLandmarker()
        return
    
    landmark_dim = lib.getFaceLandmarkDimension()
    print(f"Landmark dimension: {landmark_dim}")
    
    # Open webcam
    video_path = ROOT_PATH + "data/video/face_smile.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open webcam")
        lib.releaseFaceLandmarker()
        return
    
    # Create Open3D visualizer
    vis, pcd, coord_frame = create_open3d_visualizer()
    
    # Result buffer
    faces = (FaceLandmarkResult * 2)()
    
    print("Starting detection loop.")
    print("Controls:")
    print("  - Left mouse drag: Rotate view")
    print("  - Right mouse drag: Pan view") 
    print("  - Scroll wheel: Zoom in/out")
    print("  - Press 'q' in OpenCV window to quit")
    
    try:
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            max_width, max_height = 720, 960
            h_frame, w_frame = frame.shape[:2]
            scale = min(max_width / w_frame, max_height / h_frame)
            new_w, new_h = int(w_frame * scale), int(h_frame * scale)
            frame = cv2.resize(frame, (new_w, new_h))


            h, w = frame.shape[:2]
            
            # Call C API
            frame_data = frame.ctypes.data_as(c_char_p)
            face_count = lib.runFaceLandmarkVideo(
                frame_data, w, h, w * 3,
                MPP_FACE_NO_FLIP, MPP_FACE_ROTATION_0,
                MPP_FACE_IMAGE_TYPE_BGR,
                faces, 2
            )
            
            if face_count < 0:
                print(f"Error in runFaceLandmarkVideo: {face_count}")
                break
            
            # Draw on OpenCV window
            display_frame = frame.copy()
            draw_landmarks_opencv(display_frame, faces, face_count)
            
            # Add FPS counter
            cv2.putText(display_frame, f"Faces: {face_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Face Landmarks - OpenCV", display_frame)
            
            # Update Open3D window (preserves camera view for mouse interaction)
            update_open3d_visualization(vis, pcd, faces, face_count, w, h)
            
            # Check for quit
            if count == 0:
                sleep(10)  # Slow down first frame for viewing

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break

            count += 1
            
    finally:
        # Cleanup
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        vis.destroy_window()
        lib.releaseFaceLandmarker()
        print("Done!")


if __name__ == "__main__":
    main()

