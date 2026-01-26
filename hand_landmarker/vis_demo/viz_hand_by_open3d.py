#!/usr/bin/env python3
"""
Hand Landmark C API Demo with OpenCV and Open3D Visualization
Uses the C API to detect hand landmarks from webcam and displays results in two windows
"""


from time import sleep
import time
import open3d as o3d
import numpy as np
import cv2
import ctypes
from ctypes import c_int, c_float, c_char_p, POINTER, Structure

# Constants matching C API
MPP_HAND_LANDMARK_MAX_POINTS = 21
MPP_HAND_IMAGE_TYPE_BGR = 1
MPP_HAND_NO_FLIP = 0
MPP_HAND_ROTATION_0 = -1

ROOT_PATH = '/home/moo/work/my_lab/mpp_project/mediapiep_cmake_private/'

start_time =  time.time()
end_time = time.time()

print("Time elapsed in importing modules: {:.3f} seconds".format(end_time - start_time))

# One-time view initialization flag for Open3D
VIEW_INIT_DONE = False


# Hand landmark names for reference
HAND_LANDMARK_NAMES = {
    0: "Wrist",
    1: "Thumb1", 2: "Thumb2", 3: "Thumb3", 4: "Thumb4",
    5: "Index1", 6: "Index2", 7: "Index3", 8: "Index4",
    9: "Middle1", 10: "Middle2", 11: "Middle3", 12: "Middle4",
    13: "Ring1", 14: "Ring2", 15: "Ring3", 16: "Ring4",
    17: "Pinky1", 18: "Pinky2", 19: "Pinky3", 20: "Pinky4"
}

# Hand connections (same as kHandConnections in hand_landmark.h)
HAND_CONNECTIONS = [
    (0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17), (1, 2),
    (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11),
    (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)
]


# Define C API structures
class HandLandmarkResult(Structure):
    _fields_ = [
        ("landmark_count", c_int),
        ("rect", c_float * 4),  # [x, y, w, h]
        ("score", c_float),
        ("radians", c_float),
        ("points", c_float * (MPP_HAND_LANDMARK_MAX_POINTS * 3)),       # pixel coords: x,y,z,...
    ]


# Load shared library
lib = ctypes.CDLL(ROOT_PATH + 'build/libvision_hand_landmarker.so')

# Define function signatures
lib.initHandLandmarker.argtypes = [c_int, c_int]
lib.initHandLandmarker.restype = c_int

lib.loadModelHandDetectFromFile.argtypes = [c_char_p, c_int]
lib.loadModelHandDetectFromFile.restype = c_int

lib.loadModelHandLandmarkFromFile.argtypes = [c_char_p, c_int]
lib.loadModelHandLandmarkFromFile.restype = c_int

lib.runHandLandmarkVideo.argtypes = [c_char_p, c_int, c_int, c_int, c_int, c_int, c_int, 
                                      POINTER(HandLandmarkResult), c_int]
lib.runHandLandmarkVideo.restype = c_int

lib.releaseHandLandmarker.argtypes = []
lib.releaseHandLandmarker.restype = c_int


def draw_landmarks_opencv(img, hands, hand_count):
    """Draw hand landmarks on OpenCV image with skeleton"""
    h, w = img.shape[:2]
    
    for i in range(hand_count):
        hand = hands[i]
        
        # Draw bounding box
        x, y, bw, bh = hand.rect
        cv2.rectangle(img, (int(x), int(y)), (int(x + bw), int(y + bh)), (255, 0, 0), 2)
        
        # Draw score
        score_text = f"Hand {i} | Score: {hand.score:.2f}"
        cv2.putText(img, score_text, (int(x), int(y) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw hand skeleton connections
        for idx1, idx2 in HAND_CONNECTIONS:
            if idx1 < hand.landmark_count and idx2 < hand.landmark_count:
                p1 = (int(hand.points[idx1 * 3] * w), int(hand.points[idx1 * 3 + 1] * h))
                p2 = (int(hand.points[idx2 * 3] * w), int(hand.points[idx2 * 3 + 1] * h))
                cv2.line(img, p1, p2, (0, 255, 255), 2)
        
        # Draw landmarks
        for k in range(hand.landmark_count):
            px = int(hand.points[k * 3] * w)
            py = int(hand.points[k * 3 + 1] * h)

            # print x y z
            print(f"Hand {i} Landmark {k} ({HAND_LANDMARK_NAMES.get(k, 'Unknown')}):")
            print(f"  Pixel: x={(hand.points[k * 3] * w):.4f}, y={(hand.points[k * 3 + 1] * h):.4f}, z={hand.points[k * 3 + 2]:.4f}")
            cv2.circle(img, (px, py), 3, (255, 0, 255), -1)
    
    return img


def create_open3d_visualizer():
    """Create and configure Open3D visualizer for hand"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Hand 3D Landmarks - Drag to Rotate", width=800, height=600)
    
    # Configure rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 8.0
    
    # Add initial coordinate frame (sized for pixel space)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    vis.add_geometry(coord_frame)
    
    # Create empty point cloud and add it
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    # Create empty line set for skeleton
    line_set = o3d.geometry.LineSet()
    vis.add_geometry(line_set)
    
    # Set initial camera view for pixel coordinate space
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])  # Look from front towards hand (into -Z)
    ctr.set_up([0, -1, 0])     # Y-axis points down (image coordinates)
    ctr.set_zoom(0.3)
    
    return vis, pcd, line_set, coord_frame


def update_open3d_visualization(vis, pcd, line_set, hands, hand_count):
    """Update Open3D visualization with hand landmarks and skeleton using world coordinates"""
    
    if hand_count == 0:
        # Clear geometries but keep them
        pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        line_set.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        line_set.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        vis.update_geometry(pcd)
        vis.update_geometry(line_set)
        vis.poll_events()
        vis.update_renderer()
        return
    
    # Collect all landmarks from all hands
    all_points = []
    all_colors = []
    all_lines = []
    line_colors = []
    point_offset = 0
    
    # Color scheme for different hands
    colors_per_hand = [
        [1, 0, 0],  # Red for first hand
        [1, 0, 0],  # Red for first hand
        # [0, 1, 0],  # Green for second hand
    ]
    
    for hand_idx in range(hand_count):
        hand = hands[hand_idx]
        hand_color = colors_per_hand[hand_idx % len(colors_per_hand)]
        
        # Add landmarks for this hand using pixel coordinates (scaled for better visualization)
        hand_points = []
        for k in range(hand.landmark_count):
            # Use pixel coordinates - they have better range and spread
            # Multiply by image dimensions to convert from normalized [0,1] to pixel space
            x = hand.points[k * 3] * 640      # pixel x coordinate
            y = hand.points[k * 3 + 1] * 480  # pixel y coordinate  

            z_offset = 50.0   # 把整个手推到相机前方

            z = hand.points[k * 3 + 2] * 480 - z_offset
            # z = hand.points_world[k * 3 + 2] * 480  # pixel z coordinate (scaled to match x/y range)
            
            print("k = ", k, f"Pixel: x={x:.2f}, y={y:.2f}, z={z:.2f}",)

            all_points.append([x, y, z])
            hand_points.append(len(all_points) - 1)
            
            # Color points by joint type
            if k == 0:  # Wrist - bright color
                all_colors.append(hand_color)
            else:
                # Shade other points slightly
                all_colors.append([c * 0.8 for c in hand_color])
        
        # Add skeleton connections for this hand
        for idx1, idx2 in HAND_CONNECTIONS:
            if idx1 < hand.landmark_count and idx2 < hand.landmark_count:
                all_lines.append([hand_points[idx1], hand_points[idx2]])
                line_colors.append(hand_color)
    
    if len(all_points) > 0:
        # Update point cloud
        pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))
        
        # Update line set (skeleton)
        line_set.points = o3d.utility.Vector3dVector(np.array(all_points))
        if len(all_lines) > 0:
            line_set.lines = o3d.utility.Vector2iVector(np.array(all_lines, dtype=np.int32))
            line_set.colors = o3d.utility.Vector3dVector(np.array(line_colors))

        # Initialize camera to face the hand on first detection
        global VIEW_INIT_DONE
        if not VIEW_INIT_DONE:
            np_pts = np.array(all_points)
            center = np_pts.mean(axis=0)
            ctr = vis.get_view_control()
            ctr.set_lookat(center.tolist())
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, -1, 0])
            ctr.set_zoom(0.6)
            VIEW_INIT_DONE = True
        
        # Update geometries (this preserves camera view)
        vis.update_geometry(pcd)
        vis.update_geometry(line_set)
    
    vis.poll_events()
    vis.update_renderer()


def main():
    # Initialize C API
    print("Initializing Hand Landmarker C API...")
    ret = lib.initHandLandmarker(2, 0)  # max 2 hands, CPU
    if ret != 0:
        print(f"Failed to initialize hand landmarker: {ret}")
        return
    
    # Load models
    detect_model = ROOT_PATH + "hand_detector/models/palm_detection_full.mnn"
    landmark_model = ROOT_PATH + "hand_landmarker/models/hand_landmark_full.mnn"
    
    print("Loading detection model...")
    ret = lib.loadModelHandDetectFromFile(detect_model.encode('utf-8'), 0)
    if ret != 0:
        print(f"Failed to load detection model: {ret}")
        lib.releaseHandLandmarker()
        return
    
    print("Loading landmark model...")
    ret = lib.loadModelHandLandmarkFromFile(landmark_model.encode('utf-8'), 0)
    if ret != 0:
        print(f"Failed to load landmark model: {ret}")
        lib.releaseHandLandmarker()
        return
    
    # Open webcam
    video_path = ROOT_PATH + "data/video/hands03.mp4"
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open webcam")
        lib.releaseHandLandmarker()
        return
    
    # Create Open3D visualizer
    vis, pcd, line_set, coord_frame = create_open3d_visualizer()
    
    # Result buffer
    hands = (HandLandmarkResult * 2)()
    
    print("Starting hand detection loop.")
    print("Controls:")
    print("  - Left mouse drag: Rotate view")
    print("  - Right mouse drag: Pan view") 
    print("  - Scroll wheel: Zoom in/out")
    print("  - Press 'q' in OpenCV window to quit")
    print("\nHand Landmark Points (0-20):")
    print("  0: Wrist")
    print("  1-4: Thumb")
    print("  5-8: Index")
    print("  9-12: Middle")
    print("  13-16: Ring")
    print("  17-20: Pinky")
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()

            max_width, max_height = 960, 720
            h_frame, w_frame = frame.shape[:2]
            scale = min(max_width / w_frame, max_height / h_frame)
            new_w, new_h = int(w_frame * scale), int(h_frame * scale)
            frame = cv2.resize(frame, (new_w, new_h))

            if not ret:
                break
            
            h, w = frame.shape[:2]
            
            # Call C API
            frame_data = frame.ctypes.data_as(c_char_p)
            hand_count = lib.runHandLandmarkVideo(
                frame_data, w, h, w * 3,
                MPP_HAND_NO_FLIP, MPP_HAND_ROTATION_0,
                MPP_HAND_IMAGE_TYPE_BGR,
                hands, 2
            )
            
            if hand_count < 0:
                print(f"Error in runHandLandmarkVideo: {hand_count}")
                break
            
            # Draw on OpenCV window
            # display_frame = frame.copy()
            frame = draw_landmarks_opencv(frame, hands, hand_count)
            
            # Add info counter
            cv2.putText(frame, f"Hands: {hand_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Resize frame for display if too large
            # Resize frame for display while maintaining aspect ratio

            cv2.imshow("Hand Landmarks - OpenCV", frame)
            
            # Update Open3D window (preserves camera view for mouse interaction)
            update_open3d_visualization(vis, pcd, line_set, hands, hand_count)
            
            frame_count += 1
            
            if frame_count == 1:
                sleep(10)  # Slow down first frame for viewing

            # Check for quit
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            
    finally:
        # Cleanup
        print(f"Processed {frame_count} frames. Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        vis.destroy_window()
        lib.releaseHandLandmarker()
        print("Done!")


if __name__ == "__main__":
    main()

