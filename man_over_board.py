import cv2
import numpy as np
import math
from calib import calib_videos
from detection import *

def extend_line_to_left_right_edges(x1, y1, x2, y2, width):
    """
    Calulate the intersections of a short line with the left and right border of the image.
    """
    if x1 == x2:
        return (x1, 0), (x2, width)
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return (0, b), (width, m * width + b)

def detect_horizon(image):
    """
    Here we can detect the horizon line from the frame image.
    """
    height, width = image.shape[:2]
    blurred = cv2.GaussianBlur(image, (5, 5), 0) # Removing unnecessary details.
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 150, 200)
    # Fine tuned parameters for straight line detecting.
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=325, minLineLength=width/30, maxLineGap=10000)

    if lines is not None:
        best_line = None
        max_len = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # Pick the longest line from good lines as the best.
            if length > max_len:
                max_len = length
                best_line = (x1, y1, x2, y2)

        if best_line is not None:
            x1, y1, x2, y2 = best_line
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            tilt_angle_final = np.arctan2(y2 - y1, x2 - x1)
            return (x1, y1, x2, y2), tilt_angle_final, image
        
    return None, None, image

def calculate_pitch(horizon_y, camera_matrix):
    """
    Calculating pitch with the middle point's y of horizon line.
    """
    cy = camera_matrix[1, 2]
    fy = camera_matrix[1, 1]
    pitch = math.atan2(horizon_y - cy, fy)
    return pitch

def correct_image_perspective(image, pitch, tilt, camera_matrix):
    """
    Stablizing the image given pitch, tilt and camera matrix
    """
    h, w = image.shape[:2]

    # Rotation matrices
    R_pitch = np.array([[1, 0, 0],
                        [0, np.cos(pitch), -np.sin(pitch)],
                        [0, np.sin(pitch), np.cos(pitch)]], dtype=np.float32)

    R_tilt = np.array([[np.cos(tilt), -np.sin(tilt), 0],
                       [np.sin(tilt), np.cos(tilt), 0],
                       [0, 0, 1]], dtype=np.float32)

    R = R_tilt @ R_pitch

    # Perspective transformation matrix： K * R * K^-1
    K_inv = np.linalg.inv(camera_matrix)
    H = camera_matrix @ R @ K_inv
    corrected_image = cv2.warpPerspective(image, H, (w, h))

    return corrected_image

def calculate_distance_to_target(x, y, camera_matrix, camera_height=2.5):
    """
    Return the real distance of the object based on its location in the image.
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    x_offset = x - cx
    y_offset = y - cy

    if y_offset <= 0:
        print("The object is above the horizon, distance undetectable!")
        return 0

    z = camera_height * fy / y_offset
    x = (x_offset / fx) * z
    distance = np.sqrt(x**2 + z**2)
    
    return distance

def display_object_information(frame, loc_x, loc_y, success, camera_matrix):
    """
    Render the buoy information for the frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 255, 255)
    loc_point = (int(loc_x), int(loc_y))
    distance = calculate_distance_to_target(loc_x, loc_y, camera_matrix)
    if success:
        cv2.circle(frame, loc_point, 8, (0, 0, 255), 2)
        subtitle = f"Bouy distance: {round(distance, 1)} meters."
    else:
        text_color = (0, 0, 255)
        subtitle = f"Bouy distance: {round(distance, 1)} meters (target lost)."
    cv2.putText(frame, subtitle, (10, 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    return frame

def process_video(input_video, output_video, camera_matrix, dist_coeffs):
    """
    First, use the intrinsic matrix to calibrate the video.
    Second, rectify the video based on horizon information.
    """
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % fps == 0:
            print(f"Calibration and stablization progress: {frame_count // fps}s")
        
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        horizon_line, tilt_angle, horizon_frame = detect_horizon(undistorted_frame)
        
        if horizon_line is not None:
            x1, y1, x2, y2 = horizon_line
            left, right = extend_line_to_left_right_edges(x1, y1, x2, y2, width)
            cv2.line(horizon_frame, (int(left[0]), int(left[1])), (int(right[0]), int(right[1])), (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Stablization
        if horizon_line is not None:
            pitch = calculate_pitch((left[1] + right[1]) / 2, camera_matrix)
            corrected_frame = correct_image_perspective(horizon_frame, pitch, -tilt_angle, camera_matrix)
            cv2.imshow('Stablized frame', corrected_frame)
            out.write(corrected_frame)
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def detect_buoy(input_video, output_video, camera_matrix, start_x=659, start_y=587):
    """
    Use optical flow to estimate the location of the buoy.
    """
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    _, prev_frame = cap.read()
    ROI_x, ROI_y = start_x, start_y
    
    frame_count = 0
    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % fps == 0:
            print(f"Detection progress: {frame_count // fps}s")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        loc_x, loc_y, success = detect_object_in_ROI(prev_frame, curr_frame, ROI_x, ROI_y)
        prev_frame = curr_frame.copy()
        ROI_x, ROI_y = loc_x, loc_y
        output_frame = display_object_information(curr_frame, loc_x, loc_y, success, camera_matrix)
        cv2.imshow('Object detection:', output_frame)
        out.write(output_frame)
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_video_path = 'MAH01462.MP4'
    stable_video_path = 'output_video_stablized.mp4'
    output_video_path = 'distance_estimation.mp4'
    # Getting the intrinsic coefficients of the camera with checkerboard.
    camera_matrix, dist_coeffs = calib_videos()
    # Calibration and stablization of the video.
    process_video(input_video_path, stable_video_path, camera_matrix, dist_coeffs)
    # Detect the buoy and estimate the distance, then generate the output video.
    detect_buoy(stable_video_path, output_video_path, camera_matrix)