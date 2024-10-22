import cv2
import numpy as np
import math
from calib import calib_videos

# Getting the intrinsic coefficients of the camera with checkerboard.
camera_matrix, dist_coeffs = calib_videos()

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


def process_video(input_video, output_video, camera_matrix, dist_coeffs):
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
        # if frame_count % fps == 0:
        #     print(f"Time: {frame_count // fps}s")
        
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


if __name__ == '__main__':
    input_video_path = 'MAH01462.MP4'
    output_video_path = 'output_video_stablized.mp4'
    process_video(input_video_path, output_video_path, camera_matrix, dist_coeffs)