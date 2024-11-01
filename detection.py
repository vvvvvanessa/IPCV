import cv2
import numpy as np
import math


def detect_object_in_ROI(prev_frame, next_frame, center_x, center_y, ROI_width=31, ROI_height=11):
    """
    Detect the location of the object in prev_frame (within ROI).
    1. If object detected, output its location in prev_frame.
    2. If object undetected, output the location prediction as reference for ROI.
    """
    # ROI borderline definition
    x_start = max(center_x - ROI_width // 2, 0)
    y_start = max(center_y - ROI_height // 2, 0)
    x_end = min(center_x + ROI_width // 2, prev_frame.shape[1] - 1)
    y_end = min(center_y + ROI_height // 2, prev_frame.shape[0] - 1)
    prev_roi = prev_frame[y_start:y_end, x_start:x_end]
    next_roi = next_frame[y_start:y_end, x_start:x_end]
    next_gray = cv2.cvtColor(next_roi, cv2.COLOR_BGR2GRAY)
    
    # Canny edge detection
    edges = cv2.Canny(next_gray, 180, 200)
    # cv2.imshow("edges", edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    loc_x, loc_y = center_x, center_y
    is_success = False

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # calculate the center of the contour as the detection point.
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            loc_x, loc_y = x_start + cX, y_start + cY
            is_success = True
        else:
            is_success = False

    return loc_x, loc_y, is_success