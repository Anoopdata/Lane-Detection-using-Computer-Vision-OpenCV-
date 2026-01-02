# lane_detection.py
# ------------------------------
# Self-Driving Car Lane Detection
# ------------------------------

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from moviepy.editor import VideoFileClip

# ------------------------------
# Helper Functions
# ------------------------------

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    if lines is None:
        return

    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            if slope < -0.5:
                left_lines.append((slope, x1, y1, x2, y2))
                left_weights.append(length)
            elif slope > 0.5:
                right_lines.append((slope, x1, y1, x2, y2))
                right_weights.append(length)

    def average_lane(lines, weights):
        if len(lines) == 0:
            return None
        slope_avg = np.dot([l[0] for l in lines], weights) / np.sum(weights)
        x_coords = [l[1] for l in lines] + [l[3] for l in lines]
        y_coords = [l[2] for l in lines] + [l[4] for l in lines]
        intercept = np.mean(y_coords) - slope_avg * np.mean(x_coords)
        y1 = img.shape[0]
        y2 = int(img.shape[0]*0.6)
        x1 = int((y1 - intercept)/slope_avg)
        x2 = int((y2 - intercept)/slope_avg)
        return ((x1, y1), (x2, y2))

    left_lane = average_lane(left_lines, left_weights)
    right_lane = average_lane(right_lines, right_weights)

    if left_lane is not None:
        cv2.line(img, left_lane[0], left_lane[1], color, thickness)
    if right_lane is not None:
        cv2.line(img, right_lane[0], right_lane[1], color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(
        img, rho, theta, threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

# ------------------------------
# Process Image Pipeline
# ------------------------------

def process_image(image):
    gray = grayscale(image)
    blur = gaussian_blur(gray, 5)
    edges = canny(blur, 50, 150)

    imshape = image.shape
    vertices = np.array([[
        (0, imshape[0]),
        (imshape[1]*0.45, imshape[0]*0.6),
        (imshape[1]*0.55, imshape[0]*0.6),
        (imshape[1], imshape[0])
    ]], dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices)

    lines_img = hough_lines(
        masked_edges,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        min_line_len=20,
        max_line_gap=300
    )

    result = weighted_img(lines_img, image)
    return result

# ------------------------------
# Main Execution
# ------------------------------

# Create output folder
os.makedirs("test_videos_output", exist_ok=True)

# White lane video
white_output = '/home/inxee/Videos/walktest/center_line_detection/CarND-LaneLines-P1/test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip('/home/inxee/Videos/walktest/center_line_detection/CarND-LaneLines-P1/test_videos/solidWhiteRight.mp4')
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(
    white_output,
    audio=False,
    codec='libx264',
    fps=clip1.fps  # important to avoid NoneType error
)

# Yellow lane video
yellow_output = '/home/inxee/Videos/walktest/center_line_detection/CarND-LaneLines-P1/test_videos_output/solidYellowLeft.mp4'
clip2 = VideoFileClip('/home/inxee/Videos/walktest/center_line_detection/CarND-LaneLines-P1/test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(
    yellow_output,
    audio=False,
    codec='libx264',
    fps=clip2.fps
)






















'''
# lane_detection.py
# ------------------------------
# Self-Driving Car Lane Detection
# ------------------------------

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from moviepy.editor import VideoFileClip

# ------------------------------
# Helper Functions
# ------------------------------

def grayscale(img):
    cv2.imwrite("rgb.jpg", img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        

def canny(img, low_threshold, high_threshold):
    cv2.imwrite("gray.jpg", img)
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    cv2.imwrite("canny.jpg", img)
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    if lines is None:
        return

    left_lines = []
    left_weights = []
    #right_lines = []
    #right_weights = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            if slope < -0.5:
                left_lines.append((slope, x1, y1, x2, y2))
                left_weights.append(length)
            #elif slope > 0.5:
                #right_lines.append((slope, x1, y1, x2, y2))
                #right_weights.append(length)

    def average_lane(lines, weights):
        if len(lines) == 0:
            return None
        slope_avg = np.dot([l[0] for l in lines], weights) / np.sum(weights)
        x_coords = [l[1] for l in lines] + [l[3] for l in lines]
        y_coords = [l[2] for l in lines] + [l[4] for l in lines]
        intercept = np.mean(y_coords) - slope_avg * np.mean(x_coords)
        y1 = img.shape[0]
        y2 = int(img.shape[0]*0.6)
        x1 = int((y1 - intercept)/slope_avg)
        x2 = int((y2 - intercept)/slope_avg)
        return ((x1, y1), (x2, y2))

    left_lane = average_lane(left_lines, left_weights)
    #right_lane = average_lane(right_lines, right_weights)

    if left_lane is not None:
        cv2.line(img, left_lane[0], left_lane[1], color, thickness)
    #if right_lane is not None:
        #cv2.line(img, right_lane[0], right_lane[1], color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(
        img, rho, theta, threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1.0, γ=0):
    return cv2.addWeighted(initial_img, α, img, β, γ)

# ------------------------------
# Process Image Pipeline
# ------------------------------

def process_image(image):
    gray = grayscale(image)
    blur = gaussian_blur(gray, 5)
    edges = canny(blur, 50, 80)

    imshape = image.shape
    vertices = np.array([[
        (0, imshape[0]),
        (imshape[1]*0.45, imshape[0]*0.6),
        (imshape[1]*0.55, imshape[0]*0.6),
        (imshape[1], imshape[0])
    ]], dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices)

    lines_img = hough_lines(
        masked_edges,
        rho=1,
        theta=np.pi/180,
        threshold=20,
        min_line_len=30,
        max_line_gap=50
    )

    result = weighted_img(lines_img, image)
    return result

# ------------------------------
# Main Execution
# ------------------------------

# Create output folder
os.makedirs("test_videos_output", exist_ok=True)

# White lane video
#white_output = '/home/inxee/Videos/walktest/center_line_detection/CarND-LaneLines-P1/output2.mp4'
#clip1 = VideoFileClip('/home/inxee/Videos/walktest/center_line_detection/CarND-LaneLines-P1/test_videos/solidWhiteRight.mp4')
#white_clip = clip1.fl_image(process_image)
#white_clip.write_videofile(
 #   white_output,
  #  audio=False,
  #  codec='libx264',
  #  fps=clip1.fps  # important to avoid NoneType error
#)

# Yellow lane video


yellow_output = '/home/inxee/Videos/walktest/center_line_detection/CarND-LaneLines-P1/output1.mp4'

clip2 = VideoFileClip('/home/inxee/Videos/walktest/center_line_detection/CarND-LaneLines-P1/test_videos_output/output_part_01.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(
    yellow_output,
    audio=False,
    codec='libx264',
    fps=clip2.fps
)
'''