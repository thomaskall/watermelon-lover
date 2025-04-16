
import cv2
import numpy as np
from time import *
from glob import glob as glob # glob
import os
import json
from loguru import logger
from threading import Thread

logger.add("logs/extract_features.log")
logger.add("logs/debug.log", filter=lambda record: record["level"].name == "DEBUG")

cwd = os.getcwd()
BOX_HEIGHT: float = 0.5 # to-scale height of the box in meters

def get_color(channel_dict, img_masked, original_img):
    """Return average RGB values for the thresholded image"""
    
    blur_kernel = (channel_dict["fuzz"], channel_dict["fuzz"])
    mask_kernel = (channel_dict["kernel"], channel_dict["kernel"])

    lower_bgr = np.array([channel_dict["lower"]["b"], channel_dict["lower"]["g"], channel_dict["lower"]["r"]])
    upper_bgr = np.array([channel_dict["upper"]["b"], channel_dict["upper"]["g"], channel_dict["upper"]["r"]])
    thresholded = cv2.inRange(img_masked, lower_bgr, upper_bgr)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, (8,8), iterations=1)
    img_blur = cv2.GaussianBlur(img_masked, mask_kernel, sigmaX=0, sigmaY=0)
    _, mask = cv2.threshold(img_blur, 175, 255, cv2.THRESH_BINARY)
    img_masked = cv2.bitwise_and(original_img, original_img, mask=mask)

    img_masked = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)
    average = (0, 0, 0)
    # Find the average colors of each channel
    
    return average

def calculate_volume(image: np.array) -> int:
    volume = 0
    for row in image[0]:
        r = sum(row) / 2
        volume +=  np.pi * r**2
    return volume
    
def calculate_height(image: np.array) -> float:
    """Use the BOX_HEIGHT to calculate the metric height of one pixel"""
    # Bottom of the watermelon is the ground
    try:
        coords = cv2.findNonZero(image) # (x,y)
        assert coords is not None
        bottom_row = max(coords, key=lambda x: x[0][1])[0][1]
        return BOX_HEIGHT / (bottom_row + 1)
    except AssertionError as e:
        logger.critical("No watermelon detected, exiting the program")
        exit(1)
    except Exception as e:
        logger.warning(f"Error when calculating pixel height")

def extract_features(img: np.array):
    with open("./color_dicts/yellow_checkpoint.json", 'r') as json_file:
        yellow_channel_dict = json.load(json_file)
    with open("./color_dicts/green_checkpoint.json", 'r') as json_file:
        green_channel_dict = json.load(json_file)

    mask_kernel = np.ones((green_channel_dict["kernel"], green_channel_dict["kernel"]), np.uint8)
    
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([green_channel_dict["lower"]["h"], green_channel_dict["lower"]["s"], green_channel_dict["lower"]["v"]])
    upper_hsv = np.array([green_channel_dict["upper"]["h"], green_channel_dict["upper"]["s"], green_channel_dict["upper"]["v"]])
    hsv_image = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    hsv_image = cv2.morphologyEx(hsv_image, cv2.MORPH_CLOSE, mask_kernel, iterations=1)
    hsv_image = cv2.GaussianBlur(hsv_image, (3, 3), sigmaX=0, sigmaY=0)

    _, mask = cv2.threshold(hsv_image, 200, 255, cv2.THRESH_BINARY)

    volume = calculate_volume(mask)
    print("Watermelon's volume was %.2f" % volume)

    # This is where things will be different, start threads
    img_masked = cv2.bitwise_and(img, img, mask=mask)
    green_stuff = find_colors(green_channel_dict, img_masked, img)
    yellow_stuff = find_colors(yellow_channel_dict, img_masked, img)

def main():
    imgs = glob(os.path.join(cwd, "img", "*"))
    for img in imgs:
        extract_features(cv2.imread(img))

if __name__ == "__main__":
    main()