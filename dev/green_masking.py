import cv2
import numpy as np
from keyboard import read_key
from threading import Thread
from time import *
from pprint import pp
from glob import glob as glob # glob
import os
import json
import argparse
import pdb

cwd = os.getcwd()

channel_dict = {
    "upper": {
        "r": 255,
        "g": 255,
        "b": 255,

        "h": 179,
        "s": 255,
        "v": 255
    },
    "lower": {
        "r": 0,
        "g": 0,
        "b": 0,

        "h": 0,
        "s": 0,
        "v": 0
    },
    "kernel": 1,
    "fuzz": 3
}
pic_idx = 0

def keyboard_watcher(len_imgs):
    global channel_dict
    global pic_idx
    name_dict = {
        "r": "red",
        "g": "green",
        "b": "blue",

        "h": "hue",
        "s": "saturation",
        "v": "value",

        "k": "kernel"
    }
    channel = "g"
    level = "lower"
    print("\nStarting off on lower green channel")

    last = None

    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    while True:
        key = read_key()
        
        match key:
            case "u":
                level = "upper"
                print(f"\n{name_dict[channel]} upper limit selected")
                sleep(0.5)
            case "l":
                level = "lower"
                print(f"\n{name_dict[channel]} lower limit selected")
                sleep(0.5)

            # Color channels
            case "r" | "g" | "b" | "h" | "s" | "v":
                channel = key
                print(f"\n{level} {name_dict[key]} selected, currently at {channel_dict[level][key]}")
                sleep(0.5)

            # kernel
            case "k":
                channel = key
                print("\nKernel selected")
                sleep(0.5)

            case "f":
                channel = key
                print("Gaussian blur kernel chosen")
                sleep(0.5)

            # Control over channel values
            case "up":
                if channel != "k" and channel != "f":
                    if level == "upper":
                        if channel == "h":
                            max_val = 179
                        else: max_val = 255
                    else:
                        max_val = channel_dict["upper"][channel] - 2

                if last == "up" or last == "down":
                    print(LINE_UP, end=LINE_CLEAR)

                if channel == "k":
                    channel_dict["kernel"] += 1
                    print(f"({channel_dict['kernel']}, {channel_dict['kernel']})")
                    sleep(0.5)
                elif channel == "f":
                    channel_dict["fuzz"] += 2
                    print(f"({channel_dict['fuzz']}, {channel_dict['fuzz']})")
                    sleep(0.5)
                elif channel_dict[level][channel] == max_val:
                    print(f"Max value for {name_dict[channel]} ({max_val}) already reached")
                else:
                    channel_dict[level][channel] += 1
                    print(f"{channel_dict[level][channel]}")

            case "down":
                if channel == "f" or channel == "k":
                    min_val = 1
                else:
                    if level == "lower":
                        min_val = 0
                    else:
                        min_val = channel_dict["lower"][channel] + 1

                if last == "up" or last == "down":
                    print(LINE_UP, end=LINE_CLEAR)

                if channel == "k":
                    if channel_dict["kernel"] == min_val:
                        print(f"Can't go any lower than ({min_val}, {min_val})")
                    else:
                        channel_dict["kernel"] -= 1
                        print(f"({channel_dict['kernel']}, {channel_dict['kernel']})")
                        sleep(0.5)
                elif channel == "f":
                    if channel_dict["fuzz"] == min_val:
                        print(f"Can't go any lower than ({min_val}, {min_val})")
                    else:
                        channel_dict["fuzz"] -= 2
                        print(f"({channel_dict['fuzz']}, {channel_dict['fuzz']})")
                        sleep(0.5)
                elif channel_dict[level][channel] == min_val:
                    print(f"Min value for {name_dict[channel]} ({min_val}) already reached")
                else:
                    channel_dict[level][channel] -= 1
                    print(f"{channel_dict[level][channel]}")

            case "right":
                if last == "left" or last == "right":
                    print(LINE_UP, end=LINE_CLEAR)
                if pic_idx < len_imgs-1:
                    pic_idx += 1
                    print(f"Image {pic_idx}")
                    sleep(1)
                else:
                    print(f"Image {pic_idx}, no more to the right")

            case "left":
                if last == "left" or last == "right":
                    print(LINE_UP, end=LINE_CLEAR)
                if pic_idx > 0:
                    pic_idx -= 1
                    print(f"Image {pic_idx}")
                    sleep(1)
                else:
                    print(f"Image 0, no more to the left")

            # Enter a value for current channel
            case "n":
                sleep(0.5)
                if channel == "h":
                    max_val = 179
                else: max_val = 255

                if channel == "k":
                    val = input(f"\nEnter a value to set the kernel to:\n" + LINE_CLEAR)
                else:
                    val = input(f"\nEnter a value to assign to {level} {name_dict[channel]} from 0-{max_val}, currently {channel_dict[level][channel]}: \n" + LINE_CLEAR)
                valid = False
                
                while not valid:
                    try:
                        if "q" in val:
                            exit()

                        val = int(val)
                        if val < 0 or val > max_val:
                            print("\nBruh...fr? Enter a valid number or something with 'q' to quit the program.")
                            val = input(f"Try again... \n" + LINE_CLEAR)
                        else:
                            if channel == "k":
                                channel_dict['kernel'] = val
                            else:
                                channel_dict[level][channel] = val
                            valid = True
                            print("Value updated")
                            pp(channel_dict)
                            sleep(1)

                    except Exception as e:
                        print(f"An error occurred:\n\t{e}")
                        val = input("\nTry again...\n" + LINE_CLEAR)
            

            # Print what we've got so far
            case "enter":
                print()
                pp(channel_dict)
                with open("./color_dicts/green_checkpoint.json", 'w') as json_file:
                    json.dump(channel_dict, json_file, indent=4) 
                sleep(1)

            # quit the program
            case "q":
                exit()

            case _:
                print(f'"{key}" not recognized')
                print('If you are trying to exit the program, press "q"')
                sleep(1)

        last = key




def main(args):

    global cwd, channel_dict, pic_idx, imgs
    if args.load:
        with open("./color_dicts/green_checkpoint.json", 'r') as json_file:
            channel_dict = json.load(json_file)
    imgs = glob(os.path.join(cwd, "img", "*"))
    pic_idx = 0

    key_thread = Thread(target=keyboard_watcher, args=(len(imgs),), daemon=True)
    key_thread.start()

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Dark Green", cv2.WINDOW_NORMAL)
    cv2.namedWindow("HSV", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    
    while key_thread.is_alive():
        # read image
        img = cv2.imread(imgs[pic_idx])
        w, h, _ = img.shape
        height_scale = w/h
        h = 500
        w = int(h * height_scale)
        cv2.resizeWindow("Original", h, w)
        cv2.imshow("Original", img)

        mask_kernel = np.ones((channel_dict["kernel"], channel_dict["kernel"]), np.uint8)
        blur_kernel = (channel_dict["fuzz"], channel_dict["fuzz"])
        
        # Get the watermelon by itself
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([channel_dict["lower"]["h"], channel_dict["lower"]["s"], channel_dict["lower"]["v"]])
        upper_hsv = np.array([channel_dict["upper"]["h"], channel_dict["upper"]["s"], channel_dict["upper"]["v"]])
        # Threshold
        hsv_image = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        if channel_dict["kernel"]:
            hsv_image = cv2.morphologyEx(hsv_image, cv2.MORPH_CLOSE, mask_kernel, iterations=1)
        # Smooth the mask
        hsv_image = cv2.GaussianBlur(hsv_image, (3, 3), sigmaX=0, sigmaY=0)
        _, mask = cv2.threshold(hsv_image, 200, 255, cv2.THRESH_BINARY)
        cv2.resizeWindow("HSV", h, w)
        cv2.imshow("HSV", hsv_image)

        # Mask the OG image
        img_masked = cv2.bitwise_and(img, img, mask=mask)
        # Find the specific color
        lower_bgr = np.array([channel_dict["lower"]["b"], channel_dict["lower"]["g"], channel_dict["lower"]["r"]])
        upper_bgr = np.array([channel_dict["upper"]["b"], channel_dict["upper"]["g"], channel_dict["upper"]["r"]])
        green_img = cv2.inRange(img_masked, lower_bgr, upper_bgr)
        if channel_dict["kernel"]:
            green_img = cv2.morphologyEx(green_img, cv2.MORPH_OPEN, (8,8), iterations=1)

        # Smooth the mask
        green_img = cv2.GaussianBlur(green_img, blur_kernel, sigmaX=0, sigmaY=0)
        _, mask = cv2.threshold(green_img, 175, 255, cv2.THRESH_BINARY)
        cv2.resizeWindow("Dark Green", h, w)
        cv2.imshow("Dark Green", green_img)

        # Resulting image
        img_masked = cv2.bitwise_and(img, img, mask=mask)
        cv2.resizeWindow("Result", h, w)
        cv2.imshow("Result", img_masked)


        cv2.waitKey(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--load",
        required=False,
        default=None,
        action="store_true"
    )
    args = parser.parse_args()
    main(args)