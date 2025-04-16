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

cwd = os.getcwd()

param_dict = {
    "upper": 200,
    "lower": 100,
    "kernel": 1,
    "sobel": 5,
    "dx": 1,
    "dy": 1
}
pic_idx = 0

def keyboard_watcher(len_imgs):
    global param_dict
    global pic_idx
    name_dict = {
        "u": "upper",
        "l": "lower",
        "s": "sobel",
        "k": "kernel",
        "x": "dx",
        "y": "dy"
    }
    last = None
    param = "kernel"

    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    while True:
        key = read_key()
        
        match key:
            case "u" | "l":
                param = name_dict[key]
                print(f"\{param} threshold selected")
                sleep(0.5)

            case "x" | "y":
                param = name_dict[key]
                if param_dict[param]:
                    param_dict[param] = 0
                    print(f"Toggling {param} to OFF")
                else:
                    param_dict[param] = 1
                    print(f"Toggling {param} to ON")
                sleep(0.5)

            # Sobel Kernel
            case "s":
                param = name_dict[key]
                print(f"\nSobel kernel selected, currently at {param_dict[param]}")
                sleep(0.5)

            # Gaussian blur kernel
            case "k":
                param = name_dict[key]
                print("\nGaussian Blurring kernel selected")
                sleep(0.5)

            # Control over channel values
            case "up":
                if param == "lower":
                    max_val = param_dict["upper"] - 1
                else:
                    max_val = 255

                if last == "up" or last == "down":
                    print(LINE_UP, end=LINE_CLEAR)

                if param_dict[param] == max_val:
                    print(f"Max value for {param} ({max_val}) already reached")
                else:
                    if param == "kernel" or param == "sobel":
                        param_dict[param] += 2
                    else:
                        param_dict[param] += 1
                    print(f"{param_dict[param]}")
                    sleep(0.05)

            case "down":
                if param == "upper":
                    min_val = param_dict["lower"] + 1
                else:
                    min_val = 1

                if last == "up" or last == "down":
                    print(LINE_UP, end=LINE_CLEAR)

                if param_dict[param] == min_val:
                    print(f"Min value for {param} ({min_val}) already reached")
                else:
                    if param == "kernel" or param == "sobel":
                        param_dict[param] -= 2
                    else:
                        param_dict[param] -= 1
                    print(f"{param_dict[param]}")
                    sleep(0.05)

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

            # Print what we've got so far
            case "enter":
                print()
                pp(param_dict)
                with open("./edge_checkpoint.json", 'w') as json_file:
                    json.dump(param_dict, json_file, indent=4) 
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
    global cwd, param_dict, pic_idx
    if args.load:
        with open("./edge_checkpoint.json", 'r') as json_file:
            param_dict = json.load(json_file)
    imgs = glob(os.path.join(cwd, "img", "*"))

    key_thread = Thread(target=keyboard_watcher, args=(len(imgs),), daemon=True)
    key_thread.start()

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original", 720, 480)
    cv2.namedWindow("Sobel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sobel", 720, 480)
    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Canny", 720, 480)
    
    while key_thread.is_alive():
        # read image
        img = cv2.imread(imgs[pic_idx]) # BGR
        cv2.imshow("Original", img)

        kernel = (param_dict["kernel"], param_dict["kernel"])
        img_blur = cv2.GaussianBlur(img, kernel, sigmaX=0, sigmaY=0)
        
        canny = cv2.Canny(image=img_blur, threshold1=param_dict["lower"], threshold2=param_dict["upper"])
        cv2.imshow("Canny", canny)

        sobel = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=param_dict["dx"], dy=param_dict['dy'], ksize=param_dict["sobel"])
        cv2.imshow("Sobel", sobel)


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