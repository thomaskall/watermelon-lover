import cv2
import numpy as np
from keyboard import read_key
from time import *
from glob import glob as glob # glob
import os
from pprint import pp
import json
from threading import Thread
import pdb

cwd = os.getcwd()

screenshot = False

def key_thread(len_imgs):
    global screenshot, pic_idx, channels_dict
    name_dict = {
        "r": "red",
        "g": "green",
        "b": "blue",

        "h": "hue",
        "s": "saturation",
        "v": "value",

        "k": "kernel"
    }
    channel = "s"
    level = "lower"
    print("Starting on the lower threshold for saturation")

    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'

    last = None

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
                print(f"\n{level} {name_dict[key]} selected, currently at {channels_dict[level][key]}")
                sleep(0.5)

            # kernel
            case "k":
                channel = key
                print("\nKernel selected")
                sleep(0.5)

            # Control over channel values
            case "up":
                if channel != "k":
                    if level == "upper":
                        if channel == "h":
                            max_val = 179
                        else: max_val = 255
                    else:
                        max_val = channels_dict["upper"][channel] - 2

                if last == "up" or last == "down":
                    print(LINE_UP, end=LINE_CLEAR)

                if channel == "k":
                    channels_dict["kernel"] += 1
                    print(f"({channels_dict['kernel']}, {channels_dict['kernel']})")
                    sleep(0.5)
                elif channels_dict[level][channel] == max_val:
                    print(f"Max value for {name_dict[channel]} ({max_val}) already reached")
                else:
                    channels_dict[level][channel] += 1
                    print(f"{channels_dict[level][channel]}")

            case "down":
                if channel != "k":
                    if level == "lower":
                        min_val = 0
                    else:
                        min_val = channels_dict["lower"][channel] + 1

                if last == "up" or last == "down":
                    print(LINE_UP, end=LINE_CLEAR)

                if channel == "k":
                    if channels_dict["kernel"] == 0:
                        print(f"Can't go any lower than (0, 0)")
                    else:
                        channels_dict["kernel"] -= 1
                        print(f"({channels_dict['kernel']}, {channels_dict['kernel']})")
                        sleep(0.5)
                elif channels_dict[level][channel] == min_val:
                    print(f"Min value for {name_dict[channel]} ({min_val}) already reached")
                else:
                    channels_dict[level][channel] -= 1
                    print(f"{channels_dict[level][channel]}")

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
                    val = input(f"\nEnter a value to assign to {level} {name_dict[channel]} from 0-{max_val}, currently {channels_dict[level][channel]}: \n" + LINE_CLEAR)
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
                                channels_dict['kernel'] = val
                            else:
                                channels_dict[level][channel] = val
                            valid = True
                            print("Value updated")
                            pp(channels_dict)
                            sleep(1)

                    except Exception as e:
                        print(f"An error occurred:\n\t{e}")
                        val = input("\nTry again...\n" + LINE_CLEAR)
            

            # Print what we've got so far
            case "enter":
                print()
                pp(channels_dict)
                with open("./checkpoint.json", 'w') as json_file:
                    json.dump(channels_dict, json_file, indent=4) 
                screenshot = True
                sleep(1)

            # quit the program
            case "q":
                exit()

            case _:
                print(f'"{key}" not recognized')
                print('If you are trying to exit the program, press "q"')
                sleep(1)

        last = key



def main():
    global screenshot, pic_idx, cwd, channels_dict

    with open("./checkpoint.json", 'r') as json_file:
        channels_dict = json.load(json_file)
    with open("./edge_checkpoint.json", 'r') as json_file:
        edges_dict = json.load(json_file)

    imgs = glob(os.path.join(cwd, "img", "*"))
    pic_idx = 3

    thread = Thread(target=key_thread, args=(len(imgs),))
    thread.start()

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Binary Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Masked", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)

    while thread.is_alive():
        img = cv2.imread(imgs[pic_idx])
        w, h, _ = img.shape
        height_scale = w/h
        h = 500
        w = int(h * height_scale)
        cv2.resizeWindow("Original", h, w)
        cv2.imshow("Original", img)

        kernel = np.ones((channels_dict["kernel"], channels_dict["kernel"]), np.uint8)
        
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([channels_dict["lower"]["h"], channels_dict["lower"]["s"], channels_dict["lower"]["v"]])
        upper_hsv = np.array([channels_dict["upper"]["h"], channels_dict["upper"]["s"], channels_dict["upper"]["v"]])
        hsv_image = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        if channels_dict["kernel"]:
            hsv_image = cv2.morphologyEx(hsv_image, cv2.MORPH_CLOSE, kernel, iterations=1)

        kernel = (edges_dict["kernel"], edges_dict["kernel"])
        hsv_image = cv2.GaussianBlur(hsv_image, kernel, sigmaX=0, sigmaY=0)
        _, mask = cv2.threshold(hsv_image, 200, 255, cv2.THRESH_BINARY)
        cv2.resizeWindow("Binary Mask", h, w)
        cv2.imshow("Binary Mask", mask)

        img_masked = cv2.bitwise_and(img, img, mask=mask)
        img_blur = cv2.GaussianBlur(img_masked, kernel, sigmaX=0, sigmaY=0)
        cv2.resizeWindow("Masked", h, w)
        cv2.imshow("Masked", img_blur)
        
        canny = cv2.Canny(image=img_blur, threshold1=edges_dict["lower"], threshold2=edges_dict["upper"])
        cv2.resizeWindow("Canny", h, w)
        cv2.imshow("Canny", canny)

        if screenshot:
            cv2.imwrite(f"img_masked{pic_idx}.png", img_masked)
            cv2.imwrite(f"edges{pic_idx}.png", canny)
            screenshot = False

        cv2.waitKey(1)

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()