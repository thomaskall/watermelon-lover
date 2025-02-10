import cv2
import glob
import os
from typing import List
import subprocess
import re

def get_camera_paths() -> List[str]:
    cmd = "libcamera-hello --list-cameras"
    output = subprocess.check_output(cmd, shell=True).decode()
    
    # Extract camera paths using regex
    camera_paths = [path.strip(")") for path in re.findall(r"(/base/axi/[^\s]+)", output)]
    
    return camera_paths

def build_pipeline(camera_path, width=640, height=480, fps=30):
    return (
        f"libcamerasrc camera-name=\"{camera_path}\" ! \
        video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! \
        videoconvert ! appsink"
    )



def show_camera_feeds(cameras: List[str]):
    """Display video feeds for all detected cameras."""
    if len(cameras) >= 2:
        pipeline_1 = build_pipeline(cameras[0])
        pipeline_2 = build_pipeline(cameras[1])

        cap1 = cv2.VideoCapture(pipeline_1, cv2.CAP_GSTREAMER)
        cap2 = cv2.VideoCapture(pipeline_2, cv2.CAP_GSTREAMER)

        if not cap1.isOpened() or not cap2.isOpened():
            print("Error: Could not open one or both cameras.")
            exit()

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                print("Error: Could not read frames.")
                break

            # Show both camera feeds side by side
            combined = cv2.hconcat([frame1, frame2])
            cv2.imshow("Dual Camera Feed", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
    else:
        print("Not enough cameras detected.")

if __name__ == "__main__":
    # Example usage
    cameras: List[str] = get_camera_paths()
    print("Detected Cameras:", cameras)
    
    if cameras:
        show_camera_feeds(cameras)
    else:
        print("No cameras detected.")
