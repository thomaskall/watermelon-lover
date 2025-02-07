import time
import sys
import threading
from picamera2 import Picamera2, Preview
import numpy as np
import cv2

frame = None

def show_frame():
    global frame
    while True:
        if frame is not None:
            print("Showing frame")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Camera", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def show_camera_feed(camera_id, picam2):
    global frame
    picam2.start()
    print("Picam started")
    i = 0
    while True:
        frame = picam2.capture_array()

    picam2.stop()
    cv2.destroyAllWindows()

def main():
    #show_camera_feed(0, Picamera2())
    cam_ids = [0]

    picam2_instances = {}
    for cam_id in cam_ids:
        picam2_instances[cam_id] = Picamera2()

    threads = []
    for cam_id, picam2 in picam2_instances.items():
        thread = threading.Thread(target=show_camera_feed, args=(cam_id, picam2))
        threads.append(thread)
        thread.start()
    show_frame_thread = threading.Thread(target=show_frame, args=())
    threads.append(show_frame_thread)
    show_frame_thread.start()
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
