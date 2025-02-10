import cv2
import glob
import os
from typing import List

def get_camera_device_paths():
    # Find all media devices in /dev that are related to cameras
    media_devices = glob.glob('/dev/media*')

    # Filter out devices that are not actually camera-related
    camera_paths = [dev for dev in media_devices if os.path.exists(dev)]
    
    return camera_paths


def show_camera_feeds(cameras: List[str]):
    """Display video feeds for all detected cameras."""
    caps = {cam: cv2.VideoCapture(cam) for cam in cameras}

    while True:
        for cam, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Camera {cam}", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    camera_paths = get_camera_device_paths()
    if camera_paths:
        print(f"Available camera devices: {camera_paths}")
    else:
        print("No camera devices found.")
    
    if camera_paths:
        show_camera_feeds(camera_paths)
    else:
        print("No cameras detected.")
