import cv2

def list_cameras(max_cameras=10):
    """Detect all available cameras by attempting to open them."""
    cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    return cameras

def show_camera_feeds(cameras):
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
    detected_cameras = list_cameras()
    
    if detected_cameras:
        print(f"Detected Cameras: {detected_cameras}")
        show_camera_feeds(detected_cameras)
    else:
        print("No cameras detected.")
