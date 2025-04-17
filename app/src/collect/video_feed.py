import cv2
import os
from typing import List
import subprocess
import re
import numpy as np

def get_camera_paths() -> List[str]:
    """Get the paths to the cameras from the libcamera-hello command"""
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

class Camera:
    def __init__(self, id: int, camera_path: str):
        self.id: int = id
        self.camera_path: str = camera_path
        self.pipeline: str = build_pipeline(camera_path)
        self.cap: cv2.VideoCapture = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)

    def capture_frame(self) -> tuple[bool, np.ndarray]:
        """Capture a frame from the camera"""
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.id}")
            return
        return self.cap.read()

    def release(self):
        self.cap.release()
        print(f"Released camera {self.id}")

class CameraController:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        if not os.path.exists(self.results_dir):
            print(f"Images data directory does not exist, creating it")
            os.makedirs(self.results_dir)
            
        self.camera_paths: List[str] = get_camera_paths()        
        if not self.camera_paths or len(self.camera_paths) == 0:
            print("No cameras detected.")
        else:
            print(f"Cameras detected: {self.camera_paths}")
        
        self.cameras = [
            Camera(i, path)
            for i, path in enumerate(self.camera_paths)
        ]

    def save_images(self, watermelon_id: str, dimensions: tuple[int, int]=(640, 480)) -> str | None:
        """Save images from both cameras and return the path of the combined image"""
        try:
            watermelon_data_dir = os.path.join(self.results_dir, watermelon_id)
            if not os.path.exists(watermelon_data_dir):
                print(f"Data directory for {watermelon_id} does not exist, creating it")
                os.makedirs(watermelon_data_dir)

            # Get frames from both cameras
            ret1, frame1 = self.cameras[0].capture_frame()
            ret2, frame2 = self.cameras[1].capture_frame()

            num_frames_captured = 0
            if ret1:
                num_frames_captured += 1
            if ret2:
                num_frames_captured += 1
            
            if num_frames_captured < 1:
                raise ValueError("Failed to capture frames from any camera")
            
            if num_frames_captured == 1:
                frame = frame1 if ret1 else frame2
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, dimensions)
                image_path = os.path.join(watermelon_data_dir, f"{self.cameras[0].id if ret1 else self.cameras[1].id}.png")
                cv2.imwrite(image_path, frame_resized)
                return image_path
            
            # Convert frames to RGB
            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            
            # Resize frames to ensure they have the same height
            height = max(frame1_rgb.shape[0], frame2_rgb.shape[0])
            if frame1_rgb is not None:
                frame1_resized = cv2.resize(frame1_rgb, (int(frame1_rgb.shape[1] * height/frame1_rgb.shape[0]), height))
            if frame2_rgb is not None:
                frame2_resized = cv2.resize(frame2_rgb, (int(frame2_rgb.shape[1] * height/frame2_rgb.shape[0]), height))
            
            # Concatenate frames horizontally
            combined_frame = np.hstack((frame1_resized, frame2_resized))
            combined_frame = cv2.resize(combined_frame, dimensions)
            
            # Save the combined image
            image_path = os.path.join(watermelon_data_dir, f"{self.cameras[0].id}_{self.cameras[1].id}.png")
            cv2.imwrite(image_path, cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
            
            return image_path
        
        except ValueError as e:
            print(f"Error saving images: {e}")
            return None
            
        except Exception as e:
            print(f"Error saving images: {e}")
            return None

    def release(self):
        """Clean up resources"""
        print("Releasing Camera Controller")
        print("Stopping camera processes...")
        for camera in self.cameras:
            camera.release()
        print("All cameras have been released")
        

def video_feed():
    camera_paths: List[str] = get_camera_paths()
    print(f"Cameras detected: {camera_paths}")
    if camera_paths:
        show_camera_feeds(camera_paths)
    else:
        print("No cameras detected.")


if __name__ == "__main__":
    video_feed()