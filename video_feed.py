import cv2
import glob
import os
from typing import List, Tuple, Any
import subprocess
import re
import numpy as np
from multiprocessing import Process, Queue, Event
import time

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
    def __init__(self, id: int, camera_path: str, frame_queue: Queue, stop_event):
        self.id = id
        self.camera_path = camera_path
        self.pipeline = build_pipeline(camera_path)
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.process = Process(target=self._capture_frames)
        self.process.start()

    def _capture_frames(self):
        """Continuously capture frames in a separate process"""
        cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.id}")
            return

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                # Add timestamp to track frame freshness
                self.frame_queue.put((self.id, time.time(), frame))
            else:
                print(f"Error: Could not read frame from camera {self.id}")
                break

        cap.release()

    def release(self):
        self.stop_event.set()
        self.process.join()

class CameraController:
    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.data_dir = os.path.join(self.session_dir, "img")
        print("Initializing Video Feed")
        print(f"Images data directory: {self.data_dir}")
        if not os.path.exists(self.data_dir):
            print(f"Images data directory does not exist, creating it")
            os.makedirs(self.data_dir)
            
        self.camera_paths: List[str] = get_camera_paths()
        print(f"Cameras detected: {self.camera_paths}")
        
        if not self.camera_paths:
            print("No cameras detected.")
            exit()
            
        self.frame_queues = [Queue() for _ in self.camera_paths]
        self.stop_event = Event()
        self.cameras = [
            Camera(i, path, queue, self.stop_event) 
            for i, (path, queue) in enumerate(zip(self.camera_paths, self.frame_queues))
        ]
        
        # Dictionary to store the latest frame from each camera
        self.latest_frames = {}
        
        self.display_process = None
        self.take_picture_event = Event()
        self.picture_name = None
        self.display_running = Event()

    def __del__(self):
        self.release()
        cv2.destroyAllWindows()

    def process_frames(self):
        """Process all available frames from the queues"""
        for queue in self.frame_queues:
            while not queue.empty():
                camera_id, timestamp, frame = queue.get()
                self.latest_frames[camera_id] = (timestamp, frame)

    def _display_loop(self):
        """Run the display loop in a separate process"""
        self.display_running.set()
        while self.display_running.is_set():
            self.process_frames()
            
            frames = []
            current_time = time.time()
            for camera_id in range(len(self.cameras)):
                if camera_id in self.latest_frames:
                    timestamp, frame = self.latest_frames[camera_id]
                    if current_time - timestamp < 1.0:
                        frames.append(frame)

            if frames:
                combined = cv2.hconcat(frames)
                
                # Check if we need to take a picture
                if self.take_picture_event.is_set():
                    self._save_current_frames()
                    self.take_picture_event.clear()

                    # Create white flash frame of the same size
                    flash_frame = np.full_like(combined, 255, dtype=np.uint8)
                    cv2.imshow("Camera Feeds", flash_frame)
                    cv2.waitKey(50)  # Show flash for 50ms
                    
                    # Show the actual frame again
                    cv2.imshow("Camera Feeds", combined)
                else:
                    cv2.imshow("Camera Feeds", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def _save_current_frames(self):
        """Save the current frames when triggered"""
        if not self.picture_name:
            print("No picture name provided")
            return
            
        current_time = time.time()
        for camera_id in range(len(self.cameras)):
            if camera_id in self.latest_frames:
                timestamp, frame = self.latest_frames[camera_id]
                if current_time - timestamp < 1.0:
                    cv2.imwrite(f"{self.data_dir}/{self.picture_name}_{camera_id}.jpg", frame)
                else:
                    print(f"Warning: Stale frame for camera {camera_id}")
            else:
                print(f"Error: No frame available for camera {camera_id}")

    def start_display(self):
        """Start displaying camera feeds in a separate process"""
        if self.display_process is None or not self.display_process.is_alive():
            self.display_process = Process(target=self._display_loop)
            self.display_process.start()
        else:
            print("Display already running")

    def stop_display(self):
        """Stop the display process"""
        if self.display_process and self.display_process.is_alive():
            self.display_running.clear()
            self.display_process.join()
            self.display_process = None

    def take_picture(self, name: str):
        """Signal the display process to take a picture"""
        if not self.display_process or not self.display_process.is_alive():
            print("Error: Display not running")
            return
            
        self.picture_name = name
        self.take_picture_event.set()
        
        # Wait briefly to ensure the picture is taken
        timeout = 1.0  # seconds
        start_time = time.time()
        while self.take_picture_event.is_set():
            if time.time() - start_time > timeout:
                print("Warning: Picture taking timed out")
                self.take_picture_event.clear()
                break
            time.sleep(0.01)

    def release(self):
        """Clean up resources"""
        self.stop_display()
        self.stop_event.set()
        for camera in self.cameras:
            camera.release()
        print("Cameras have been released")
        for queue in self.frame_queues:
            while not queue.empty() or queue.qsize() > 0:
                queue.get()
        print("Frame queues have been cleared")
        print("All resources have been released, quitting program")

def video_feed():
    camera_paths: List[str] = get_camera_paths()
    print(f"Cameras detected: {camera_paths}")
    if camera_paths:
        show_camera_feeds(camera_paths)
    else:
        print("No cameras detected.")


if __name__ == "__main__":
    video_feed()