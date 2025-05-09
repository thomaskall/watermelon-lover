import cv2
import os
from typing import List
import subprocess
import re
import numpy as np
from multiprocessing import Process, Queue, Event
import queue
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
        self.id: int = id
        self.camera_path: str = camera_path
        self.pipeline: str = build_pipeline(camera_path)
        
        # Don't create VideoCapture here - create it in the process
        self.frame_queue: Queue = frame_queue
        self.stop_event = stop_event
        self.process: Process = Process(target=self._capture_frames)
        self.process.start()

    def _capture_frames(self):
        """Continuously capture frames in a separate process"""
        # Create VideoCapture object inside the process
        cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.id}")
            return

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if ret:
                    # Add timestamp to track frame freshness
                    self.frame_queue.put((self.id, time.time(), frame))
                else:
                    print(f"Error: Could not read frame from camera {self.id}")
                    break
                time.sleep(0.01)  # Small delay to prevent overwhelming the queue

        except Exception as e:
            print(f"Error in camera {self.id}: {e}")
            
        finally:
            print(f"Releasing camera {self.id}")
            cap.release()  # Release the local cap object

    def release(self):
        print(f"Releasing camera {self.id}")
        self.stop_event.set()
        self.process.join(timeout=2)
        if self.process.is_alive():
            print(f"Camera {self.id} process didn't stop gracefully, terminating...")
            self.process.terminate()
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

        self.command_queue = Queue()
        
        self.frame_queues = [Queue() for _ in self.camera_paths]
        self.stop_event = Event()
        self.cameras = [
            Camera(i, path, queue, self.stop_event) 
            for i, (path, queue) in enumerate(zip(self.camera_paths, self.frame_queues))
        ]
        
        # Dictionary to store the latest frame from each camera
        self.latest_frames = {}
        
        # Video display
        self.display_process = None
        self.display_running = Event()

        # Picture capture
        self.picture_queue = Queue()
        self.take_picture_event = Event()
        self.picture_name = None

    def _process_frames(self):
        """Process all available frames from the queues"""
        for queue in self.frame_queues:
            while not queue.empty():
                camera_id, timestamp, frame = queue.get()
                self.latest_frames[camera_id] = (timestamp, frame)

    def _display_loop(self):
        """Run the display loop in a separate process"""
        self.display_running.set()
        while self.display_running.is_set():
            try:
                self._process_frames()
                
                frames = []
                current_time = time.time()
                for camera_id in range(len(self.cameras)):
                    if camera_id in self.latest_frames:
                        timestamp, frame = self.latest_frames[camera_id]
                        if current_time - timestamp < 1.0:
                            frames.append(frame)

                if frames:
                    combined = cv2.hconcat(frames)

                    picture_name: str | None = None
                    try:
                        picture_name = self.picture_queue.get_nowait()
                    except queue.Empty:
                        pass
                    
                    # Check if we need to take a picture
                    if self.take_picture_event.is_set() and picture_name is not None:
                        self._save_current_frames(picture_name, self.latest_frames)
                        self.take_picture_event.clear()

                        # Create white flash frame of the same size
                        flash_frame = np.full_like(combined, 255, dtype=np.uint8)
                        cv2.imshow("Camera Feeds", flash_frame)
                        cv2.waitKey(50)  # Show flash for 50ms

                    cv2.imshow("Camera Feeds", combined)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"{e}")
                break
        print("Exiting display loop")
        cv2.destroyAllWindows()

    def _save_current_frames(self, name, latest_frames):
        """Save the current frames when triggered"""
        print(f"Name for the pictures: {name}")
        if not name:
            print("No picture name provided")
            return
            
        current_time = time.time()
        for camera_id in latest_frames:
            timestamp, frame = latest_frames[camera_id]  # Unpack only timestamp and frame
            if current_time - timestamp < 1.0:
                cv2.imwrite(f"{self.data_dir}/{name}_{camera_id}.jpg", frame)
            else:
                print(f"Warning: Stale frame for camera {camera_id}")

    def start_display(self):
        """Start displaying camera feeds in a separate process"""
        if self.display_process is None or not self.display_process.is_alive():
            self.display_process = Process(target=self._display_loop)
            self.display_process.start()
        else:
            print("Display already running")

    def take_picture(self, name: str):
        """Signal the display process to take a picture"""
        if not self.display_process or not self.display_process.is_alive():
            print("Error: Display not running")
            return
            
        self.picture_queue.put(name)
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
        print("Releasing Camera Controller")
        
        # First stop all cameras to prevent new frames
        self.stop_event.set()
        print("Stopping camera processes...")
        for camera in self.cameras:
            camera.release()
        print("All cameras have been released")

        # Clear all queues to prevent blocking
        print("Clearing frame queues...")
        for queue in self.frame_queues:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except queue.Empty:
                    break
        
        # Now stop the display process
        print("Stopping display process...")
        self.display_running.clear()
        if self.display_process and self.display_process.is_alive():
            self.display_process.join(timeout=2)
            if self.display_process.is_alive():
                print("Display process didn't stop gracefully, terminating...")
                self.display_process.terminate()
                self.display_process.join()

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