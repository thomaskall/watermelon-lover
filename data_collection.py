import cv2
import numpy as np
import os
import time
import csv
from threading import Thread
from datetime import datetime
from queue import Queue
from keyboard import read_key
import pyautogui
from video_feed import CameraController

def make_timestamp():
    """Make a timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

class DataCollector:
    def __init__(self, base_dir="data"):
        """Initialize the data collection system"""
        self.base_dir = base_dir
        self.metadata_file = os.path.join(base_dir, "metadata.csv")
        self.gui_title = "Data Collection"
        
        # Get watermelon ID before starting session
        self.watermelon_id = self._get_watermelon_id()
        if not self.watermelon_id:  # User cancelled
            raise ValueError("Watermelon ID is required")
            
        self.session_dir = self._create_session_dir()
        self.session_id = os.path.basename(self.session_dir)
        
        # Initialize controllers
        self.camera_controller = CameraController(self.session_dir)
        # self.audio_controller = AudioController(self.session_dir)
        
        # State management
        self.is_running = True
        self.collection_count = 0
        
        # Command queue for thread safety
        self.command_queue = Queue()
        
        # Ensure metadata file exists with headers
        self._init_metadata_file()
    
    def _init_metadata_file(self):
        """Initialize metadata CSV file if it doesn't exist"""
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['session_id', 'timestamp', 'watermelon_id', 'size_img', 'size_wav', 'score'])
    
    def _get_watermelon_id(self) -> str:
        """Prompt user for watermelon ID"""
        return pyautogui.prompt(
            text='Enter the watermelon ID:',
            title=self.gui_title,
            default=None
        )
    
    def _get_watermelon_score(self) -> str | None:
        """Prompt user for watermelon score at end of session"""
        return pyautogui.prompt(
            text='Enter score for this watermelon (Brix Scale):',
            title=self.gui_title,
            default=None
        )
    
    def _create_session_dir(self):
        """Create a new session directory with timestamp and watermelon ID"""
        session_dir = os.path.join(self.base_dir, f"session_{self.watermelon_id}")
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    
    def _save_metadata(self, score: str | None):
        """Save session metadata to CSV file"""
        print(f"Session directory: {self.session_dir}")
        size_img = os.path.getsize(os.path.join(self.session_dir, "img"))
        print(f"Size of img directory: {size_img}")
        size_wav = os.path.getsize(os.path.join(self.session_dir, "wav"))
        print(f"Size of wav directory: {size_wav}")

        with open(self.metadata_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.session_id,
                make_timestamp(),
                self.watermelon_id,
                size_img,
                size_wav,
                score
            ])
    
    def _keyboard_listener(self):
        """Listen for keyboard commands in a separate thread"""
        while self.is_running:
            try:
                # Space bar to capture data
                key = read_key()
                match key:
                    case "space":
                        print("Capturing data...")
                        self.command_queue.put('capture')
                    case "a":
                        print("Capturing audio...")
                        self.command_queue.put('audio')
                    case "v":
                        print("Capturing video...")
                        self.command_queue.put('video')
                    # 'q' to quit
                    case "q" | "esc" | "ctrl+c":
                        print("Quitting...")
                        self.command_queue.put('quit')
                        break

                    case _:
                        print(f'"{key}" not recognized')
                        print('If you are trying to exit the program, press "q"')
                        time.sleep(1)
                
                # Add more keyboard commands as needed
                time.sleep(0.01)  # Prevent high CPU usage
                
            except Exception as e:
                print(f"Keyboard listener error: {e}")
                break
    
    def _process_commands(self):
        """Process commands from the queue"""
        while not self.command_queue.empty():
            command = self.command_queue.get()
            match command:
                case 'capture':
                    self._capture_data()
                case 'audio':
                    self._capture_audio()
                case 'video':
                    self._capture_video()
                case 'quit':
                    self.stop()
    
    def _capture_data(self):
        """Capture both video and audio data"""
        timestamp = make_timestamp()
        base_name = f"sample_{self.collection_count:04d}_{timestamp}"
        
        try:
            # Capture camera data
            self.camera_controller.take_picture(base_name)
            
            # Capture audio data (TODO)
            # self.audio_controller.capture_audio(base_name)
            
            self.collection_count += 1
            print(f"Captured sample {self.collection_count}")
            
        except Exception as e:
            print(f"Error capturing data: {e}")

    def _capture_audio(self):
        """Capture audio data"""
        timestamp = make_timestamp()
        base_name = f"sample_{self.collection_count:04d}_{timestamp}"
        self.audio_controller.capture_audio(base_name)
    
    def _capture_video(self):
        """Capture video data"""
        timestamp = make_timestamp()
        base_name = f"sample_{self.collection_count:04d}_{timestamp}"
        self.camera_controller.take_picture(base_name)
    
    def start(self):
        """Start the data collection system"""
        print("\nStarting data collection system...")
        print(f"Watermelon ID: {self.watermelon_id}")
        print(f"Saving data to: {self.session_dir}")
        print("\nControls:")
        print("  SPACE - Capture data")
        print("  A     - Capture audio")
        print("  V     - Capture video")
        print("  Q     - Quit")
        print("\nWaiting for commands...")
        
        # Start the camera display
        self.camera_controller.start_display()
        
        # Start keyboard listener
        self.keyboard_thread = Thread(target=self._keyboard_listener)
        self.keyboard_thread.start()
        
        # Main loop
        try:
            while self.is_running:
                self._process_commands()
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            self.stop()
            
        finally:
            self.cleanup()
    
    def stop(self):
        """Stop the data collection system"""
        self.is_running = False
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up resources...")

        self.camera_controller.release()
        # self.audio_controller.release()  # TODO

        if self.keyboard_thread.is_alive():
            self.keyboard_thread.join()
        
        # Get watermelon score before cleanup
        score: str | None = self._get_watermelon_score()
        self._save_metadata(score)
        print(f"Saved score {score} for watermelon {self.watermelon_id}")
        
        print(f"Collection complete. {self.collection_count} samples captured.")
        print(f"Data saved in: {self.session_dir}")

def main():
    try:
        collector = DataCollector()
        collector.start()
    except ValueError as e:
        print(f"Error: {e}")
        print("Exiting program...")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Exiting program...")

if __name__ == "__main__":
    main()
    