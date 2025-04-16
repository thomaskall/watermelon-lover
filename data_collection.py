import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime
import pyautogui
from video_feed import CameraController
from audio import AudioController

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
        "-v",
        "--visual",
        action="store_true",
        required=False,
        help="whether or not to collect visual data"
        )
parser.add_argument(
        "-a",
        "--audio",
        action="store_true",
        required=False,
        help="whether or not to collect audio data"
        )
parser.add_argument(
        "--audio-method",
        type=str,
        choices=["tap", "sweep"],
        required=False,
        help="method to use for audio capture. Options: 'tap or 'sweep'"
        )
args = parser.parse_args()
def make_timestamp():
    """Make a timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

class DataCollector:
    def __init__(self, base_dir="data"):
        """Initialize the data collection system"""
        if args.visual and args.audio:
            raise ValueError("Cannot collect both visual and audio data at the same time")
        elif not args.visual and not args.audio:
            raise ValueError("Must collect either visual or audio data. Use flag -v (visual) or flag -a (audio) to specify data type.")
        self.base_dir = base_dir
        self.metadata_file = os.path.join(base_dir, "metadata.csv")
        self.gui_title = "Data Collection"
        
        # Get watermelon ID before starting session
        self.watermelon_id = self._get_watermelon_id()
        if not self.watermelon_id:  # User cancelled
            raise ValueError(f"Watermelon ID is required... check out {self.metadata_file} for past sessions and naming conventions")
            
        self.session_dir = self._create_session_dir()
        self.session_id = os.path.basename(self.session_dir)
        
        # Initialize controllers
        if args.visual:
            self.camera_controller = CameraController(self.session_dir)
        elif args.audio:
            self.audio_controller = AudioController(self.session_dir)

        # State management
        self.is_running = True
        
        # Ensure metadata file exists with headers
        self._init_metadata_file()
        self.samples_captured: int = 0
    
    def _init_metadata_file(self):
        """Initialize metadata CSV file if it doesn't exist"""
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['session_id', 'timestamp', 'watermelon_id', 'size_img', 'size_wav', 'weight'])
    
    def _get_watermelon_id(self) -> str:
        """Prompt user for watermelon ID"""
        return pyautogui.prompt(
            text='Enter the watermelon ID:',
            title=self.gui_title,
            default=None
        )
    
    def _get_watermelon_weight(self) -> str | None:
        """Prompt user for watermelon weight at end of session"""
        return pyautogui.prompt(
            text='Enter weight for this watermelon (kg):',
            title=self.gui_title,
            default=None
        )
    
    def _create_session_dir(self):
        """Create a new session directory with timestamp and watermelon ID"""
        session_dir = os.path.join(self.base_dir, f"session_{self.watermelon_id}")
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    
    def _save_metadata(self, weight: str | None):
        """Save session metadata to CSV file"""
        if args.visual:
            size_img = os.path.getsize(os.path.join(self.session_dir, "img"))
            print(f"Size of img directory: {size_img}")
        if args.audio:
            size_wav = os.path.getsize(os.path.join(self.session_dir, "wav")) // 2
            print(f"Size of wav directory: {size_wav}")
        if weight is not None:
            print(f"Weight: {weight}")

        with open(self.metadata_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.session_id,
                make_timestamp(),
                self.watermelon_id,
                size_img if args.visual else None,
                size_wav if args.audio else None,
                weight
            ])
    
    def _keyboard_listener(self):
        """Listen for keyboard commands using simple input"""
        time.sleep(5)
        while True:
            try:
                command = input("\nENTER to capture sample, type 'q' and ENTER to quit...\n").lower().strip()

                if command == "":  # Enter key pressed
                    print("Capturing data...")
                    self._capture_data()
                elif command == "q":
                    print("Quitting...")
                    return
                else:
                    print(f'Command "{command}" not recognized')
                    print('Available commands: [Enter], q')
                
            except Exception as e:
                print(f"Input error: {e}")
                return
    
    def _capture_data(self):
        """Capture video or audio data"""
        timestamp = make_timestamp()
        base_name = f"sample_{timestamp}"
        
        try:
            # Capture camera data
            if args.visual:
                self.camera_controller.take_picture(base_name)
            
            elif args.audio:
                self.audio_controller.capture_audio(base_name)
            
            print(f"Sample {base_name} captured")
            self.samples_captured += 1
            print(f"Total samples captured: {self.samples_captured}")

        except Exception as e:
            print(f"Error capturing data: {e}")
    
    def start(self):
        """Start the data collection system"""
        # Start the camera display
        if args.visual:
            self.camera_controller.start_display()
        print("\nStarting data collection system...")
        print(f"Watermelon ID: {self.watermelon_id}")
        print(f"Saving data to: {self.session_dir}")
        
        # Start keyboard listener
        self._keyboard_listener()
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up resources...")

        if args.visual:
            self.camera_controller.release()
        
        # Get watermelon score before cleanup
        weight: str | None = self._get_watermelon_weight()
        self._save_metadata(weight)
        
        print(f"Collection complete.")
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
    
