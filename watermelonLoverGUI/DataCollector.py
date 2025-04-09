# Modified from original data collection code for automating test cycles.

import numpy as np
import os
import time
import csv
import serial
from weight import weightSensor
from datetime import datetime
from video_feed import CameraController
from audio import AudioController

from argparse import ArgumentParser

# Initialization parameters for weight sensor.
port = '/dev/tty.usbserial-110' # Replace with serial port: ls /dev/tty* | grep usb
baudrate = 9600
timeout = 1

# parser = ArgumentParser()
# parser.add_argument(
#         "-v",
#         "--visual",
#         action="store_true",
#         required=False,
#         help="whether or not to collect visual data"
#         )
# parser.add_argument(
#         "-a",
#         "--audio",
#         action="store_true",
#         required=False,
#         help="whether or not to collect audio data"
#         )
# parser.add_argument(
#         "--audio-method",
#         type=str,
#         choices=["tap", "sweep"],
#         required=False,
#         help="method to use for audio capture. Options: 'tap or 'sweep'"
#         )
# args = parser.parse_args()

def make_timestamp():
    """Make a timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

class DataCollector:
    #Serialization format
    custom_format = "%Y-%m-%d_%H-%M-%S_%f"

    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.metadata_file = os.path.join(base_dir, "metadata.csv")
        self.watermelon_id = self._get_watermelon_id()

        self.session_dir = self._create_session_dir()
        self.session_id = os.path.basename(self.session_dir)
        
        # Initialize controllers
        self.camera_controller = CameraController(self.session_dir)
        self.audio_controller = AudioController(self.session_dir)

        # Initialize weight sensors
        self.sensor = weightSensor(port=port, baudrate=baudrate, timeout=timeout)
        self.sensor.connect_serial()

        # State management
        self.is_running = True
        
        # Ensure metadata file exists with headers
        self._init_metadata_file()
    
    def _init_metadata_file(self):
        """Initialize metadata CSV file if it doesn't exist"""
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['session_id', 'timestamp', 'watermelon_id', 'size_img', 'size_wav', 'weight'])
    
    def _get_watermelon_id(self) -> str:
        """Generates watermelon ID"""
        #TODO: Come up with final way to serialize watermelon id.

        return datetime.now().strftime(self.custom_format)
    
    def _get_watermelon_weight(self) -> str | None:
        """Queries weight sensors for watermelon weight using Serial"""
        #TODO: Add code to communicate with weight sensors
        return self.sensor.get_data()
        
    
    def _create_session_dir(self):
        """Create a new session directory with timestamp and watermelon ID"""
        session_dir = os.path.join(self.base_dir, f"session_{self.watermelon_id}")
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    
    def _save_metadata(self, weight: str | None):
        """Save session metadata to CSV file"""
        size_img = os.path.getsize(os.path.join(self.session_dir, "img"))
        print(f"Size of img directory: {size_img}")
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
                #TODO: Double check if this is correct, not sure how to determine if file exists (should always exist?)
                size_img if size_img > 0 else None,
                size_wav if size_wav > 0 else None,
                weight
            ])
    
    def _capture_data(self):
        """Capture video or audio data"""
        timestamp = make_timestamp()
        base_name = f"sample_{timestamp}"
        
        try:
            self.camera_controller.take_picture(base_name)
            self.audio_controller.capture_audio(base_name)

            print(f"Sample {base_name} captured")

        except Exception as e:
            print(f"Error capturing data: {e}")
    
    def start(self):
        """Start the data collection system"""
        self._capture_data()

        print("\nStarting data collection system...")
        print(f"Watermelon ID: {self.watermelon_id}")
        print(f"Saving data to: {self.session_dir}")
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up resources...")

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
    