# Modified from original data collection code for automating test cycles.

import numpy as np
import os
from typing import Literal
from .weight import weightSensor
from datetime import datetime
from .video_feed import CameraController
from .audio import AudioController
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated

# Initialization parameters for weight sensor.
port = '/dev/ttyUSB0' # Replace with serial port: ls /dev/tty* | grep usb

def make_timestamp() -> str:
    """Make a timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Datatype to hold data passed down the pipeline.
class WatermelonData(BaseModel):
    id: str = Field(default_factory=lambda: make_timestamp())
    image_path: str | None = None
    wav_file: str | None = None
    spectrogram: str | None = None
    weight: float | None = None
    cycle_type: Literal["sweep", "tap", "impulse"] | None = None

    @field_validator("weight")
    def validate_weight(cls, weight) -> float | None:
        if weight is not None and weight <= 0.0:
            raise ValueError("Weight must be greater than 0.0")
        return weight

    def is_complete(self) -> bool:
        """Check if all required data has been collected"""
        return all([
            self.image_path is not None,
            self.wav_file is not None,
            self.spectrogram is not None,
            self.weight is not None,
            self.cycle_type is not None
        ])

    def to_dict(self) -> dict[str, str | float | None]:
        return self.model_dump()
    
    def to_json(self) -> str:
        return self.model_dump_json()
    

class DataCollector:
    def __init__(self, results_dir_base: str = "data"):
        self.watermelon_id: str = make_timestamp()
        self.results_dir = self._create_results_dir(results_dir_base)

        # Initialize data object
        self.data = WatermelonData(id=self.watermelon_id)
        
        # Initialize controllers
        self.camera_controller = CameraController(self.results_dir)
        self.audio_controller = AudioController(self.results_dir)

        # Initialize weight sensors
        self.weight_sensor = weightSensor(port=port, baudrate=9600, timeout=3)
        self.weight_sensor.connect_serial()

    def _get_watermelon_weight(self) -> str | None:
        """Queries weight sensors for watermelon weight using Serial"""
        return self.weight_sensor.get_data()
    
    def _create_results_dir(self, base_name: str):
        """Create a new results directory with timestamp and watermelon ID"""
        results_dir = os.path.join(base_name, f"{self.watermelon_id}")
        os.makedirs(results_dir, exist_ok=True)
        return results_dir
    
    def _capture_data(self, cycle_type: Literal["sweep", "tap", "impulse"]) -> WatermelonData | None:
        """Capture video or audio data"""
        timestamp = make_timestamp()
        self.base_name = f"sample_{timestamp}"
        data = None
        
        try:
            #self.camera_controller.take_picture(base_name)
            self.data.wav_file = self.audio_controller.capture_audio(self.base_name, cycle_type)
            data.weightData = self.sensor.get_data()
            print(f"Sample {self.base_name} captured")
            
        except Exception as e:
            print(f"Error capturing data: {e}")
        return data
    
    def start(self) -> WatermelonData:
        """Start the data collection system"""
        data = self._capture_data()

        print("\nStarting data collection system...")
        print(f"Watermelon ID: {self.watermelon_id}")
        print(f"Saving data to: {self.session_dir}")
        
        return data
        
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up resources...")

        #self.camera_controller.release()
        
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
    
