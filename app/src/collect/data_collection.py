from __future__ import annotations
import numpy as np
import os
from typing import Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated

from .video_feed import CameraController
from .audio import AudioController
from .weight import WeightSensor

# Initialization parameters for weight sensor.
port = '/dev/ttyUSB0' # Replace with serial port: ls /dev/tty* | grep usb

def make_timestamp() -> str:
    """Make a timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Datatype to hold data passed down the pipeline.
class WatermelonData(BaseModel):
    id:                 Annotated[str, Field(default_factory=make_timestamp)]
    cycle_type:         Annotated[Literal["sweep", "tap", "impulse"], Field(discriminator="cycle_type")]
    image_path:         Annotated[str | None, Field(default=None)]
    wav_path:           Annotated[str | None, Field(default=None)]
    spectrogram_path:   Annotated[str | None, Field(default=None)]
    weight:             Annotated[float | None, Field(default=None)]
    brix_prediction:    Annotated[float | None, Field(default=None)]

    @field_validator("weight")
    def validate_weight(cls, weight) -> float | None:
        if weight is not None:
            weight = float(weight)
            if weight <= 0.0:
                raise ValueError("Weight must be greater than 0.0")
            return weight
        return None
    def is_complete(self) -> bool:
        """Check if all required data has been collected"""
        return all([
            self.wav_path is not None,
            self.spectrogram_path is not None,
            self.weight is not None,
            self.brix_prediction is not None
        ])
    def to_dict(self) -> dict[str, str | float | None]:
        return self.model_dump()
    def to_json(self) -> str:
        return self.model_dump_json()

class DataCollector:
    def __init__(self, results_dir_base: str = "data"):
        self.results_dir: str = self._create_results_dir(results_dir_base)
        
        # Initialize peripherals
        self.camera_controller = CameraController(self.results_dir)
        self.audio_controller = AudioController(self.results_dir)
        self.weight_sensor = WeightSensor(port=port, baudrate=9600, timeout=3)
        self.weight_sensor.connect_serial()
    
    def _create_results_dir(self, results_dir_base: str) -> str:
        """Create a new results directory with timestamp and watermelon ID"""
        results_dir = os.path.join(os.getcwd(), results_dir_base)
        os.makedirs(results_dir, exist_ok=True)
        return results_dir
    
    def capture_data(self, data: WatermelonData) -> WatermelonData | None:
        """Capture audio and weight data"""
        try:
            data.wav_path = self.audio_controller.capture_audio(data.id, data.cycle_type)
            data.weight = float(self.weight_sensor.get_data())
            print(f"Sample {data.id} captured")
            
        except Exception as e:
            print(f"Error capturing data: {e}")
        return data
    
    def get_image_path(self, cycle_type: Literal["sweep", "tap"], dimensions: tuple[int, int]) -> str | None:
        """Get the image path"""
        data: WatermelonData = WatermelonData(
            cycle_type=cycle_type,
            image_path=self.camera_controller.save_images(cycle_type, dimensions)
        )
        return data
    
    def cleanup(self):
        """Clean up resources"""
        self.camera_controller.release()
        self.weight_sensor.close()

def main():
    try:
        collector = DataCollector()
        collector.capture_data()
    except ValueError as e:
        print(f"Error: {e}")
        print("Exiting program...")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Exiting program...")

if __name__ == "__main__":
    main()
    
