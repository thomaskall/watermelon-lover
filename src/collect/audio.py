import sounddevice as sd
import numpy as np
import wave
import scipy.io.wavfile as wavfile
import os
from typing import Literal

class AudioController:
    def __init__(self, results_dir: str):
        # Audio settings
        self.sample_rate: int = 44100
        self.duration: int = 5
        self.channels: int = 1
        self.sweep_path: str = 'sine_50Hz_to_400Hz.wav'
        
        # Device settings
        self.input_device_index = 2
        self.output_device_index = 2
        sd.default.device = (self.input_device_index, self.output_device_index)
        sd.default.samplerate = self.sample_rate
        
        # Directory setup
        self.results_dir = results_dir
        if not os.path.exists(self.results_dir):
            print(f"Audio data directory does not exist, creating it")
            os.makedirs(self.results_dir)
            
        # Load audio file
        try:
            self.sweep_signal, _ = self._load_wav(self.sweep_path)
        except Exception as e:
            print(f"Error loading wav file: {e}")
            self.sweep_signal = None
            
        print("Initializing Audio Controller")
        print(f"Available audio devices:\n{sd.query_devices()}")

    @property
    def audio_duration(self) -> int:
        """Get the duration of the audio"""
        return self.duration

    def _load_wav(self, filename):
        """Load a WAV file for playback"""
        with wave.open(filename, 'rb') as wf:
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            audio_data = np.frombuffer(wf.readframes(num_frames), dtype=np.int16)
        return audio_data, sample_rate
        
    def capture_audio(self, watermelon_id: str, method: Literal["tap", "sweep"]="tap") -> str:
        """Capture audio data"""        
        if method == "tap":
            print("Tapping audio")
        elif method == "sweep":
            print("Sweeping audio")
            if self.sweep_signal is None:
                print("Warning: No audio file loaded for playback.... using 'tap' method")
                method = "tap"

        watermelon_data_dir = os.path.join(self.results_dir, watermelon_id)
        if not os.path.exists(watermelon_data_dir):
            print(f"Data directory for {watermelon_id} does not exist, creating it")
            os.makedirs(watermelon_data_dir)

        output_file = os.path.join(watermelon_data_dir, f"{method}.wav")
        print(f"Audio sample method: {method}")
        print(f"Audio sample file: {output_file}")

        if method == "sweep":
            # Ensure audio length matches the duration by repeating it
            num_repeats = int(np.ceil(self.sample_rate * self.duration / len(self.sweep_signal)))
            audio_to_play = np.tile(self.sweep_signal, num_repeats)

            # Make sure the data fits the duration
            audio_to_play = audio_to_play[:self.sample_rate * self.duration]

            # Play and record simultaneously
            print("Playing sweep and recording audio...")
            recorded_audio = sd.playrec(
                audio_to_play, 
                samplerate=self.sample_rate, 
                channels=self.channels, 
                dtype='int16',
                blocking=True
            )
        else:
            print("Recording taps...")
            recorded_audio = sd.rec(
                self.sample_rate * self.duration,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16',
                blocking=True
            )
        # Save raw audio
        wavfile.write(output_file, self.sample_rate, recorded_audio)
        return output_file


if __name__ == "__main__":
    controller = AudioController("data/test")
    controller.capture_audio("test")
