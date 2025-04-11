import sounddevice as sd
import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import scipy.io.wavfile as wavfile
import os
import time

def moving_average(data, window_size):
    if window_size % 2 == 0: #needs to be odd
        window_size += 1
    smoothed_data = np.copy(data)
    half_window = window_size // 2
    for i in range(half_window, len(data) - half_window):
        smoothed_data[i] = np.mean(data[i - half_window:i + half_window + 1])
    # For the boundaries, just use the available neighboring values
    smoothed_data[:half_window] = np.mean(data[:window_size])
    smoothed_data[-half_window:] = np.mean(data[-window_size:])
    return smoothed_data

class AudioController:
    def __init__(self, session_dir: str):
        # Audio settings
        self.sample_rate = 44100
        self.duration = 5
        self.channels = 1
        self.wav_file = 'sine_50Hz_to_400Hz.wav'
        
        # Device settings
        self.input_device_index = 2
        self.output_device_index = 2
        sd.default.device = (self.input_device_index, self.output_device_index)
        sd.default.samplerate = self.sample_rate
        
        # Directory setup
        self.session_dir = session_dir
        self.data_dir = os.path.join(self.session_dir, "wav")
        if not os.path.exists(self.data_dir):
            print(f"Audio data directory does not exist, creating it")
            os.makedirs(self.data_dir)
            
        # Load audio file
        try:
            self.audio_to_play, _ = self._load_wav(self.wav_file)
        except Exception as e:
            print(f"Error loading wav file: {e}")
            self.audio_to_play = None
            
        print("Initializing Audio Controller")
        print(f"Available audio devices: {sd.query_devices()}")
        print(f"Audio data directory: {self.data_dir}")

    def _load_wav(self, filename):
        """Load a WAV file for playback"""
        with wave.open(filename, 'rb') as wf:
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            audio_data = np.frombuffer(wf.readframes(num_frames), dtype=np.int16)
        return audio_data, sample_rate
        
    def capture_audio(self, base_name: str, method: str = "tap") -> str:
        """Capture audio data"""
        if self.audio_to_play is None:
            print("Warning: No audio file loaded for playback.... using 'tap' method")
            method = "tap"
        
        if method == "tap":
            print("Tapping audio")
        elif method == "sweep":
            print("Sweeping audio")
        else:
            print("Invalid method")
            return
        
        output_file = os.path.join(self.data_dir, f"{method}_{base_name}.wav")
        print(f"Audio sample method: {method}")
        print(f"Audio sample file: {output_file}")

        if method == "sweep":
            # Ensure audio length matches the duration by repeating it
            num_repeats = int(np.ceil(self.sample_rate * self.duration / len(self.audio_to_play)))
            audio_to_play = np.tile(self.audio_to_play, num_repeats)

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

        print(f"Min: {recorded_audio.min()}")
        print(f"Max: {recorded_audio.max()}")
        
        # Save raw audio
        wavfile.write(output_file, self.sample_rate, recorded_audio)

        # Process and save FFT
        self._process_and_save_fft(recorded_audio, base_name)
        return output_file

    def _process_and_save_fft(self, recorded_audio, base_name):
        """Process the recorded audio and save FFT plot"""
        # Flatten and compute FFT
        audio_data = recorded_audio.flatten()
        fft_result = fft(audio_data)
        fft_freq = fftfreq(len(fft_result), d=1/self.sample_rate)

        # Get positive frequencies
        positive_freq = fft_freq[:len(fft_freq)//2]
        positive_fft = np.abs(fft_result[:len(fft_result)//2])

        # Find peak frequency
        peak_index = np.argmax(positive_fft)
        peak_frequency = positive_freq[peak_index]
        print(f"Peak Frequency: {peak_frequency} Hz")

        # Generate and save plot
        smoothed_magnitudes = moving_average(positive_fft, window_size=7)
        plt.figure(figsize=(10, 6))
        plt.plot(positive_freq, smoothed_magnitudes)
        plt.title('Smoothed FFT of Recorded Audio')
        plt.xlabel('Frequency (Hz)')
        plt.xlim((50, 800))
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.savefig(f"{self.data_dir}/{base_name}_FFT.png")
        plt.close()

if __name__ == "__main__":
    controller = AudioController("data/test")
    controller.capture_audio("test")
