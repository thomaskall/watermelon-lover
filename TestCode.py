# Audio Code for RPi 5 USB Audio

# Instructions to set up audio output device via USB Adapter:
    # 1.) To check if audio adapter is detected run: 
        # aplay -l

    # 2.) Should list "card 1" as USB device, edit ASLA config file to set the device to default with: 
        # sudo nano /etc/asound.conf

        # pcm.!default {
        #     type hw
        #     card 1
        # }

        # ctl.!default {
        #     type hw
        #     card 1
        # }

    # 3.) Save and exit (Ctrl + X, then Y to confirm changes, and Enter to exit).

    # 4.) Test with this command:
        # speaker-test -t wav -c 2

    # 5.) Use command "alsamixer" with L/R arrow keys to adjust volume


# Instructions to set up audio Input Device:
    # 1.) To check if audio adapter is detected run: 
        # arecord -l 

    # 2.) See Step 2-5 above
    
    # 3.) Test record and and play recording with:
        # arecord test.wav 
            # (press Ctrl+c to stop recording)
        # aplay test.wav


import sounddevice as sd
import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import scipy.io.wavfile as wavfile
from scipy.io.wavfile import write


print(sd.query_devices()) #use this to list audio devices if needed

# Sampling rate (Hz), choose a rate compatible with your device (usually 44100 or 16000 Hz)


SAMPLE_RATE = 48000  # Your audio device's sample rate
DURATION = 5  # Duration of the recording in seconds
WAV_FILE = 'sine_50Hz_to_1000Hz.wav'  # Input .wav file for playback
OUTPUT_FILE = 'recorded_audio_output.wav'  # Output .wav file to save the recorded audio
CHANNELS = 1  # Mono audio input/output

input_device_index = 0  # Replace with the index of your input device
output_device_index = 1  # Replace with the index of your output device

# Setting the input and output devices for recording and playback
sd.default.device = (input_device_index, output_device_index)
sd.default.samplerate = SAMPLE_RATE

# Load the .wav file for playback
def load_wav(filename):
    with wave.open(filename, 'rb') as wf:
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        audio_data = np.frombuffer(wf.readframes(num_frames), dtype=np.int16)
    return audio_data, sample_rate

# Load the sine wave sweep for playback
audio_to_play, play_sample_rate = load_wav(WAV_FILE)

# Ensure audio length matches the duration by repeating it
num_repeats = int(np.ceil(SAMPLE_RATE * DURATION / len(audio_to_play)))
audio_to_play = np.tile(audio_to_play, num_repeats)

# Make sure the data fits the duration (adjust if needed)
audio_to_play = audio_to_play[:SAMPLE_RATE * DURATION]

# Play and record simultaneously using playrec
print("Playing and recording simultaneously...")
recorded_audio = sd.playrec(audio_to_play, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', blocking=True)

print(f"Min: {recorded_audio.min()}")
print(f"Max: {recorded_audio.max()}")
np.savetxt('array', recorded_audio, delimiter=',')

wavfile.write('output.wav',SAMPLE_RATE,recorded_audio)
#myrecording = sd.rec(int(DURATION * SAMPLE_RATE), channels=2, samplerate=SAMPLE_RATE)

# obj = wave.open('sound.wav','wb')
# obj.setnchannels(1)
# obj.setsampwidth(2)
# obj.setframerate(SAMPLE_RATE)
# #obj.writeframes(myrecording)
# obj.writeframes(recorded_audio)
# sd.wait()
# obj.close()


# print("Playing recording...")
# sd.play(myrecording, SAMPLE_RATE)
# sd.wait()

# Save the recorded audio to a .wav file
# write(OUTPUT_FILE, SAMPLE_RATE, recorded_audio)
print(f"Recorded audio saved to: {OUTPUT_FILE}")
# Perform FFT on the recorded signal
# Flatten the recorded audio to 1D if necessary (it's mono in this case, so it's already 1D)
audio_data = recorded_audio.flatten()

# Compute the FFT (Fast Fourier Transform)
fft_result = fft(audio_data)
fft_freq = fftfreq(len(fft_result), d=1/SAMPLE_RATE)

# Only keep the positive frequencies (real part of FFT)
positive_freq = fft_freq[:len(fft_freq)//2]
positive_fft = np.abs(fft_result[:len(fft_result)//2])

# Find the peak frequency in the FFT result
peak_index = np.argmax(positive_fft)  # Find index of the maximum value in the FFT
peak_frequency = positive_freq[peak_index]  # Get the corresponding frequency

# Print the peak frequency
print(f"Peak Frequency: {peak_frequency} Hz")

# Plot the FFT of the recorded signal
plt.figure(figsize=(10, 6))
plt.plot(positive_freq, positive_fft)
plt.title('FFT of Recorded Audio')
plt.xlabel('Frequency (Hz)')
plt.xlim((70, 1000))
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()