import librosa
import librosa.display
import matplotlib.pyplot as plt

file_path = "dataset/crack/c1.wav"  # change file if needed

audio, sr = librosa.load(file_path)

plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio, sr=sr)
plt.title("Waveform of Audio Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()