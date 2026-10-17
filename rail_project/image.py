import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

file_path = "dataset/crack/c2.wav"  # make sure this file exists

audio, sr = librosa.load(file_path)

# MFCC
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title("MFCC Features")
plt.savefig("mfcc.png")
plt.show()