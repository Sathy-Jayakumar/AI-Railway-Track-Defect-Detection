import os
import time
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Function to extract MFCC features from audio file
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Dataset folder
base_path = "dataset"

# Categories
categories = ["normal", "crack", "fracture", "wear"]
X = []
y = []

# Read training dataset
for category in categories:
    folder_path = os.path.join(base_path, category)

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)

            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(category)
            except Exception as e:
                print("Error reading:", file_path)
                print(e)

# Convert to arrays
X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model trained successfully.")
print("Waiting for new audio files...\n")

# Folder where new test files will be added
test_folder = "new_test"

# Stores already checked files
checked_files = set()

# Continuous detection loop
while True:
    for file in os.listdir(test_folder):
        if file.endswith(".wav") and file not in checked_files:

            file_path = os.path.join(test_folder, file)

            try:
                features = extract_features(file_path).reshape(1, -1)
                prediction = model.predict(features)

                print(file, "→", prediction[0])

                checked_files.add(file)

            except Exception as e:
                print("Error in file:", file)
                print(e)

    # Wait 2 seconds before checking again
    time.sleep(2)