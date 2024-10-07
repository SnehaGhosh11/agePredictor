import sounddevice as sd
import wavio
import librosa
import numpy as np
import pandas as pd

# Parameters for recording
samplerate = 22050  # Sample rate in Hz
duration = 5  # Duration of the recording in seconds
channels = 1  # Mono recording

# Step 1: Record the audio
print("Recording... Speak into the microphone.")
audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='float32')
sd.wait()  # Wait until the recording is finished
print("Recording finished.")

# Step 2: Save the recording as a WAV file
wavio.write("recorded_audio.wav", audio_data, samplerate, sampwidth=2)
print("Audio file saved as 'recorded_audio.wav'.")

# Step 3: Load the saved audio and extract features using librosa
audio_path = "recorded_audio.wav"
y, sr = librosa.load(audio_path, sr=samplerate)

# Step 4: Extract features as per the dataset

# Extract Chroma STFT
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
chroma_stft_mean = np.mean(chroma_stft)

# Extract RMSE (Root Mean Square Energy)
rmse = librosa.feature.rms(y=y)
rmse_mean = np.mean(rmse)

# Extract Spectral Centroid
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
spectral_centroid_mean = np.mean(spectral_centroid)

# Extract Spectral Bandwidth
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
spectral_bandwidth_mean = np.mean(spectral_bandwidth)

# Extract Spectral Roll-off
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
rolloff_mean = np.mean(rolloff)

# Extract Zero Crossing Rate
zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
zero_crossing_rate_mean = np.mean(zero_crossing_rate)

# Extract MFCC (Mel-frequency cepstral coefficients)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
mfcc_mean = np.mean(mfcc, axis=1)

# Step 5: Prepare the data for saving
feature_names = [
    'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 
    'rolloff', 'zero_crossing_rate'
] + [
    'mfcc_' + str(i) for i in range(1, 21)
]

feature_values = np.concatenate((
    [chroma_stft_mean, rmse_mean, spectral_centroid_mean, spectral_bandwidth_mean, 
     rolloff_mean, zero_crossing_rate_mean],
    mfcc_mean
))

# Step 6: Save features in a CSV file
features_df = pd.DataFrame([feature_values], columns=feature_names)
features_df.to_csv('extracted_audio_features.csv', index=False)

print("Extracted features saved in 'extracted_audio_features.csv'.")
