import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Optional: suppress dtype warning
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# Load the EEG dataset (replace path if needed)
df_raw = pd.read_csv("Epileptic Seizure Recognition.csv", header=None)

# Drop header row and convert data columns to numeric
df_no_header = df_raw.drop(index=0)
numeric_data = df_no_header.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
numeric_data.dropna(inplace=True)

# Extract EEG signal (X) and labels (y)
X = numeric_data.iloc[:, 0:178].values
y = numeric_data.iloc[:, 178].values

# Pick one sample to analyze
sample_idx = 0
eeg_signal = X[sample_idx]
label_val = int(y[sample_idx])

# Sampling rate
fs = 178  # Hz

# FFT
frequencies = np.fft.rfftfreq(len(eeg_signal), d=1/fs)
fft_magnitude = np.abs(np.fft.rfft(eeg_signal))

# Define EEG frequency bands
bands = {
    'Delta (0.5-4 Hz) Deep sleep': (0.5, 4),
    'Theta (4-8 Hz) Drowsiness': (4, 8),
    'Alpha (8-13 Hz) Calm, restful': (8, 13),
    'Beta (13-30 Hz) Focus, alertness': (13, 30),
    'Gamma (30-100 Hz) High cognition,learning': (30, 60)  # cap at 60 Hz for display
}
colors = ['lightblue', 'lightgreen', 'khaki', 'salmon', 'plum']

# Map EEG class labels to description
label_map = {
    1: "Seizure",
    2: "Tumor region",
    3: "Healthy region (tumor patient)",
    4: "Eyes closed",
    5: "Eyes open"
}

# Plot
plt.figure(figsize=(12, 6))
plt.plot(frequencies, fft_magnitude, label='EEG Spectrum', color='black')

# Highlight EEG bands
for (band_name, (low, high)), color in zip(bands.items(), colors):
    plt.axvspan(low, high, color=color, alpha=0.4, label=band_name)

plt.title(f"EEG Frequency Spectrum (Label {label_val}: {label_map.get(label_val, 'Unknown')})")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 60)
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()



