import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
import numpy as np

# Suppress Dtype warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# Load dataset and clean
df_raw = pd.read_csv("Epileptic Seizure Recognition.csv", header=None)
df_no_header = df_raw.drop(index=0)
numeric_data = df_no_header.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
numeric_data.dropna(inplace=True)

X = numeric_data.iloc[:, 0:178].values
y = numeric_data.iloc[:, 178].values

# Plot setup
fig, ax = plt.subplots()
x = list(range(178))
line, = ax.plot(x, X[0], color='orange')
label = ax.text(0.02, 0.95, "", transform=ax.transAxes)

ax.set_ylim([X.min(), X.max()])
ax.set_xlim([0, 177])
ax.set_title("EEG Signal Animation")
ax.set_xlabel("Time Sample")
ax.set_ylabel("Amplitude")

# Animate using a loop
for i in range(200):
    idx = i % len(X)
    line.set_ydata(X[idx])
    label.set_text(f"Label: {int(y[idx])}")
    fig.canvas.draw()
    plt.pause(0.2)

plt.show()


# Pick one sample (you can loop through more later)
sample_idx = 0
eeg_signal = X[sample_idx]

# Sampling rate (approximate)
fs = 178  # samples per second

# Apply FFT
frequencies = np.fft.rfftfreq(len(eeg_signal), d=1/fs)
fft_magnitude = np.abs(np.fft.rfft(eeg_signal))

# Plot power spectrum
plt.figure(figsize=(10, 5))
plt.plot(frequencies, fft_magnitude)
plt.title(f"Frequency Spectrum of EEG Signal (Label: {int(y[sample_idx])})")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 60)  # Focus on 0 to 60 Hz
plt.show()
