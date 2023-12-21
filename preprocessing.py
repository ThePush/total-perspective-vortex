import mne
from mne.datasets import eegbci
import matplotlib.pyplot as plt
import numpy as np

# Load data (example: subject 1, motor imagery tasks)
subject = 1
tasks = [6, 10, 14]  # motor imagery: fists vs feet
raw_files = [
    mne.io.read_raw_edf(f, preload=True, stim_channel="auto")
    for f in eegbci.load_data(subject, tasks)
]
raw = mne.concatenate_raws(raw_files)

# Visualize raw data
raw.plot(duration=5, n_channels=30)

# Filter data (e.g., band-pass filter between 7 and 30 Hz)
raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

# Visualize data after filtering
raw.plot(duration=5, n_channels=30)
plt.show()

# Compute PSD using compute_psd with multitaper method
spectrum = raw.compute_psd(
    method="multitaper", tmin=10, tmax=20, fmin=7, fmax=30, picks="eeg"
)
data, freqs = spectrum.get_data(return_freqs=True)

# Plot PSD
fig, ax = plt.subplots()
ax.plot(freqs, 10 * np.log10(data.mean(axis=0)), color="k")
ax.set(
    title="Multitaper PSD (dB)",
    xlabel="Frequency (Hz)",
    ylabel="Power/Frequency (dB/Hz)",
)
plt.savefig("psd.png")
raw.compute_psd().plot()
plt.show()
