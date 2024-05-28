import numpy as np
from scipy.signal import convolve, correlate

def wiener_filter(signal, noise, filter_length):
    # Estimate the power spectral density (PSD) of the signal and noise
    signal_psd = np.abs(np.fft.fft(signal)) ** 2
    noise_psd = np.abs(np.fft.fft(noise)) ** 2

    # Estimate the cross-power spectral density (CPSD) between signal and noise
    cross_psd = np.fft.fftshift(correlate(signal, noise, mode='full'))

    # Wiener filter frequency response
    wiener_filter_freq = np.conj(noise_psd) / (signal_psd + noise_psd)

    # Apply Wiener filter in the frequency domain
    filtered_signal_freq = np.fft.fft(signal) * wiener_filter_freq

    # Inverse FFT to obtain filtered signal in time domain
    filtered_signal = np.real(np.fft.ifft(filtered_signal_freq))

    # Truncate to the original length
    filtered_signal = filtered_signal[:len(signal)]

    return filtered_signal

# Example usage
np.random.seed(42)

# Generate a random signal and add noise
N = 256
signal = np.sin(2 * np.pi * 0.1 * np.arange(N))  # Example sinusoidal signal
noise = 0.5 * np.random.randn(N)  # Example Gaussian noise
noisy_signal = signal + noise

# Apply Wiener filter
filter_length = 50  # Adjust as needed
filtered_signal = wiener_filter(noisy_signal, noise, filter_length)

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(signal, label='Original Signal', color='blue')
plt.plot(noisy_signal, label='Noisy Signal', alpha=0.7, color='red')
plt.plot(filtered_signal, label='Wiener Filtered Signal', linestyle='--', color='green')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Wiener Filter Example')
plt.show()
