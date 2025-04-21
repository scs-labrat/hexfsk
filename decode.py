import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def fsk_to_hex_spectrum(signal, sampling_rate=44100, symbol_duration=0.1):
    """
    Decode an FSK signal back into a hex string using the audio spectrum.
    """
    freq_map = {
        1000: '0', 1100: '1', 1200: '2', 1300: '3',
        1400: '4', 1500: '5', 1600: '6', 1700: '7',
        1800: '8', 1900: '9', 2000: 'A', 2100: 'B',
        2200: 'C', 2300: 'D', 2400: 'E', 2500: 'F'
    }
    n_samples = int(sampling_rate * symbol_duration)
    hex_string = ""
    
    for i in range(0, len(signal), n_samples):
        chunk = signal[i:i + n_samples]
        
        # Compute FFT to find dominant frequency
        fft_spectrum = np.fft.fft(chunk)
        freqs = np.fft.fftfreq(len(chunk), 1 / sampling_rate)
        dominant_freq = freqs[np.argmax(np.abs(fft_spectrum[:len(freqs) // 2]))]
        
        # Map frequency to hex character
        closest_freq = min(freq_map.keys(), key=lambda f: abs(f - dominant_freq))
        hex_char = freq_map[closest_freq]
        hex_string += hex_char
        
        # Visualise spectrum for the chunk
        plt.figure(figsize=(8, 4))
        plt.plot(freqs[:len(freqs) // 2], np.abs(fft_spectrum)[:len(freqs) // 2])
        plt.title(f"Detected Frequency: {dominant_freq:.2f} Hz (Mapped to '{hex_char}')")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()
    
    return hex_string

# Example: Decode the saved signal
# Load the FSK signal (you can replace this with an actual recording)
from scipy.io.wavfile import read
sampling_rate, received_signal = read("hex_fsk_output.wav")

# Decode the received signal
decoded_hex = fsk_to_hex_spectrum(received_signal)
print(f"Decoded Hex: {decoded_hex}")