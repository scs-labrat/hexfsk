import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

def hex_to_fsk_spectrum(hex_string, symbol_duration=0.1, sampling_rate=44100):
    """
    Modulate a hex string into FSK and visualise the spectrum.
    """
    freq_map = {
        '0': 1000, '1': 1100, '2': 1200, '3': 1300,
        '4': 1400, '5': 1500, '6': 1600, '7': 1700,
        '8': 1800, '9': 1900, 'A': 2000, 'B': 2100,
        'C': 2200, 'D': 2300, 'E': 2400, 'F': 2500
    }
    signal = np.array([])
    
    for char in hex_string:
        frequency = freq_map[char.upper()]
        t = np.linspace(0, symbol_duration, int(sampling_rate * symbol_duration), endpoint=False)
        tone = np.sin(2 * np.pi * frequency * t)
        signal = np.concatenate([signal, tone])
        
        # Plot the frequency spectrum for the current tone
        fft_spectrum = np.fft.fft(tone)
        freqs = np.fft.fftfreq(len(tone), 1 / sampling_rate)
        plt.figure(figsize=(8, 4))
        plt.plot(freqs[:len(freqs) // 2], np.abs(fft_spectrum)[:len(freqs) // 2])
        plt.title(f"Frequency Spectrum for '{char}' ({frequency} Hz)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()
    
    return signal

# Generate FSK signal from a hex string
hex_string = "1A3F"
fsk_signal = hex_to_fsk_spectrum(hex_string)

# Save the modulated signal as a WAV file
write("hex_fsk_output.wav", 44100, (fsk_signal * 32767).astype(np.int16))


