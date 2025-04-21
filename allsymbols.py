import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

def hex_to_fsk_combined_spectrum(hex_string, symbol_duration=0.1, sampling_rate=44100):
    """
    Modulate a hex string into FSK and plot a combined spectrum for all symbols.
    """
    freq_map = {
        '0': 1000, '1': 1100, '2': 1200, '3': 1300,
        '4': 1400, '5': 1500, '6': 1600, '7': 1700,
        '8': 1800, '9': 1900, 'A': 2000, 'B': 2100,
        'C': 2200, 'D': 2300, 'E': 2400, 'F': 2500
    }
    signal = np.array([])
    combined_spectrum = None

    for char in hex_string:
        frequency = freq_map[char.upper()]
        t = np.linspace(0, symbol_duration, int(sampling_rate * symbol_duration), endpoint=False)
        tone = np.sin(2 * np.pi * frequency * t)
        signal = np.concatenate([signal, tone])
        
        # Compute the FFT spectrum for the current tone
        fft_spectrum = np.fft.fft(tone)
        freqs = np.fft.fftfreq(len(tone), 1 / sampling_rate)
        
        # Accumulate the magnitudes in the combined spectrum
        if combined_spectrum is None:
            combined_spectrum = np.abs(fft_spectrum)
        else:
            combined_spectrum += np.abs(fft_spectrum)

    # Plot the combined frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(freqs[:len(freqs) // 2], combined_spectrum[:len(freqs) // 2])
    plt.title("Combined Frequency Spectrum for Hex String")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    return signal

# Generate FSK signal from a hex string and plot combined spectrum
hex_string = "0123456789ABCDEF"
fsk_signal = hex_to_fsk_combined_spectrum(hex_string)

# Save the modulated signal as a WAV file
write("hex_fsk_output_combined.wav", 44100, (fsk_signal * 32767).astype(np.int16))
