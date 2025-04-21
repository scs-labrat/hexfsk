import numpy as np
from scipy.io.wavfile import write, read
import matplotlib.pyplot as plt

# Frequency maps for Sender A and Sender B
FREQ_MAP_A = {
    1000: '0', 1100: '1', 1200: '2', 1300: '3',
    1400: '4', 1500: '5', 1600: '6', 1700: '7',
    1800: '8', 1900: '9', 2000: 'A', 2100: 'B',
    2200: 'C', 2300: 'D', 2400: 'E', 2500: 'F'
}
FREQ_MAP_B = {
    1050: '0', 1150: '1', 1250: '2', 1350: '3',
    1450: '4', 1550: '5', 1650: '6', 1750: '7',
    1850: '8', 1950: '9', 2050: 'A', 2150: 'B',
    2250: 'C', 2350: 'D', 2450: 'E', 2550: 'F'
}

# Packet structure specs
PREAMBLE = "AA AA AA"  # 3 bytes
END_BIT = "0F"          # 1 byte

# Utility functions for hex and plaintext conversion
def plaintext_to_hex(plaintext):
    """Convert plaintext to hexadecimal string."""
    return plaintext.encode('utf-8').hex().upper()

def hex_to_plaintext(hex_string):
    """Convert hexadecimal string back to plaintext."""
    return bytes.fromhex(hex_string).decode('utf-8')

def calculate_checksum(payload_bytes):
    """Calculate checksum as the sum of payload bytes modulo 256."""
    return sum(payload_bytes) % 256

# Handshake mechanism
def handshake(sender, receiver, initial_tone=1000, offset=50, symbol_duration=0.1, sampling_rate=44100):
    """Perform a handshake between sender and receiver."""
    def generate_tone(frequency):
        t = np.linspace(0, symbol_duration, int(sampling_rate * symbol_duration), endpoint=False)
        return np.sin(2 * np.pi * frequency * t)

    # Sender A sends initial handshake tone
    handshake_tone = generate_tone(initial_tone)
    print(f"Sender (A) sends handshake tone: {initial_tone} Hz")

    # Receiver B responds with acknowledgment and proposed tone
    receiver_ack_tone = generate_tone(initial_tone)
    receiver_proposed_tone = generate_tone(initial_tone + offset)
    combined_receiver_tone = np.concatenate([receiver_ack_tone, receiver_proposed_tone])
    print(f"Receiver (B) responds with: {initial_tone} Hz and {initial_tone + offset} Hz")

    # Sender A confirms by sending the proposed tone
    sender_confirmation_tone = generate_tone(initial_tone + offset)
    print(f"Sender (A) confirms with: {initial_tone + offset} Hz")

    return handshake_tone, combined_receiver_tone, sender_confirmation_tone

# Encoding with a frequency map
def encode_with_frequency_map(hex_string, freq_map, symbol_duration=0.1, sampling_rate=44100):
    """Encode a hex string using a specific frequency map."""
    signal = np.array([])
    for char in hex_string:
        frequency = list(freq_map.keys())[list(freq_map.values()).index(char)]
        t = np.linspace(0, symbol_duration, int(sampling_rate * symbol_duration), endpoint=False)
        tone = np.sin(2 * np.pi * frequency * t)
        signal = np.concatenate([signal, tone])
    return signal

# Decoding with a frequency map
def decode_with_frequency_map(signal, freq_map, symbol_duration=0.1, sampling_rate=44100):
    """Decode a signal using a specific frequency map."""
    n_samples = int(sampling_rate * symbol_duration)
    reversed_freq_map = {float(k): v for k, v in freq_map.items()}  # Ensure frequency keys are float
    decoded_hex = ""
    for i in range(0, len(signal), n_samples):
        chunk = signal[i:i + n_samples]
        fft_spectrum = np.fft.fft(chunk)
        freqs = np.fft.fftfreq(len(chunk), 1 / sampling_rate)
        dominant_freq = freqs[np.argmax(np.abs(fft_spectrum[:len(freqs) // 2]))]
        closest_freq = min(reversed_freq_map.keys(), key=lambda f: abs(f - dominant_freq))
        decoded_hex += reversed_freq_map[closest_freq]
    return decoded_hex

# Full packet encoding
def encode_packet(plaintext, freq_map, symbol_duration=0.1, sampling_rate=44100):
    """Encode plaintext into a full FSK packet."""
    hex_payload = plaintext_to_hex(plaintext)
    payload_bytes = bytes.fromhex(hex_payload)
    checksum = calculate_checksum(payload_bytes)

    # Construct packet
    packet_hex = f"{PREAMBLE.replace(' ', '')}{hex_payload}{checksum:02X}{END_BIT}"
    print(f"Constructed Packet (Hex): {packet_hex}")

    # Encode packet as FSK signal
    return encode_with_frequency_map(packet_hex, freq_map, symbol_duration, sampling_rate)

# Full packet decoding
def decode_packet(signal, freq_map, symbol_duration=0.1, sampling_rate=44100):
    """Decode an FSK signal into plaintext by interpreting the packet."""
    decoded_hex = decode_with_frequency_map(signal, freq_map, symbol_duration, sampling_rate)

    # Extract packet components
    preamble = decoded_hex[:6]
    payload = decoded_hex[6:-4]
    received_checksum = int(decoded_hex[-4:-2], 16)
    end_bit = decoded_hex[-2:]

    # Verify preamble and end bit
    if preamble != PREAMBLE.replace(' ', '') or end_bit != END_BIT:
        raise ValueError("Invalid packet structure.")

    # Verify checksum
    payload_bytes = bytes.fromhex(payload)
    calculated_checksum = calculate_checksum(payload_bytes)
    if received_checksum != calculated_checksum:
        raise ValueError("Checksum mismatch: Packet corrupted.")

    # Convert payload to plaintext
    plaintext = hex_to_plaintext(payload)
    print(f"Decoded Packet: Payload Hex -> {payload}, Plaintext -> {plaintext}")
    return plaintext

# Example usage
if __name__ == "__main__":
    # Perform handshake
    handshake_tone, combined_receiver_tone, sender_confirmation_tone = handshake("A", "B")

    # Encode plaintext to FSK packet (Sender A)
    plaintext_message = "Hello, this is a test of the HEXFSK communication protocol"
    fsk_signal_a = encode_packet(plaintext_message, FREQ_MAP_A)

    # Plot frequency over time for the transmission
    sampling_rate = 44100
    symbol_duration = 0.1
    t = np.linspace(0, len(fsk_signal_a) / sampling_rate, len(fsk_signal_a), endpoint=False)
    plt.figure(figsize=(10, 6))
    plt.specgram(fsk_signal_a, Fs=sampling_rate, NFFT=1024, noverlap=512, cmap='viridis')
    plt.title("Frequency over Time for FSK Transmission")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")
    plt.show()

    # Save the modulated signal to a WAV file
    write("fsk_packet_output_a.wav", 44100, (fsk_signal_a * 32767).astype(np.int16))

    # Decode FSK packet (Receiver B)
    sampling_rate, received_signal_a = read("fsk_packet_output_a.wav")
    decoded_message_b = decode_packet(received_signal_a, FREQ_MAP_A)
    print(f"Decoded Message (Receiver B): {decoded_message_b}")
