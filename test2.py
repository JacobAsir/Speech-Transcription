import pyaudio
import wave
import os
from faster_whisper import WhisperModel

# Constants
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1  # Mono audio
RATE = 16000  # Sampling rate (16kHz)
CHUNK = 1024  # Number of frames per buffer
SILENCE_THRESHOLD = 100  # Threshold to detect silence (adjust as needed)
MODEL_SIZE = "medium.en"  # Use the medium model for better accuracy
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Initialize Whisper model
print("Loading Whisper model...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")  # Use CPU
print("Model loaded.")

def record_chunk(p, stream, file_path, chunk_length=3):
    """
    Record an audio chunk and save it to a file.
    """
    frames = []
    for _ in range(0, int(RATE / CHUNK * chunk_length)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    # Save the audio chunk to a file
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def transcribe_chunk(model, file_path):
    """
    Transcribe an audio chunk using the Whisper model.
    """
    segments, _ = model.transcribe(file_path, beam_size=5)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def main():
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    accumulated_transcription = ""  # Initialize an empty string to accumulate transcriptions

    print("Speak into your microphone. Press Ctrl+C to stop.")

    try:
        while True:
            # Record a chunk of audio
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file)

            # Transcribe the audio chunk
            transcription = transcribe_chunk(model, chunk_file)

            # Print the transcription in real-time
            if transcription.strip():  # Only print non-empty transcriptions
                print(NEON_GREEN + transcription + RESET_COLOR)

            # Append the new transcription to the accumulated transcription
            accumulated_transcription += transcription + " "

            # Remove the temporary audio file
            os.remove(chunk_file)

    except KeyboardInterrupt:
        print("Stopping...")
        # Write the accumulated transcription to the log file
        with open("log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)
    finally:
        # Clean up
        print("LOG:" + accumulated_transcription)
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()