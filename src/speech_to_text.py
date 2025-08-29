import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

def convert_audio_to_text(audio_path):
    """
    Converts an audio file to text using Google Web Speech API.
    For longer audio files, it splits the audio into chunks.
    """
    r = sr.Recognizer()
    full_text = ""

    try:
        # Load the audio file
        audio = AudioSegment.from_file(audio_path)

        # Split audio into chunks where silence is detected
        # Adjust these parameters based on your audio characteristics
        chunks = split_on_silence(audio,
                                  min_silence_len=500,  # milliseconds of silence
                                  silence_thresh=-40,   # dBFS below reference
                                  keep_silence=200      # milliseconds of silence to keep at the end of chunk
                                 )

        if not chunks:
            return "Error: No speech detected or audio too short."

        for i, chunk in enumerate(chunks):
            # Export chunk to a temporary WAV file
            chunk_filename = f"temp_chunk_{i}.wav"
            chunk.export(chunk_filename, format="wav")

            with sr.AudioFile(chunk_filename) as source:
                audio_listened = r.record(source)
                try:
                    text = r.recognize_google(audio_listened)
                    full_text += text + " "
                except sr.UnknownValueError:
                    # print(f"Could not understand audio in chunk {i}")
                    pass
                except sr.RequestError as e:
                    return f"Could not request results from Google Speech Recognition service; {e}"
            os.remove(chunk_filename) # Clean up temp file

    except Exception as e:
        return f"Error processing audio file: {e}"

    return full_text.strip()

if __name__ == '__main__':
    # Example usage (requires an audio file named 'test_audio.wav' in the same directory)
    # For testing, you can record a short audio file and save it as test_audio.wav
    # or use a sample audio file.
    # Ensure you have pydub and speech_recognition installed: pip install pydub SpeechRecognition
    # Also, ffmpeg is required for pydub: sudo apt-get install ffmpeg
    
    # Create a dummy audio file for testing if it doesn't exist
    if not os.path.exists("test_audio.wav"):
        print("Please create a 'test_audio.wav' file for testing.")
        print("You can use a tool like Audacity to record and save a short WAV file.")
    else:
        print("Converting audio to text...")
        text = convert_audio_to_text("test_audio.wav")
        print(f"Converted Text: {text}")



