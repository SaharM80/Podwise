import os
import wave
from uuid import uuid4
from typing import List, Union
from dotenv import load_dotenv
from elevenlabs import ElevenLabs

# Load environment variables
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Create ElevenLabs client
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


def chunk_text(text: str, max_chars: int = 2500) -> List[str]:
    """Splits text into chunks under max_chars, breaking at word boundaries."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_chars:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def tts(text: str, merge_output: bool = True) -> Union[str, List[str]]:
    """
    Converts text to speech using ElevenLabs with PCM output (WAV-compatible).

    Args:
        text: Text to convert.
        merge_output: Whether to merge all chunk files into one WAV.

    Returns:
        - If merge_output=True: path to merged WAV
        - If merge_output=False: list of individual WAV file paths
    """
    os.makedirs("output", exist_ok=True)
    chunks = chunk_text(text)
    filepaths = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"Generating audio for chunk {i}/{len(chunks)} ({len(chunk)} chars)")
        generator = client.text_to_speech.convert(
            voice_id="nPczCjzI2devNBz1zQrb",
            text=chunk,
            output_format="pcm_22050",  # PCM format
            model_id="eleven_multilingual_v2",
        )

        filepath = f"output/{uuid4().hex}.wav"
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)         # mono
            wf.setsampwidth(2)         # 16-bit PCM
            wf.setframerate(22050)     # match PCM output
            wf.writeframes(b"".join(generator))

        filepaths.append(filepath)

    # Merge WAV files if requested
    if merge_output and len(filepaths) > 1:
        merged_path = f"output/{uuid4().hex}_merged.wav"
        with wave.open(merged_path, "wb") as wf_out:
            # Set same params as first chunk
            with wave.open(filepaths[0], "rb") as wf_first:
                wf_out.setparams(wf_first.getparams())

            # Append all frames
            for fp in filepaths:
                with wave.open(fp, "rb") as wf_in:
                    wf_out.writeframes(wf_in.readframes(wf_in.getnframes()))

        print(f"Merged audio saved to: {merged_path}")
        return merged_path

    return filepaths if not merge_output else filepaths[0]



# def tts(text: str) -> str:
#     """Converts a given text to realistic speech and returns the saved audio."""
#     generator = client.text_to_speech.convert(
#         voice_id="nPczCjzI2devNBz1zQrb",
#         output_format="mp3_44100_128",
#         text=text,
#         model_id="eleven_multilingual_v2",
#     )
#     filepath = f"output/{uuid4().hex}.mp3"
#     with open(filepath, 'wb') as f:
#             for chunk in generator:
#                 f.write(chunk)
#     return filepath


# def tts(text: str) -> str:
#     """
#     Converts given text to speech using HiggsAudioTTS,
#     returns the path to the saved audio file.
#     """
#     filepath = client.speak(text)
#     return filepath
