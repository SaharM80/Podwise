import os
import uuid
from pathlib import Path
from dotenv import load_dotenv

import torch
import torchaudio

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message


class HiggsAudioTTS:
    """
    Locally-run Higgs Audio v2 TTS using BosonAI's HiggsAudioServeEngine.
    """

    def __init__(
        self,
        model_path="bosonai/higgs-audio-v2-generation-3B-base",
        tokenizer_path="bosonai/higgs-audio-v2-tokenizer",
        output_dir="output",
        default_temperature=0.3,
        default_top_p=0.95,
        default_max_new_tokens=1024
    ):
        # Load .env (optional for other configs)
        load_dotenv()

        # Prepare output folder
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Defaults
        self.default_params = dict(
            temperature=default_temperature,
            top_p=default_top_p,
            max_new_tokens=default_max_new_tokens
        )

        # Detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model locally (positional args only)
        self.serve_engine = HiggsAudioServeEngine(model_path, tokenizer_path, self.device)

        # System prompt for better generation quality
        self.system_prompt = (
            "Generate audio following instruction.\n\n"
            "<|scene_desc_start|>\n"
            "Audio is recorded from a quiet room.\n"
            "<|scene_desc_end|>"
        )

    def speak(self, text, **kwargs):
        """
        Generate speech from text and save it to a WAV file.
        Returns path to the saved file.
        """
        params = {**self.default_params, **kwargs}

        # Prepare messages for the model
        messages = [
            Message("system", self.system_prompt),
            Message("user", text),
        ]

        # Run local inference
        output = self.serve_engine.generate(
            ChatMLSample(messages),
            params["max_new_tokens"],
            params["temperature"],
            params["top_p"],
            50,  # top_k
            ["<|end_of_text|>", "<|eot_id|>"]  # stop_strings
        )

        # Save audio
        file_path = self.output_dir / f"{uuid.uuid4().hex}.wav"
        torchaudio.save(str(file_path), torch.from_numpy(output.audio)[None, :], output.sampling_rate)

        return str(file_path)
