import importlib
from typing import Protocol, Any
import soundfile as sf
from chaterelle.enzymes.path_utils import validate_file_path
from faster_whisper import WhisperModel as FWModel
from faster_whisper.vad import VadOptions
import chaterelle.substrate as _  
import torch

class TranscriptionModel(Protocol):
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio from the given path to text."""
        ...  # Implementation should be provided by the model class


def load_model(model_name: str, *, model_size: str = "base", vad: bool = True, **kwargs) -> TranscriptionModel:
    def get_device():
        check_cuda = torch.cuda.is_available()
        device = "cuda" if check_cuda else "cpu"
        print(f"[INFO] torch.cuda.is_available(): {check_cuda}")
        print(f"[INFO] Using device: {device}")
        if device == "cpu":
            print("[WARNING] CUDA (GPU) is not available. Transcription will be slow.")
        return device

    if model_name in ("whisper", "whisper-turbo"):
        import whisper

        class WhisperModel:
            def __init__(self, model_size="base", **model_kwargs):
                device = get_device()
                self.model = whisper.load_model(model_size, device=device, **model_kwargs)

            def transcribe(self, audio_path: str) -> str:
                result = self.model.transcribe(audio_path)
                text = result["text"]
                if isinstance(text, list):
                    text = " ".join(str(t) for t in text)
                return str(text)

        # For backward compatibility, treat 'whisper-turbo' as 'turbo' model_size
        size = "turbo" if model_name == "whisper-turbo" else model_size
        return WhisperModel(size, **kwargs)
    elif model_name == "faster-whisper":
        class FasterWhisperModel:
            def __init__(self, model_size="base", device=None, compute_type="auto", vad=True, **model_kwargs):
                device = device or get_device()
                self.model = FWModel(model_size, device=device, compute_type=compute_type, **model_kwargs)
                self.vad = vad

            def transcribe(self, audio_path: str) -> str:
                vad_opts = VadOptions() if self.vad else None
                segments, _ = self.model.transcribe(
                    audio_path,
                    vad_filter=self.vad,
                    vad_parameters=vad_opts,
                )
                return " ".join(segment.text for segment in segments)

        return FasterWhisperModel(model_size, vad=vad, **kwargs)
    else:
        module = importlib.import_module(
            f".stt_models.{model_name}", package=__package__
        )
        # can add vad param if compatible with custom model
        return module.Model(model_size=model_size, **kwargs)


def load_audio(audio_path: str) -> Any:
    audio_path = validate_file_path(
        audio_path,
        must_exist=True,
        allowed_extensions=[".wav", ".mp3", ".flac", ".ogg"],
    )
    try:
        audio, sample_rate = sf.read(audio_path)
        return audio, sample_rate
    except Exception as e:
        raise ValueError(f"Error loading audio file {audio_path}: {e}")


def transcribe_audio(model: TranscriptionModel, audio_path: str) -> str:
    audio_path = validate_file_path(
        audio_path,
        must_exist=True,
        allowed_extensions=[".wav", ".mp3", ".flac", ".ogg"],
    )
    return model.transcribe(audio_path)


def main(audio_path: str, model_name: str, *, model_size: str = "base", vad: bool = True, **kwargs) -> str:
    if model_name == "faster-whisper":
        model = load_model(
            model_name,
            model_size=model_size,
            vad=vad,
            **kwargs
        )
    else:
        model = load_model(
            model_name,
            model_size=model_size,
            **kwargs
        )
    transcript = transcribe_audio(model, audio_path)
    print(transcript)
    return transcript


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe audio using a specified model. Model variant/size/type depends on backend."
    )
    parser.add_argument(
        "audio_path", type=str, help="Path to the audio file to transcribe."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="whisper-turbo",
        help="Name of the transcription model to use.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        help="Model size/variant/type (e.g. tiny, base, small, medium, large-v2, turbo, etc.). See model backend docs for valid options."
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD for faster-whisper and compatible models."
    )
    args = parser.parse_args()

    main(
        args.audio_path,
        args.model,
        model_size=args.model_size,
        vad=not args.no_vad
    )
