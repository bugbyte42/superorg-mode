# Chaterelle 🍄🎤
---
Simple speech-to-text tool for transforming spoken observations and tasks into rich transcripts - *without rotting the details.* The goal is to snappily generate raw transcripts that are then refined to help with recognizing key nouns, custom names, scientific terms (e.g., plant/animal species), and domain-specific jargon.

---

## Prerequisites

* **Python 3.9+**
* **Git**
* **pipx** (recommended) or **pip** for installing the `uv` environment tool
* *(Optional)* **NVIDIA GPU** with up-to-date drivers for GPU acceleration

---

## 1. Clone the Repository

```bash
git clone https://github.com/bugbyte42/superorg-mode.git
cd superorg-mode/spore/chaterelle
```

---

## 2. Environment Setup

We use Astral’s [uv](https://github.com/astral-sh/uv) to manage isolated Python environments. You can substitute `venv`, `virtualenv`, or `conda` if you prefer.

1. **Install `uv`** (via `pipx` or `pip`):

   ```bash
   pipx install uv
   # or
   pip install --user uv
   ```

2. **Initialize a new venv** in the project directory:

   ```bash
   uv init venv
   ```

3. **Activate & sync dependencies**:

   ```bash
   uv shell      # Spawns a shell with the venv activated
   uv sync       # Installs everything in uv.lock
   ```

   > You can also run commands prefixed with `uv`, e.g., `uv pip install <package>`, `uv run ruff check --fix`, etc.

---

## 3. Install FFmpeg

FFmpeg is required for audio decoding/processing (used by Whisper, Pydub, etc.).

* **macOS (Homebrew):**

  ```bash
  brew install ffmpeg
  ```

* **Ubuntu/Debian:**

  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```

* **Windows:**

  1. Download a static build from [ffmpeg.org](https://ffmpeg.org/download.html)
  2. Extract and add the `bin` folder (e.g., `C:\ffmpeg\bin`) to your `PATH`.

* **Verify:**

  ```bash
  ffmpeg -version
  ```

---

## 4. Install CUDA Toolkit (Optional)

For GPU acceleration with **faster-whisper** and **CTranslate2**:

1. Download & install **CUDA Toolkit 12.x** from NVIDIA:
   [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)

   * Ensure CUDA’s `bin/` (and on Linux, `lib64/`) paths are in your `PATH`/`LD_LIBRARY_PATH`.
2. Verify driver & GPU status:

   ```bash
   nvidia-smi
   ```
3. Install PyTorch with CUDA support (example for CUDA 12.8):

   ```bash
   uv pip install torch torchvision torchaudio \
     --index-url https://download.pytorch.org/whl/cu128
   ```
4. Test CUDA in Python:

   ```python
   import torch
   print("CUDA available:", torch.cuda.is_available())
   torch.zeros(1).cuda()
   ```

> **Note:** Installing PyTorch with CUDA 12.x wheels before other dependencies avoids version conflicts. See [UV issue #7202](https://github.com/astral-sh/uv/issues/7202).

---

---

## 5. Install Project Dependencies

From the project root (where `pyproject.toml` lives):

```bash
uv pip install -e .
```

This installs Chaterelle in “editable” mode, so any local changes take effect immediately.

---

## 6. Running Transcriptions

```bash
uv run -m chaterelle.rhizomorphs.transcribe \
  data/HARVARD_raw/Harvard\ list\ 01.wav \
  --model faster-whisper
```

---

## Troubleshooting & Tips

* **CPU-only usage:** Skip CUDA steps; faster-whisper will run on CPU.
* **Custom models:** Place model files under `stt_models/` and reference with `--model` (look at model loading in rhizomorphs.transcribe.py to ensure compatibility).
* **Contributions welcome!** Open issues or PRs for bugs and feature requests.
