# DocToSpeech (Coqui) <img src=https://github.com/C0m3b4ck/DocToSpeech-coqui/blob/main/icon.png>
[![MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](LICENSE)
[![Pinokio](https://img.shields.io/badge/Platform-Pinokio-FF6B35.svg?style=for-the-badge&logo=pinokio&logoColor=white)](https://pinokio.computer)

One-click Pinokio launcher for DocToSpeech with Coqui TTS models and StyleTTS 2.

## Included Models

| Model | VRAM | Cloning | Languages | Presets |
|-------|------|---------|-----------|---------|
| XTTS v2 | ~4 GB | Yes | 17 languages | Yes |
| Bark | ~4-12 GB | No | Multi-lingual | Yes |
| VITS | ~2-4 GB | No | English | Yes |
| YourTTS | ~3-5 GB | Yes | Multi-lingual | Yes |
| StyleTTS 2 | ~4-8 GB | Yes | English | No |

All 5 models share a single virtual environment.

## Requirements

- Python 3.11
- 8 GB RAM minimum, 16 GB+ recommended
- 2-6+ GB GPU VRAM depending on model
- Pinokio

### IMPORTANT COMPATIBILITY NOTE - READ BEFORE USE

**Poppler** is required for PDF text extraction on all platforms. If you plan to process PDF files, you must install Poppler separately — it is **not** included with pip dependencies.

| Platform | Install Command |
|----------|-----------------|
| **Linux (Debian/Ubuntu)** | `sudo apt install poppler-utils` |
| **Linux (Fedora/RHEL)** | `sudo dnf install poppler-utils` |
| **Linux (Arch)** | `sudo pacman -S poppler` |
| **macOS** | `brew install poppler` |
| **Windows** | Download from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases), extract, and add the `Library/bin` folder to your system PATH |

After installing, verify with: `pdftotext -v`

Without Poppler, PDF documents will fail to load in the Gradio web UI and CLI.

## Install

1. Open Pinokio
2. Click **Discover**
3. Search for **DocToSpeech (Coqui)**
4. Click **Download** then **Install**

Or manually copy this folder to your Pinokio `api/` directory.

## Usage

After installation, Pinokio will show a **Start** button. Click it to launch the Gradio web UI at `http://localhost:7860`.

## Other Versions

| Launcher | Models | Venv |
|----------|--------|------|
| **DocToSpeech (Coqui)** (this one) | XTTS v2, Bark, VITS, YourTTS, StyleTTS 2 | Shared |
| [DocToSpeech (Chatterbox)](https://github.com/C0m3b4ck/DocToSpeech-Chatterbox) | Resemble Chatterbox | Separate |
| [DocToSpeech (Tortoise)](https://github.com/C0m3b4ck/DocToSpeech-Tortoise) | Tortoise TTS | Separate |
| [DocToSpeech (OpenVoice)](https://github.com/C0m3b4ck/DocToSpeech-OpenVoice) | OpenVoice V2 | Separate |
| [DocToSpeech (CSM)](https://github.com/C0m3b4ck/DocToSpeech-CSM) | Sesame CSM-1B | Separate |

## Full Documentation

For CLI usage, batch processing, Ollama sanitization, and more, see the [main DocToSpeech repository](https://github.com/C0m3b4ck/DocToSpeech).

## Credits

Started on July 12th, 2026 by [C0m3b4ck](https://github.com/C0m3b4ck).

### TTS Engines
- [Coqui TTS](https://github.com/idiap/coqui-ai-TTS) (XTTS v2, Bark, VITS, YourTTS) -- [CPML License](https://coqui.ai/cpml)
- [StyleTTS 2](https://github.com/yl4579/StyleTTS2) by Yuanhao Yi -- [MIT License](https://github.com/yl4579/StyleTTS2/blob/main/LICENSE)

### Document Processing
- [epub2txt](https://github.com/aaronsw/html2text)
- [pdftotext](https://github.com/jalan/pdftotext)
- [docx2txt](https://github.com/ankushshah893/docx2txt)
- [html2text](https://github.com/Alir3z4/html2text)

### Other
- [pyfiglet](https://github.com/pwaller/pyfiglet) -- ASCII art
- [tqdm](https://github.com/tqdm/tqdm) -- Progress bars
- [Ollama](https://ollama.com) -- Local LLM for text sanitization

MPL 2.0 License.
