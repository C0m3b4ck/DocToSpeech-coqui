import os
import sys
import pathlib
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from doctospeech import (
    MODEL_REGISTRY,
    MODEL_KEYS,
    _get_device,
    _get_init_fn,
    _get_speakers,
    make_tts_voiceover,
    _chunk_text,
    load_agent_prompt,
    AGENT_MD_PATH,
    sanitize_text,
)

# ========================= MODEL AVAILABILITY =========================

_MODEL_DEPS = {
    "xtts_v2": ["TTS"],
    "bark": ["TTS"],
    "vits": ["TTS"],
    "yourtts": ["TTS"],
    "chatterbox": ["chatterbox"],
    "tortoise": ["tortoise"],
    "styletts2": ["styletts2"],
    "openvoice": ["openvoice", "melo"],
    "csm": ["transformers"],
}


def _check_imports(modules):
    for mod in modules:
        try:
            __import__(mod)
        except ImportError:
            return False
    return True


def get_available_models():
    available = {}
    for key in MODEL_KEYS:
        deps = _MODEL_DEPS.get(key, [])
        available[key] = _check_imports(deps)
    return available


AVAILABLE_MODELS = get_available_models()


def model_label(key):
    info = MODEL_REGISTRY[key]
    if AVAILABLE_MODELS[key]:
        return f"{info['name']} ({key})"
    return f"{info['name']} ({key}) -- NOT INSTALLED (needs: {info['package']})"


def available_model_keys():
    return [k for k in MODEL_KEYS if AVAILABLE_MODELS[k]]


def unavailable_model_keys():
    return [k for k in MODEL_KEYS if not AVAILABLE_MODELS[k]]


def _first_available_choice():
    """Return the label for the first available model, or first overall."""
    for key in MODEL_KEYS:
        if AVAILABLE_MODELS[key]:
            return model_label(key)
    return model_label(MODEL_KEYS[0]) if MODEL_KEYS else None


# ========================= VOICE MODE PER MODEL =========================
#
# Each model falls into one of three voice modes:
#   "clone_only"   - always needs a .wav reference (chatterbox, tortoise, styletts2, openvoice, csm)
#   "preset_only"  - only has preset speakers, no cloning (bark, vits)
#   "both"         - user can choose cloning or preset (xtts_v2, yourtts)

def _voice_mode(key):
    info = MODEL_REGISTRY.get(key, {})
    has_cloning = info.get("cloning", False)
    # Check if model actually has preset speakers
    has_presets = key in ("xtts_v2", "yourtts", "bark", "vits")
    if has_cloning and has_presets:
        return "both"
    if has_cloning:
        return "clone_only"
    return "preset_only"


def _default_language(key):
    if key == "openvoice":
        return "en"
    if key == "vits":
        return "en"
    if key in ("chatterbox", "tortoise", "styletts2", "csm"):
        return "en"
    return "en"


def _model_has_cloning(key):
    return _voice_mode(key) in ("both", "clone_only")


def _model_has_preset(key):
    return _voice_mode(key) in ("both", "preset_only")


# ========================= NON-INTERACTIVE TEXT EXTRACTION =========================

def extract_text_from_file(file_path):
    ext = pathlib.Path(file_path).suffix.lower()
    if ext == ".txt":
        with open(file_path, "r", errors="replace") as f:
            return f.read()
    elif ext == ".epub":
        return _extract_epub(file_path)
    elif ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext == ".docx":
        return _extract_docx(file_path)
    elif ext in (".html", ".htm"):
        return _extract_html(file_path)
    else:
        with open(file_path, "r", errors="replace") as f:
            return f.read()


def _extract_epub(path):
    from epub2txt import epub2txt
    chapters = epub2txt(path, outputlist=True)
    return "\n\n".join(chapters)


def _extract_pdf(path):
    import pdftotext
    with open(path, "rb") as f:
        pdf = pdftotext.PDF(f)
    pages = []
    for i, page in enumerate(pdf):
        if i > 0:
            pages.append("\n\n")
        pages.append(page)
    return "".join(pages)


def _extract_docx(path):
    import docx2txt
    return docx2txt.process(path)


def _extract_html(path):
    import html2text
    with open(path, "r", errors="replace") as f:
        html = f.read()
    h = html2text.HTML2Text()
    h.ignore_links = True
    return h.handle(html)


# ========================= MODEL LOADING =========================

def load_model(model_key):
    info = MODEL_REGISTRY[model_key]
    device = _get_device()
    init_fn = _get_init_fn(model_key)
    if init_fn is None:
        raise ValueError(f"Model '{model_key}' init function not found")
    start = time.time()
    tts_obj = init_fn(model_key, device)
    elapsed = time.time() - start
    return tts_obj, elapsed


# ========================= HELPER: extract model key from dropdown label =========================

def _key_from_label(model_choice):
    if not model_choice:
        return None
    key = model_choice.split("(")[-1].rstrip(")").strip()
    if key in MODEL_REGISTRY:
        return key
    return model_choice


# ========================= GRADIO CALLBACKS =========================

def on_model_change(model_choice):
    """Update voice controls, info panel, and language when model changes."""
    key = _key_from_label(model_choice)

    if not key or key not in MODEL_REGISTRY:
        # Nothing selected — hide everything
        return (
            gr.update(visible=False),             # voice_type_radio
            gr.update(visible=False),             # voice_ref_upload
            gr.update(visible=False),             # preset_dropdown
            gr.update(value="Select a model to see its details."),  # model_info
            gr.update(value=""),                  # model_warning
            gr.update(value="en"),                # language_input
        )

    info = MODEL_REGISTRY[key]
    available = AVAILABLE_MODELS.get(key, False)
    mode = _voice_mode(key)

    # --- Model info panel ---
    cloning_str = "Yes" if info["cloning"] else "No"
    if mode == "clone_only":
        cloning_str = "Yes (required — no preset speakers)"
    elif mode == "preset_only":
        cloning_str = "No (preset speakers only)"

    info_lines = [
        f"**{info['name']}**",
        f"VRAM: {info['vram']}",
        f"GPU: {info['gpu']}",
        f"Voice Cloning: {cloning_str}",
        f"Languages: {info['languages']}",
        f"Package: `{info['package']}`",
    ]
    info_text = "\n\n".join(info_lines)

    if not available:
        info_text += (
            f"\n\n---\n"
            f"**WARNING: This model is NOT installed in the current environment.**\n"
            f"Install it with a separate venv: `{info['package']}`"
        )

    # --- Voice type radio ---
    if mode == "clone_only":
        voice_radio = gr.update(
            visible=True,
            choices=["Cloning"],
            value="Cloning",
            interactive=True,
        )
    elif mode == "preset_only":
        voice_radio = gr.update(
            visible=True,
            choices=["Preset"],
            value="Preset",
            interactive=True,
        )
    else:
        voice_radio = gr.update(
            visible=True,
            choices=["Cloning", "Preset"],
            value="Cloning",
            interactive=True,
        )

    # --- Voice ref upload (visible only when cloning) ---
    show_clone = mode in ("clone_only", "both")
    voice_ref = gr.update(visible=show_clone)

    # --- Preset dropdown (visible only when preset available) ---
    speakers = []
    if mode in ("preset_only", "both"):
        try:
            speakers = _get_speakers(key, None)
        except Exception:
            pass

    show_preset = mode in ("preset_only", "both")
    preset_dd = gr.update(
        visible=show_preset,
        choices=speakers if speakers else [],
        value=speakers[0] if speakers else None,
    )

    # --- Warning for unavailable ---
    warning = ""
    if not available:
        warning = (
            f"**{info['name']}** is not installed. "
            f"Install with: `{info['package']}`"
        )

    # --- Language ---
    lang = _default_language(key)

    return (
        voice_radio,
        voice_ref,
        preset_dd,
        gr.update(value=info_text),
        gr.update(value=warning),
        gr.update(value=lang),
    )


def on_voice_type_change(voice_type, model_choice):
    """Show/hide cloning upload vs preset dropdown based on radio selection."""
    key = _key_from_label(model_choice)
    is_clone = voice_type == "Cloning"

    if key and _voice_mode(key) == "clone_only":
        return gr.update(visible=True), gr.update(visible=False)
    if key and _voice_mode(key) == "preset_only":
        return gr.update(visible=False), gr.update(visible=True)

    return gr.update(visible=is_clone), gr.update(visible=not is_clone)


def on_extract(file):
    if file is None:
        return ""
    try:
        text = extract_text_from_file(file.name)
        return text
    except Exception as e:
        raise gr.Error(f"Failed to extract text: {e}")


def on_sanitize(text, ollama_model, do_sanitize):
    if not do_sanitize or not text or not text.strip():
        return text, "Skipped."
    try:
        start = time.time()
        sanitized = sanitize_text(text, model=ollama_model)
        elapsed = time.time() - start
        chars_before = len(text)
        chars_after = len(sanitized)
        status = f"Sanitized in {elapsed:.1f}s: {chars_before} -> {chars_after} chars"
        return sanitized, status
    except Exception as e:
        raise gr.Error(f"Sanitization failed: {e}")


def on_generate(
    text, model_choice, voice_type, voice_ref, preset_speaker,
    language, output_name,
):
    if not text or not text.strip():
        raise gr.Error("No text to generate speech from. Extract or paste text first.")
    if not model_choice:
        raise gr.Error("Select a model first.")

    key = _key_from_label(model_choice)
    if not key or key not in MODEL_REGISTRY:
        raise gr.Error("Invalid model selection.")

    if not AVAILABLE_MODELS.get(key, False):
        raise gr.Error(
            f"Model '{MODEL_REGISTRY[key]['name']}' is not installed. "
            f"Install it with: {MODEL_REGISTRY[key]['package']}"
        )

    mode = _voice_mode(key)

    # Determine cloning vs preset based on what the model actually supports
    if mode == "clone_only":
        use_cloning = True
        if not voice_ref:
            raise gr.Error(f"{MODEL_REGISTRY[key]['name']} requires a voice reference .wav file.")
        cloning_path = voice_ref.name
        preset = ""
    elif mode == "preset_only":
        use_cloning = False
        cloning_path = ""
        preset = preset_speaker or "default"
    else:
        # "both" — respect user's radio choice
        use_cloning = voice_type == "Cloning"
        if use_cloning:
            if not voice_ref:
                raise gr.Error("Voice cloning requires a reference .wav file.")
            cloning_path = voice_ref.name
            preset = ""
        else:
            cloning_path = ""
            preset = preset_speaker or "default"

    out_name = (output_name or "output").strip()
    if not out_name.endswith(".wav"):
        out_name += ".wav"

    tmp_dir = tempfile.mkdtemp(prefix="doctospeech_")
    output_path = os.path.join(tmp_dir, out_name)

    try:
        tts_obj, load_time = load_model(key)
    except Exception as e:
        raise gr.Error(f"Failed to load model: {e}")

    try:
        make_tts_voiceover(
            text, use_cloning, preset, cloning_path,
            language or "en", tts_obj, output_path, model_key=key,
        )
    except Exception as e:
        raise gr.Error(f"Generation failed: {e}")

    if not os.path.isfile(output_path):
        raise gr.Error("Generation completed but output file not found.")

    return output_path, f"Generated in {load_time:.1f}s (model load) + synthesis"


# ========================= BUILD UI =========================

INSTALLED_CHOICES = [model_label(k) for k in MODEL_KEYS if AVAILABLE_MODELS[k]]
DEVICE = _get_device()
INITIAL_MODEL = _first_available_choice()


def _uninstalled_info():
    lines = []
    for k in MODEL_KEYS:
        if not AVAILABLE_MODELS[k]:
            info = MODEL_REGISTRY[k]
            lines.append(f"- **{info['name']}** (`{info['package']}`)")
    if not lines:
        return ""
    return "**Not installed** (each needs its own venv):\n" + "\n".join(lines)

with gr.Blocks(title="DocToSpeech") as demo:
    gr.Markdown("# DocToSpeech")
    gr.Markdown(
        "Convert documents to speech using local TTS models. "
        "Fully offline -- no API keys or cloud services required."
    )

    has_any_model = any(AVAILABLE_MODELS.values())

    if not has_any_model:
        gr.Markdown(
            "### No TTS models installed\n"
            "Install at least one model. See `requirements-coqui.txt` for the base set, "
            "or individual `requirements-*.txt` files for other models.\n\n"
            "**Each model family requires its own virtual environment** due to dependency conflicts."
        )

    gr.Markdown(f"**Device:** `{DEVICE}`")

    uninstalled_md = _uninstalled_info()
    if uninstalled_md:
        gr.Markdown(uninstalled_md)

    # ---- Document upload & text ----
    with gr.Row():
        with gr.Column(scale=1):
            doc_upload = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".epub", ".docx", ".html", ".htm", ".txt"],
            )
            extract_btn = gr.Button("Extract Text", variant="secondary")

        with gr.Column(scale=2):
            text_box = gr.Textbox(
                label="Extracted Text",
                lines=15,
                placeholder="Upload a document and click Extract, or paste text here...",
            )

    gr.Markdown("---")

    # ---- Model & voice settings ----
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                label="TTS Model",
                choices=INSTALLED_CHOICES,
                value=INITIAL_MODEL,
                info="Only installed models are shown." if INSTALLED_CHOICES else "No models installed.",
            )

            model_warning = gr.Markdown(visible=True, value="")

            model_info_display = gr.Markdown(
                value="Select a model to see its details.",
            )

            with gr.Row():
                language_input = gr.Textbox(
                    label="Language Code",
                    value="en",
                    info="e.g. en, es, fr, de (model-dependent)",
                )
                output_name_input = gr.Textbox(
                    label="Output Filename",
                    value="output",
                    info="Without .wav extension",
                )

        with gr.Column(scale=1):
            voice_type_radio = gr.Radio(
                label="Voice Type",
                choices=["Cloning"],
                value="Cloning",
                visible=True,
            )

            voice_ref_upload = gr.File(
                label="Voice Reference (.wav)",
                file_types=[".wav"],
                visible=True,
            )

            preset_dropdown = gr.Dropdown(
                label="Preset Speaker",
                choices=[],
                visible=False,
            )

    gr.Markdown("---")

    # ---- Ollama sanitization ----
    with gr.Accordion("Ollama Sanitization (optional)", open=False):
        sanitize_check = gr.Checkbox(label="Sanitize text with Ollama", value=False)
        sanitize_model_input = gr.Textbox(
            label="Ollama Model",
            value="llama3.1",
        )
        sanitize_btn = gr.Button("Sanitize Text", variant="secondary")
        sanitize_status = gr.Markdown(visible=False, value="")

    gr.Markdown("---")

    generate_btn = gr.Button("Generate Audio", variant="primary", size="lg", interactive=has_any_model)
    gen_status = gr.Markdown(value="")
    audio_output = gr.Audio(label="Generated Audio", type="filepath")

    # ---- Event wiring ----

    extract_btn.click(
        fn=on_extract,
        inputs=[doc_upload],
        outputs=[text_box],
    )

    model_dropdown.change(
        fn=on_model_change,
        inputs=[model_dropdown],
        outputs=[
            voice_type_radio,
            voice_ref_upload,
            preset_dropdown,
            model_info_display,
            model_warning,
            language_input,
        ],
    )

    voice_type_radio.change(
        fn=on_voice_type_change,
        inputs=[voice_type_radio, model_dropdown],
        outputs=[voice_ref_upload, preset_dropdown],
    )

    sanitize_btn.click(
        fn=on_sanitize,
        inputs=[text_box, sanitize_model_input, sanitize_check],
        outputs=[text_box, sanitize_status],
    )

    sanitize_check.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[sanitize_check],
        outputs=[sanitize_status],
    )

    generate_btn.click(
        fn=on_generate,
        inputs=[
            text_box, model_dropdown, voice_type_radio,
            voice_ref_upload, preset_dropdown,
            language_input, output_name_input,
        ],
        outputs=[audio_output, gen_status],
    )

    # ---- Initialize UI state for default model ----
    demo.load(
        fn=on_model_change,
        inputs=[model_dropdown],
        outputs=[
            voice_type_radio,
            voice_ref_upload,
            preset_dropdown,
            model_info_display,
            model_warning,
            language_input,
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
