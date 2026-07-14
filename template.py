import os
import sys
import time
import threading

import ollama
from tqdm import tqdm


# ========================= AGENT PROMPT =========================

AGENT_MD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.md")

def load_agent_prompt(custom_path=None):
    """Load system prompt from agent.md or a custom path."""
    path = custom_path or AGENT_MD_PATH
    if not os.path.isfile(path):
        print(f"[!] Agent prompt not found: {path}, using fallback")
        return (
            "You are a text sanitization assistant for TTS preparation.\n"
            "Clean raw extracted text so it reads naturally when spoken aloud.\n"
            "Fix OCR errors, remove artifacts, normalize whitespace.\n"
            "Output ONLY the cleaned text with no commentary."
        )
    with open(path, "r") as f:
        content = f.read().strip()
    if not content:
        print(f"[!] Agent prompt is empty: {path}, using fallback")
        return (
            "You are a text sanitization assistant for TTS preparation.\n"
            "Clean raw extracted text so it reads naturally when spoken aloud.\n"
            "Fix OCR errors, remove artifacts, normalize whitespace.\n"
            "Output ONLY the cleaned text with no commentary."
        )
    return content


# ========================= SPINNER =========================

SPIN_FRAMES = ["|", "/", "-", "\\"]

def _spinner(label, elapsed_func, stop_event):
    """Display a spinner with elapsed time until stop_event is set."""
    idx = 0
    while not stop_event.is_set():
        frame = SPIN_FRAMES[idx % len(SPIN_FRAMES)]
        secs = elapsed_func()
        print(f"\r  {frame} {label} ({secs:.0f}s)  ", end="", flush=True)
        idx += 1
        stop_event.wait(0.15)
    secs = elapsed_func()
    print(f"\r  [+] {label} ({secs:.1f}s)  ")


# ========================= CHUNKING =========================

def _chunk_text(text, max_chars=4000):
    """Split text into chunks at paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks = []
    current = []

    for para in paragraphs:
        test = "\n\n".join(current + [para])
        if len(test) > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [para]
        else:
            current.append(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ========================= SANITIZE FUNCTION =========================

def sanitize_text(text, model="llama3.1", agent_prompt=None, label="Sanitizing"):
    """
    Send text through Ollama for sanitization.
    Chunks large documents at paragraph boundaries (~4k chars).
    """
    system_prompt = load_agent_prompt(agent_prompt)
    chunks = _chunk_text(text)
    cleaned = []

    start = time.time()
    elapsed = lambda: time.time() - start

    stop_event = threading.Event()
    spinner = threading.Thread(target=_spinner, args=(f"{label} ({len(chunks)} chunk(s))", elapsed, stop_event), daemon=True)
    spinner.start()

    for i, chunk in enumerate(chunks):
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Below is a text chunk extracted from a document. "
                    f"Sanitize it for TTS readability following your system instructions. "
                    f"Output ONLY the sanitized text, nothing else.\n\n"
                    f"---\n{chunk}\n---"
                ),
            },
        ]

        response = ollama.chat(model=model, messages=messages)
        msg = response["message"]
        cleaned.append(msg.get("content", chunk))

    stop_event.set()
    spinner.join()

    return "\n\n".join(cleaned)


# ========================= INTERACTIVE MODE =========================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # File mode: sanitize a file in-place
        filepath = sys.argv[1]
        model = sys.argv[2] if len(sys.argv) > 2 else "llama3.1"

        print(f"Reading: {filepath}")
        with open(filepath, "r") as f:
            raw = f.read()
        print(f"Original: {len(raw)} chars")

        cleaned = sanitize_text(raw, model=model, label=os.path.basename(filepath))
        print(f"Cleaned:  {len(cleaned)} chars")

        backup = filepath + ".bak"
        with open(backup, "w") as f:
            f.write(raw)
        print(f"Backup:   {backup}")

        with open(filepath, "w") as f:
            f.write(cleaned)
        print(f"Saved:    {filepath}")
    else:
        # Interactive mode
        print("Text Sanitizer (interactive)")
        print("Paste text, then Ctrl+D (Linux/Mac) or Ctrl+Z (Windows) to sanitize\n")
        raw = sys.stdin.read()
        if raw.strip():
            cleaned = sanitize_text(raw)
            print(f"\n{'=' * 60}")
            print(cleaned)
            print(f"{'=' * 60}")