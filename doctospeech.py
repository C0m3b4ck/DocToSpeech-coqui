import os
import pathlib
import sys

from tqdm import tqdm
import time
from datetime import datetime

import pyfiglet


# ========================= TERMINAL COLORS =========================

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    BG_BLUE = "\033[44m"

def prompt(msg):
    return input(f"{C.CYAN}{C.BOLD}?{C.RESET} {C.WHITE}{msg}{C.RESET} ")

def info(msg):
    print(f"  {C.DIM}[i]{C.RESET} {C.DIM}{msg}{C.RESET}")

def success(msg):
    print(f"  {C.GREEN}[+]{C.RESET} {C.GREEN}{msg}{C.RESET}")

def warn(msg):
    print(f"  {C.YELLOW}[!]{C.RESET} {C.YELLOW}{msg}{C.RESET}")

def error(msg):
    print(f"  {C.RED}[x]{C.RESET} {C.RED}{msg}{C.RESET}")

def header(msg):
    width = 52
    padding = width - len(msg) - 6
    print()
    print(f"  {C.CYAN}{C.BOLD}+{'-' * (width - 2)}+{C.RESET}")
    print(f"  {C.CYAN}{C.BOLD}| {C.BOLD}{msg}{' ' * max(padding, 1)}|{C.RESET}")
    print(f"  {C.CYAN}{C.BOLD}+{'-' * (width - 2)}+{C.RESET}")

def divider():
    print(f"  {C.DIM}{'.' * 50}{C.RESET}")

def box(msg, color=C.CYAN):
    lines = msg.split("\n")
    width = max(len(l) for l in lines) + 4
    print(f"  {color}+{'-' * (width - 2)}+{C.RESET}")
    for line in lines:
        pad = width - len(line) - 4
        print(f"  {color}|{C.RESET} {C.BOLD}{line}{' ' * max(pad, 1)}{color}|{C.RESET}")
    print(f"  {color}+{'-' * (width - 2)}+{C.RESET}")

def status_line(label, value, label_color=C.DIM, value_color=C.WHITE):
    print(f"  {label_color}{label:>20}{C.RESET} {C.DIM}:{C.RESET} {value_color}{value}{C.RESET}")


# ========================= PATH SAFETY =========================

def safe_resolve(path):
    resolved = os.path.realpath(path)
    if ".." in path.split(os.sep):
        warn(f"Path contains traversal components: {path}")
    return resolved

def validate_path_exists(path, label="path"):
    if not os.path.exists(path):
        error(f"{label} does not exist: {path}")
        return None
    return os.path.realpath(path)


# ========================= AGENT PROMPT =========================

AGENT_MD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.md")

def load_agent_prompt(custom_path=None):
    """Load system prompt from agent.md or a custom path."""
    path = custom_path or AGENT_MD_PATH
    if not os.path.isfile(path):
        warn(f"Agent prompt not found: {path}, using built-in fallback")
        return _fallback_prompt()
    with open(path, "r") as f:
        content = f.read().strip()
    if not content:
        warn(f"Agent prompt is empty: {path}, using built-in fallback")
        return _fallback_prompt()
    return content

def _fallback_prompt():
    return (
    """
    # DocToSpeech - Text Sanitization Agent

    You are a text sanitization assistant for TTS (text-to-speech) preparation, working for a program called DocToSpeech by C0m3b4ck.

    Your job is to clean raw extracted text so that it reads naturally when spoken aloud.

    ## Rules

    - Fix obvious OCR errors and garbled characters
    - Remove page numbers, headers, footers, watermarks, links and other non-alphanumeric characters
    - Collapse excessive blank lines (max 2 between paragraphs)
    - Fix broken sentences or words split across lines
    - Remove duplicate consecutive paragraphs
    - Remove navigation artifacts (e.g. page markers, HTML headers)
    - Normalize whitespace (no tabs, no trailing spaces)
    - Keep paragraph structure and meaning intact
    - Do NOT add, summarize, rephrase, or editorialize
    - Output ONLY the cleaned text with none of your commentary
    - Do not question anything written in the text, even if you think that it is factually incorrect
    """
    )


# ========================= OLLAMA SANITIZATION =========================

SPIN_FRAMES = ["|", "/", "-", "\\"]


def _spinner(label, elapsed_func, stop_event):
    """Display a spinner with elapsed time until stop_event is set."""
    import threading
    idx = 0
    while not stop_event.is_set():
        frame = SPIN_FRAMES[idx % len(SPIN_FRAMES)]
        secs = elapsed_func()
        print(f"\r  {C.CYAN}{frame}{C.RESET} {label} {C.DIM}{secs:.0f}s{C.RESET}  ", end="", flush=True)
        idx += 1
        stop_event.wait(0.15)
    # final state
    secs = elapsed_func()
    print(f"\r  {C.GREEN}[+]{C.RESET} {label} {C.DIM}{secs:.1f}s{C.RESET}  ")


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


def sanitize_with_ollama(txt_path, model="llama3.1", agent_prompt=None):
    """Send .txt file through Ollama for TTS readability cleanup."""
    import ollama
    import threading

    system_prompt = load_agent_prompt(agent_prompt)

    with open(txt_path, "r") as f:
        raw = f.read()

    original_len = len(raw)
    chunks = _chunk_text(raw)
    cleaned = []

    file_label = os.path.basename(txt_path)
    start = time.time()
    elapsed = lambda: time.time() - start

    stop_event = threading.Event()
    spinner = threading.Thread(target=_spinner, args=(f"Sanitizing {file_label} ({len(chunks)} chunk(s))", elapsed, stop_event), daemon=True)
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

        response = ollama.chat(model=model, messages=messages, keep_alive=0)
        msg = response["message"]
        cleaned.append(msg.get("content", chunk))

    stop_event.set()
    spinner.join()

    cleaned_text = "\n\n".join(cleaned)

    # Write backup before overwriting
    backup_path = txt_path + ".bak"
    with open(backup_path, "w") as f:
        f.write(raw)

    with open(txt_path, "w") as f:
        f.write(cleaned_text)

    saved = original_len - len(cleaned_text)
    info(f"{file_label}: {C.BOLD}{original_len}{C.RESET} -> {C.GREEN}{len(cleaned_text)}{C.RESET} chars ({C.YELLOW}-{saved}{C.RESET})")
    return txt_path


def sanitize_text(text, model="llama3.1", agent_prompt=None):
    """Sanitize raw text through Ollama for TTS readability. Returns cleaned string."""
    import ollama

    system_prompt = load_agent_prompt(agent_prompt)
    chunks = _chunk_text(text)
    cleaned = []

    for chunk in chunks:
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
        response = ollama.chat(model=model, messages=messages, keep_alive=0)
        cleaned.append(response["message"].get("content", chunk))

    return "\n\n".join(cleaned)


def _ask_sanitize_options():
    """Shared prompt for sanitization options. Returns (do_sanitize, model, custom_agent_path)."""
    sanitize_choice = prompt("Sanitize text with Ollama for better TTS? (y/n): ").lower().strip()
    if sanitize_choice != "y":
        return False, None, None

    model = prompt("Ollama model (default: llama3.1): ").strip() or "llama3.1"

    agent_path = None
    custom = prompt("Use custom agent.md? (y/n): ").lower().strip()
    if custom == "y":
        agent_path = prompt("Path to custom agent.md: ").strip()
        agent_path = validate_path_exists(agent_path, "Agent prompt file")
        if agent_path is None:
            warn("Falling back to default agent.md")
            agent_path = None

    return True, model, agent_path


# ========================= USER INTERACTION =========================

def greet():
    print()
    ascii_art = pyfiglet.figlet_format("DocToSpeech", font="alphabet")
    print(f"{C.CYAN}{ascii_art}{C.RESET}")
    time.sleep(1)
    credit = pyfiglet.figlet_format("By C0m3b4ck under MPL 2.0")
    print(f"{C.DIM}{credit}{C.RESET}")

def goodbye():
    print()
    ascii_art = pyfiglet.figlet_format("Goodbye!", font="alphabet")
    print(f"{C.CYAN}{ascii_art}{C.RESET}")
    time.sleep(1)
    credit = pyfiglet.figlet_format("By C0m3b4ck under MPL 2.0")
    print(f"{C.DIM}{credit}{C.RESET}")


def get_user_input_volume():
    while True:
        header("Main Menu")
        status_line("Mode", f"{C.CYAN}s{C.RESET}ingle file  /  {C.CYAN}d{C.RESET}irectory")
        print()
        choice = prompt("Select mode (s/d): ").lower().strip()
        if choice == "s":
            get_user_input_docs()
        elif choice == "d":
            get_user_input_dir()
        else:
            warn("Please input 's' or 'd'")
            continue

        again = prompt("Process another document? (y/n): ").lower().strip()
        if again != "y":
            break


def get_user_input_dir():
    header("Directory Mode")

    output_dir = prompt("Output directory: ")
    output_dir = validate_path_exists(output_dir, "Output directory")
    if output_dir is None:
        return

    doc_dir = prompt("Document directory: ")
    doc_dir = validate_path_exists(doc_dir, "Document directory")
    if doc_dir is None:
        return

    while True:
        try:
            number = int(prompt("Number of extensions to convert (0 = all supported): "))
            if number < 0:
                warn("Please input a non-negative number")
                continue
            break
        except ValueError:
            warn("Please input a number")

    default_extentions = [".txt", ".pdf", ".html", ".docx", ".epub"]
    extention_list = []
    if number > 0:
        for x in range(1, number + 1):
            ext = prompt(f"Extension {x}/{number} (e.g. '.pdf'): ").lower().strip()
            if not ext.startswith("."):
                ext = "." + ext
            extention_list.append(ext)
    else:
        info("Using default extension list")
        extention_list = default_extentions

    header("Scanning files")
    file_list = [
        f for f in os.listdir(doc_dir)
        if os.path.isfile(os.path.join(doc_dir, f))
        and f.endswith(tuple(extention_list))
    ]

    if not file_list:
        warn("No matching files found in directory")
        return

    info(f"Found {C.BOLD}{len(file_list)}{C.RESET}{C.DIM} file(s){C.RESET}")
    divider()
    for f in file_list:
        ext = pathlib.Path(f).suffix.lower()
        print(f"    {C.DIM}>{C.RESET} {f}  {C.DIM}({ext}){C.RESET}")
    divider()

    epub_chapter_choice = ""
    if any(f.endswith(".epub") for f in file_list):
        while epub_chapter_choice not in ("y", "n"):
            epub_chapter_choice = prompt("Split epub files into separate chapters? (y/n): ").lower()

    header("Converting to .txt")
    txtfile_list = []
    for filename in tqdm(file_list, desc="  Converting", bar_format="    {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"):
        full_path = os.path.join(doc_dir, filename)
        file_ext = pathlib.Path(filename).suffix.lower()
        try:
            if file_ext == ".epub":
                txt_paths = epub_to_text(full_path, epub_chapter_choice)
                txtfile_list.extend(txt_paths)
            elif file_ext == ".pdf":
                txt_path = pdf_to_text(full_path)
                txtfile_list.append(txt_path)
            elif file_ext == ".docx":
                txt_path = docx_to_text(full_path)
                txtfile_list.append(txt_path)
            elif file_ext == ".doc":
                warn(f".doc not implemented yet: {filename}")
            elif file_ext in (".html", ".htm"):
                txt_path = html_to_text(full_path)
                txtfile_list.append(txt_path)
            elif file_ext == ".djvu":
                warn(f".djvu not implemented yet: {filename}")
        except Exception as e:
            error(f"Failed to convert {filename}: {e}")

    if not txtfile_list:
        warn("No files were successfully converted")
        return

    divider()
    success(f"Converted {C.BOLD}{len(txtfile_list)}{C.RESET}{C.GREEN} file(s){C.RESET}")

    # optional ollama sanitization
    do_sanitize, model, agent_path = _ask_sanitize_options()
    if do_sanitize:
        header("Sanitizing with Ollama")
        info(f"Model: {C.BOLD}{model}{C.RESET}")
        if agent_path:
            info(f"Custom agent: {C.BOLD}{agent_path}{C.RESET}")
        for txt_path in txtfile_list:
            try:
                sanitize_with_ollama(txt_path, model=model, agent_prompt=agent_path)
            except Exception as e:
                error(f"Failed to sanitize {os.path.basename(txt_path)}: {e}")
        success("Sanitization complete")

    header("TTS Options")
    model_key, tts_obj, model_info = select_model()
    if tts_obj is None:
        warn("TTS model not loaded, skipping generation")
        return

    language = prompt("Language code (e.g. 'en'): ")

    cloning_path = ""
    use_cloning = False
    preset_speaker = ""

    if model_key == "chatterbox":
        cloning_path = prompt("Path to .wav clone voice: ")
        use_cloning = True
    elif model_info["cloning"]:
        voice_choice = ""
        while voice_choice not in ("c", "p"):
            voice_choice = prompt("Voice type: cloning (c) or preset (p)? ").lower()
            if voice_choice == "c":
                cloning_path = prompt("Path to .wav clone voice: ")
                use_cloning = True
            elif voice_choice == "p":
                preset_speaker = prompt("Preset speaker name: ")
                use_cloning = False
            else:
                warn("Please input 'c' or 'p'")
    else:
        preset_speaker = prompt("Preset speaker name: ") or "default"
        use_cloning = False

    header("Generating audio")
    info(f"Processing {C.BOLD}{len(txtfile_list)}{C.RESET}{C.DIM} file(s){C.RESET}")
    for txt_path in txtfile_list:
        with open(txt_path, "r") as file:
            text_to_say = file.read()
        output_filename = os.path.basename(txt_path).rsplit('.', 1)[0] + ".wav"
        output_path = os.path.join(output_dir, output_filename)
        make_tts_voiceover(text_to_say, use_cloning, preset_speaker, cloning_path, language, tts_obj, output_path, model_key=model_key)

    success("All files processed!")


def get_user_input_docs():
    while True:
        header("Single File Mode")
        info("Supported formats: PDF, EPUB, DOCX, DOC, HTML, DJVU, TXT")

        doc_path = prompt("Document path: ")
        doc_path = validate_path_exists(doc_path, "Document")
        if doc_path is None:
            continue

        file_ext = pathlib.Path(doc_path).suffix.lower()
        divider()
        status_line("File", os.path.basename(doc_path))
        status_line("Extension", f"{C.BOLD}{file_ext}{C.RESET}")
        divider()

        try:
            if file_ext == ".epub":
                txt_paths = epub_to_text(doc_path)
                for txt_path in txt_paths:
                    get_user_input_tts(txt_path)
            elif file_ext == ".pdf":
                txt_path = pdf_to_text(doc_path)
                get_user_input_tts(txt_path)
            elif file_ext == ".docx":
                txt_path = docx_to_text(doc_path)
                get_user_input_tts(txt_path)
            elif file_ext == ".doc":
                warn("Not implemented yet (requires .doc to .docx conversion)")
                continue
            elif file_ext in (".html", ".htm"):
                txt_path = html_to_text(doc_path)
                get_user_input_tts(txt_path)
            elif file_ext == ".djvu":
                warn("Not implemented yet")
                continue
            elif file_ext == ".txt":
                info("File is already plain text, proceeding to TTS...")
                get_user_input_tts(doc_path)
            else:
                warn(f"Unrecognized extension: {file_ext}")
                choice = prompt("Treat as plain text? (y/n): ").lower()
                if choice == "y":
                    info("Treating as plain text, proceeding to TTS...")
                    get_user_input_tts(doc_path)
                else:
                    info("Returning to file selection")
                    continue
        except Exception as e:
            error(f"Something went wrong: {e}")
            retry = prompt("Try again? (y/n): ").lower()
            if retry != "y":
                break
            continue

        break


def get_user_input_tts(txt_path):
    header("TTS Configuration")

    output_file_name = prompt("Output filename (without .wav): ").strip()

    choice_timestamp = ""
    while choice_timestamp not in ("y", "n"):
        choice_timestamp = prompt("Add timestamp to filename? (y/n): ").lower()

    if choice_timestamp == "y":
        ts = datetime.now().timestamp()
        output_path = f"{output_file_name}_{ts}.wav"
    else:
        output_path = output_file_name + ".wav"

    divider()
    status_line("Output", f"{C.BOLD}{output_path}{C.RESET}")
    divider()

    # Model selection
    model_key, tts_obj, model_info = select_model()
    if tts_obj is None:
        warn("TTS model not loaded, skipping generation")
        return

    language = prompt("Language code (e.g. 'en'): ")

    cloning_path = ""
    use_cloning = False
    preset_speaker = ""

    if model_key == "chatterbox":
        cloning_path = prompt("Path to .wav clone voice: ")
        use_cloning = True
    elif model_info["cloning"]:
        voice_choice = ""
        while voice_choice not in ("c", "p"):
            voice_choice = prompt("Voice type: cloning (c) or preset (p)? ").lower()
            if voice_choice == "c":
                cloning_path = prompt("Path to .wav clone voice: ")
                use_cloning = True
            elif voice_choice == "p":
                preset_speaker = prompt("Preset speaker name: ")
                use_cloning = False
            else:
                warn("Please input 'c' or 'p'")
    else:
        preset_speaker = prompt("Preset speaker name: ") or "default"
        use_cloning = False

    with open(txt_path, "r") as file:
        file_contents = file.read()

    preview = file_contents[:200] + ("..." if len(file_contents) > 200 else "")
    divider()
    info(f"Preview: {C.DIM}{preview}{C.RESET}")
    divider()

    # optional ollama sanitization
    do_sanitize, sanitize_model, agent_path = _ask_sanitize_options()
    if do_sanitize:
        try:
            sanitize_with_ollama(txt_path, model=sanitize_model, agent_prompt=agent_path)
            with open(txt_path, "r") as file:
                file_contents = file.read()
            info("Sanitized preview:")
            preview = file_contents[:200] + ("..." if len(file_contents) > 200 else "")
            info(f"Preview: {C.DIM}{preview}{C.RESET}")
        except Exception as e:
            error(f"Failed to sanitize: {e}")
            warn("Proceeding with unsanitized text")

    make_tts_voiceover(file_contents, use_cloning, preset_speaker, cloning_path, language, tts_obj, output_path, model_key=model_key)


# ========================= DOC -> TXT =========================

def epub_to_text(doc_path, chapter_choice=None):
    from epub2txt import epub2txt

    if chapter_choice not in ("y", "n"):
        while True:
            chapter_choice = prompt("Split epub into separate chapters? (y/n): ").lower()
            if chapter_choice in ("y", "n"):
                break

    try:
        chapter_list = epub2txt(doc_path, outputlist=True)
    except Exception as e:
        error(f"Failed to read epub: {e}")
        raise

    info(f"Read {C.BOLD}{len(chapter_list)}{C.RESET}{C.DIM} chapter(s){C.RESET}")

    base_name = doc_path.rsplit('.', 1)[0]
    txt_files = []

    if chapter_choice == "y":
        for i, chapter in enumerate(chapter_list):
            chapter_file = f"{base_name}_chapter_{i + 1}.txt"
            with open(chapter_file, "w") as f:
                f.write(chapter)
            txt_files.append(chapter_file)
        del chapter_list
        success(f"Saved {len(txt_files)} chapter files")
    else:
        full_txt = "\n\n".join(chapter_list)
        del chapter_list
        txt_file = doc_path + ".txt"
        with open(txt_file, "w") as f:
            f.write(full_txt)
        del full_txt
        txt_files.append(txt_file)
        success(f"Saved: {txt_file}")

    return txt_files


def _is_scanned_pdf(pdf):
    """Check if a pdftotext.PDF object contains no meaningful text (image-only scans)."""
    total_chars = sum(len(page.strip()) for page in pdf)
    num_pages = len(pdf)
    if num_pages == 0:
        return True
    avg_chars_per_page = total_chars / num_pages
    return avg_chars_per_page < 50


def pdf_to_text_ocr(doc_path, password=None):
    """Extract text from a scanned/image-only PDF using OCR (pdf2image + pytesseract)."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError as e:
        raise ImportError(
            f"OCR dependencies not installed: {e}\n"
            "Install with: pip install pdf2image pytesseract\n"
            "Also requires Tesseract binary: sudo apt install tesseract-ocr"
        )

    info("PDF appears to be a scanned document (no text layer). Using OCR...")
    images = convert_from_path(doc_path, password=password or "")
    info(f"Converted {C.BOLD}{len(images)}{C.RESET}{C.DIM} page(s) to images{C.RESET}")

    pages = []
    for i, img in enumerate(tqdm(images, desc="  OCR", bar_format="    {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")):
        text = pytesseract.image_to_string(img)
        pages.append(text)
        if i > 0:
            pass  # handled during write

    del images

    txt_file = doc_path + ".txt"
    try:
        with open(txt_file, "w") as save_text:
            for i, page in enumerate(pages):
                if i > 0:
                    save_text.write("\n\n")
                save_text.write(page)
    except Exception as e:
        error(f"Failed to save txt: {e}")
        raise

    success(f"Saved: {txt_file} (OCR)")
    return txt_file


def pdf_to_text(doc_path):
    import pdftotext
    choice = ""
    password = None
    pdf = None

    while choice not in ("y", "n"):
        choice = prompt("Is this PDF password-protected? (y/n): ").lower()

    try:
        with open(doc_path, "rb") as f:
            if choice == "y":
                password = prompt("Enter PDF password: ")
                pdf = pdftotext.PDF(f, password)
            else:
                pdf = pdftotext.PDF(f)
    except Exception as e:
        error(f"Failed to open PDF: {e}")
        raise
    finally:
        password = None

    info(f"PDF has {C.BOLD}{len(pdf)}{C.RESET}{C.DIM} page(s){C.RESET}")

    if _is_scanned_pdf(pdf):
        warn("No extractable text found — detected as scanned/image-only PDF")
        del pdf
        return pdf_to_text_ocr(doc_path, password=password)

    show = ""
    while show not in ("y", "n"):
        show = prompt("Show all pages? (y/n): ").lower()
    if show == "y":
        for page in pdf:
            print(page)

    txt_file = doc_path + ".txt"
    try:
        with open(txt_file, "w") as save_text:
            for i, page in enumerate(pdf):
                if i > 0:
                    save_text.write("\n\n")
                save_text.write(page)
        del pdf
    except Exception as e:
        error(f"Failed to save txt: {e}")
        raise

    success(f"Saved: {txt_file}")
    return txt_file


def docx_to_text(doc_path):
    import docx2txt
    txt_string = docx2txt.process(doc_path)
    txt_file = doc_path + ".txt"

    try:
        with open(txt_file, "w") as save_text:
            save_text.write(txt_string)
        del txt_string
    except Exception as e:
        error(f"Failed to save txt: {e}")
        raise

    success(f"Saved: {txt_file}")
    return txt_file


def doc_to_text(doc_path):
    warn("Not implemented yet -- requires .doc to .docx conversion")


def html_to_text(doc_path):
    import html2text
    try:
        with open(doc_path, 'r') as file:
            html_contents = file.read()
    except Exception as e:
        error(f"Failed to read HTML: {e}")
        raise

    try:
        h = html2text.HTML2Text()
        h.ignore_links = True
        txt_string = h.handle(html_contents)
        del html_contents
    except Exception as e:
        error(f"Failed to convert HTML: {e}")
        raise

    txt_file = doc_path + ".txt"
    try:
        with open(txt_file, "w") as save_text:
            save_text.write(txt_string)
        del txt_string
    except Exception as e:
        error(f"Failed to save txt: {e}")
        raise

    success(f"Saved: {txt_file}")
    return txt_file


def djvu_to_text(doc_path):
    warn("Not implemented yet")


# ========================= TTS MODEL REGISTRY =========================

MODEL_REGISTRY = {
    "xtts_v2": {
        "name": "Coqui XTTS v2",
        "desc": "Multilingual voice cloning, 17 languages, high quality",
        "vram": "~4 GB",
        "gpu": "Recommended (CPU possible, ~5-10x slower)",
        "cloning": True,
        "languages": "17 (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja, hu, ko, bg)",
        "package": "coqui-tts",
        "init": "init_xtts",
    },
    "bark": {
        "name": "Coqui Bark",
        "desc": "Multi-lingual, supports music/sound effects, conversational",
        "vram": "~4-12 GB (depends on quality setting)",
        "gpu": "Recommended (CPU possible, very slow)",
        "cloning": False,
        "languages": "Multi-lingual (en, de, es, fr, it, ja, ko, pl, pt, ru, zh, hi, ar, tr)",
        "package": "coqui-tts",
        "init": "init_bark",
    },
    "vits": {
        "name": "Coqui VITS",
        "desc": "Fast feed-forward model, preset speakers only",
        "vram": "~2-4 GB",
        "gpu": "Recommended (CPU works)",
        "cloning": False,
        "languages": "Varies by model (English default)",
        "package": "coqui-tts",
        "init": "init_vits",
    },
    "yourtts": {
        "name": "Coqui YourTTS",
        "desc": "Multi-speaker, multi-lingual, voice cloning",
        "vram": "~3-5 GB",
        "gpu": "Recommended (CPU possible)",
        "cloning": True,
        "languages": "Multi-lingual (en, pt, fr, de, it, es, nl, pl, tr, ru, cs, ar, zh, ja, hu, ko)",
        "package": "coqui-tts",
        "init": "init_yourtts",
    },
    "chatterbox": {
        "name": "Resemble Chatterbox",
        "desc": "350M params, paralinguistic tags [laugh], fast",
        "vram": "~2-4 GB",
        "gpu": "Recommended (CPU possible)",
        "cloning": True,
        "languages": "English (+ 23 via Multilingual V3)",
        "package": "pip install chatterbox-tts (may need --no-deps if torch conflicts)",
        "init": "init_chatterbox",
    },
    "tortoise": {
        "name": "Tortoise TTS",
        "desc": "High quality autoregressive, very slow but excellent",
        "vram": "~4-6 GB (can use 4GB with batch_size=1)",
        "gpu": "Required for reasonable speed",
        "cloning": True,
        "languages": "English (mostly)",
        "package": "pip install tortoise-tts (needs old transformers, use separate venv)",
        "init": "init_tortoise",
    },
    "styletts2": {
        "name": "StyleTTS 2",
        "desc": "Human-level quality, diffusion-based style generation",
        "vram": "~4-8 GB",
        "gpu": "Recommended (CPU possible, slow)",
        "cloning": True,
        "languages": "English",
        "package": "pip install styletts2",
        "init": "init_styletts2",
    },
    "openvoice": {
        "name": "OpenVoice V2",
        "desc": "Tone color conversion, voice cloning",
        "vram": "~4 GB",
        "gpu": "Recommended (CPU possible, 3-4x slower)",
        "cloning": True,
        "languages": "Multi-lingual (via MeloTTS)",
        "package": "git clone https://github.com/myshell-ai/OpenVoice.git",
        "init": "init_openvoice",
    },
    "csm": {
        "name": "Sesame CSM-1B",
        "desc": "Conversational speech model, Llama backbone",
        "vram": "~4.5 GB (CUDA) / ~8.5 GB (CPU)",
        "gpu": "Required (CUDA 12.x)",
        "cloning": True,
        "languages": "English",
        "package": "git clone https://github.com/SesameAILabs/csm.git",
        "init": "init_csm",
    },
}

MODEL_KEYS = list(MODEL_REGISTRY.keys())


FORCE_CPU = "--cpu" in sys.argv or os.environ.get("DOCTOSPEECH_CPU", "").lower() in ("1", "true", "yes")

def _get_device():
    import torch
    if FORCE_CPU:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def show_model_info():
    """Display all available TTS models with their specs."""
    header("Available TTS Models")
    print()
    for i, key in enumerate(MODEL_KEYS, 1):
        m = MODEL_REGISTRY[key]
        clone_tag = f"{C.GREEN}cloning{C.RESET}" if m["cloning"] else f"{C.DIM}preset speakers{C.RESET}"
        print(f"  {C.CYAN}{C.BOLD}{i}.{C.RESET} {C.BOLD}{m['name']}{C.RESET}  {C.DIM}({key}){C.RESET}")
        print(f"     {C.DIM}{m['desc']}{C.RESET}")
        print(f"     VRAM: {C.YELLOW}{m['vram']}{C.RESET}  |  GPU: {m['gpu']}  |  {clone_tag}")
        print(f"     Languages: {C.DIM}{m['languages']}{C.RESET}")
        print(f"     Install: {C.DIM}{m['package']}{C.RESET}")
        print()


def select_model():
    """Prompt user to select a TTS model. Returns (model_key, tts_obj, model_info)."""
    show_model_info()

    while True:
        choice = prompt(f"Select model (1-{len(MODEL_KEYS)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(MODEL_KEYS):
                model_key = MODEL_KEYS[idx]
                break
        except ValueError:
            # Allow typing the key directly
            if choice in MODEL_REGISTRY:
                model_key = choice
                break
        warn(f"Please enter a number between 1 and {len(MODEL_KEYS)}")

    model_info = MODEL_REGISTRY[model_key]
    header(f"Initializing {model_info['name']}")

    device = _get_device()
    if FORCE_CPU:
        info(f"Device: {C.BOLD}{device}{C.RESET} {C.DIM}(forced by --cpu){C.RESET}")
    else:
        info(f"Device: {C.BOLD}{device}{C.RESET}")

    init_fn = _get_init_fn(model_key)
    if init_fn is None:
        error(f"Model '{model_key}' init function not found")
        return None, None, None

    try:
        start = time.time()
        tts_obj = init_fn(model_key, device)
        elapsed = time.time() - start
        success(f"Loaded in {C.BOLD}{elapsed:.1f}s{C.RESET}")
    except Exception as e:
        error(f"Failed to load {model_info['name']}: {e}")
        info(f"Install with: {C.BOLD}{model_info['package']}{C.RESET}")
        return None, None, None

    # Show speakers if available
    if model_info["cloning"]:
        info(f"Voice cloning: {C.GREEN}supported{C.RESET} -- provide a .wav reference")
    else:
        info(f"Voice cloning: {C.RED}not supported{C.RESET} -- use preset speakers")
        speakers = _get_speakers(model_key, tts_obj)
        if speakers:
            info(f"Available speakers ({len(speakers)}):")
            for s in speakers[:20]:
                print(f"    {C.DIM}>{C.RESET} {s}")
            if len(speakers) > 20:
                print(f"    {C.DIM}... and {len(speakers) - 20} more{C.RESET}")

    return model_key, tts_obj, model_info


# ========================= MODEL INIT FUNCTIONS =========================

def _get_init_fn(model_key):
    return {
        "xtts_v2": init_xtts,
        "bark": init_bark,
        "vits": init_vits,
        "yourtts": init_yourtts,
        "chatterbox": init_chatterbox,
        "tortoise": init_tortoise,
        "styletts2": init_styletts2,
        "openvoice": init_openvoice,
        "csm": init_csm,
    }.get(model_key)


def _get_speakers(model_key, tts_obj):
    if model_key in ("xtts_v2", "yourtts"):
        if hasattr(tts_obj, "speakers") and tts_obj.speakers:
            return list(tts_obj.speakers)
    elif model_key == "bark":
        if hasattr(tts_obj, "speaker_lookup"):
            return list(tts_obj.speaker_lookup.keys())
    elif model_key == "vits":
        if hasattr(tts_obj, "speakers") and tts_obj.speakers:
            return list(tts_obj.speakers)
    return []


def init_xtts(model_key, device):
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return tts


def init_bark(model_key, device):
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/bark").to(device)
    return tts


def init_vits(model_key, device):
    from TTS.api import TTS
    tts = TTS("tts_models/en/ljspeech/vits").to(device)
    return tts


def init_yourtts(model_key, device):
    from TTS.api import TTS
    tts = TTS("tts_models/multi-dataset/multi-dataset/your_tts").to(device)
    return tts


def init_chatterbox(model_key, device):
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError:
        raise ImportError(
            "chatterbox-tts not installed. Run: pip install chatterbox-tts"
        )
    model = ChatterboxTTS.from_pretrained(device=device)
    return model


def init_tortoise(model_key, device):
    try:
        from tortoise.api import TextToSpeech
        from tortoise.utils.audio import load_voice
    except ImportError:
        raise ImportError(
            "tortoise-tts not installed. Run: pip install tortoise-tts"
        )
    tts = TextToSpeech()
    return tts


def init_styletts2(model_key, device):
    try:
        from styletts2 import StyleTTS2
    except ImportError:
        raise ImportError(
            "styletts2 not installed. Run: pip install styletts2"
        )
    model = StyleTTS2(device=device)
    return model


def init_openvoice(model_key, device):
    try:
        from openvoice.api import ToneColorConverter
        from melo.api import TTS as MeloTTS
    except ImportError:
        raise ImportError(
            "OpenVoice not installed. Run:\n"
            "  git clone https://github.com/myshell-ai/OpenVoice.git\n"
            "  cd OpenVoice && pip install -e .\n"
            "  pip install git+https://github.com/myshell-ai/MeloTTS.git\n"
            "  python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"myshell-ai/OpenVoiceV2\", local_dir=\"checkpoints_v2\")'"
        )
    ckpt_converter = "checkpoints_v2/converter"
    if not os.path.isdir(ckpt_converter):
        raise FileNotFoundError(
            "OpenVoice checkpoints not found. Run:\n"
            "  python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"myshell-ai/OpenVoiceV2\", local_dir=\"checkpoints_v2\")'"
        )
    tone_color_converter = ToneColorConverter(
        f"{ckpt_converter}/config.json", device=device
    )
    tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
    melo_tts = MeloTTS(language="EN", device=device)
    return {"converter": tone_color_converter, "melo": melo_tts, "device": device}


def init_csm(model_key, device):
    try:
        from transformers import CsmForConditionalGeneration, AutoProcessor
    except ImportError:
        raise ImportError(
            "transformers not installed or too old. Run: pip install transformers>=4.45"
        )
    model_id = "sesame/csm-1b"
    processor = AutoProcessor.from_pretrained(model_id)
    model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)
    return {"model": model, "processor": processor}


# ========================= UNIFIED GENERATE =========================

def make_tts_voiceover(text_to_say, use_cloning, preset_voice, cloning_audio_path, language, tts_obj, output_path, model_key="xtts_v2"):
    """Unified TTS generation dispatching to the correct model."""
    if tts_obj is None:
        error("TTS model not initialized")
        return

    info(f"Generating: {C.BOLD}{output_path}{C.RESET}")
    start = time.time()

    try:
        _dispatch_generate(model_key, tts_obj, text_to_say, output_path, use_cloning, preset_voice, cloning_audio_path, language)
        elapsed = time.time() - start
        success(f"Done in {C.BOLD}{elapsed:.1f}s{C.RESET}")
        preview = text_to_say[:200] + ("..." if len(text_to_say) > 200 else "")
        info(f"Preview: {C.DIM}{preview}{C.RESET}")
    except Exception as e:
        if "CUDA out of memory" in str(e):
            error(f"CUDA out of memory: {e}")
            info("Tip: run with {C.BOLD}--cpu{C.RESET} flag or set {C.BOLD}DOCTOSPEECH_CPU=1{C.RESET}")
        else:
            error(f"TTS generation failed: {e}")
        raise


def _dispatch_generate(model_key, tts_obj, text, output_path, use_cloning, preset_voice, cloning_path, language):
    handlers = {
        "xtts_v2": _gen_xtts,
        "bark": _gen_bark,
        "vits": _gen_vits,
        "yourtts": _gen_yourtts,
        "chatterbox": _gen_chatterbox,
        "tortoise": _gen_tortoise,
        "styletts2": _gen_styletts2,
        "openvoice": _gen_openvoice,
        "csm": _gen_csm,
    }
    fn = handlers.get(model_key)
    if fn:
        fn(tts_obj, text, output_path, use_cloning, preset_voice, cloning_path, language)
    else:
        raise ValueError(f"No generator for model: {model_key}")


def _gen_xtts(tts, text, output_path, use_cloning, preset_voice, cloning_path, language):
    if use_cloning:
        tts.tts_to_file(text=text, speaker_wav=cloning_path, language=language, file_path=output_path)
    else:
        tts.tts_to_file(text=text, speaker=preset_voice, language=language, file_path=output_path)


def _gen_bark(tts, text, output_path, use_cloning, preset_voice, cloning_path, language):
    if use_cloning:
        warn("Bark does not support voice cloning, using speaker preset")
    tts.tts_to_file(text=text, speaker=preset_voice or "v2/en_speaker_6", file_path=output_path)


def _gen_vits(tts, text, output_path, use_cloning, preset_voice, cloning_path, language):
    if use_cloning:
        warn("VITS does not support voice cloning, using default speaker")
    tts.tts_to_file(text=text, file_path=output_path)


def _gen_yourtts(tts, text, output_path, use_cloning, preset_voice, cloning_path, language):
    if use_cloning:
        tts.tts_to_file(text=text, speaker_wav=cloning_path, language=language, file_path=output_path)
    else:
        tts.tts_to_file(text=text, speaker=preset_voice, language=language, file_path=output_path)


def _gen_chatterbox(model, text, output_path, use_cloning, preset_voice, cloning_path, language):
    import torch
    import torchaudio as ta
    import os
    if not cloning_path:
        raise ValueError("Chatterbox needs a reference .wav file (cloning_path)")
    if not os.path.isfile(cloning_path):
        raise FileNotFoundError(f"Reference audio not found: {cloning_path}")
    wav = model.generate(text, audio_prompt_path=cloning_path)
    ta.save(output_path, wav.cpu(), model.sr)


def _gen_tortoise(tts, text, output_path, use_cloning, preset_voice, cloning_path, language):
    import torch
    from tortoise.utils.audio import load_voice, load_voices
    if use_cloning:
        voice_samples, conditioning_latents = load_voice(cloning_path.split("/")[-1].replace(".wav", ""))
    else:
        voice_samples, conditioning_latents = load_voice(preset_voice or "random")
    gen = tts.tts_with_preset(
        text,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        preset="fast",
    )
    import torchaudio
    torchaudio.save(output_path, gen.squeeze(0).cpu(), 24000)


def _gen_styletts2(model, text, output_path, use_cloning, preset_voice, cloning_path, language):
    import torchaudio
    if use_cloning:
        wav = model.inference(text, voice_ref=cloning_path)
    else:
        wav = model.inference(text)
    torchaudio.save(output_path, wav["wav"].unsqueeze(0), wav["sampling_rate"])


def _gen_openvoice(data, text, output_path, use_cloning, preset_voice, cloning_path, language):
    import torch
    from openvoice import se_extractor
    converter = data["converter"]
    melo = data["melo"]
    device = data["device"]

    speaker_ids = melo.hps.data.spk2id
    speaker_key = list(speaker_ids.keys())[0]
    speaker_id = speaker_ids[speaker_key]

    # Generate base speech
    base_path = output_path.replace(".wav", "_base.wav")
    melo.tts_to_file(text, speaker_id, base_path, speed=1.0)

    if use_cloning and cloning_path:
        source_se, _ = se_extractor.get_se(base_path, converter, vad=False)
        target_se, _ = se_extractor.get_se(cloning_path, converter, vad=False)
        converter.convert(
            audio_src_path=base_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
            message="@DocToSpeech",
        )
        os.remove(base_path)
    else:
        os.rename(base_path, output_path)


def _gen_csm(data, text, output_path, use_cloning, preset_voice, cloning_path, language):
    import torch
    import soundfile as sf
    model = data["model"]
    processor = data["processor"]

    if use_cloning and cloning_path:
        import torchaudio
        ref_audio, ref_sr = torchaudio.load(cloning_path)
        if ref_sr != 24000:
            ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 24000)
        conversation = [
            {"role": "speaker", "content": [{"type": "audio", "audio": ref_audio}]},
            {"role": "assistant", "content": [{"type": "text", "text": text}]},
        ]
    else:
        conversation = [
            {"role": "assistant", "content": [{"type": "text", "text": text}]},
        ]

    inputs = processor.apply_chat_template(
        conversation, tokenize=True, return_dict=True
    ).to(model.device)

    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=2048)

    audio_np = audio_values.cpu().numpy().squeeze()
    sf.write(output_path, audio_np, model.config.audio_encoder.sampling_rate)


# ========================= MAIN =========================

if __name__ == "__main__":
    try:
        greet()
        get_user_input_volume()
        goodbye()
    except KeyboardInterrupt:
        print()
        warn("Interrupted")
        sys.exit(1)
    except Exception as e:
        error(f"Fatal: {e}")
        sys.exit(1)
