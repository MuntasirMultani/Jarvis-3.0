"""
============================================================
  🤖 Speech-to-Speech AI Chatbot — Powered by Groq (Free)
============================================================
  Stack:
    STT  → Groq Whisper (whisper-large-v3)
    LLM  → Groq LLaMA 3.3 70B
    TTS  → Microsoft Edge TTS (edge-tts, 100% free)

  Language Support:
    → Speak English → AcroBot replies & speaks in English
    → Speak Hindi   → AcroBot replies & speaks in Hindi
    → Switches instantly every message — no confusion

  Language Detection (3-layer):
    1. Whisper language tag  (fast, sometimes wrong)
    2. Script scan of transcript  (ground truth — never lies)
    3. Default → English

  State Machine:
    IDLE  ──(wake word)──►  LISTENING  ──(speech)──►  SPEAKING
      ▲                          │                        │
      └────────(10s silence)─────┘◄───────────────────────┘

  Wake word: "Hello" (or just "Acrobot")
============================================================
"""

import os
import asyncio
import tempfile
import queue
import time
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
from groq import Groq
import edge_tts
import pygame
from dotenv import load_dotenv

import fitz
import faiss
from sentence_transformers import SentenceTransformer
print("Program started...", flush=True)

print("Loading environment variables...", flush=True)

load_dotenv()

print("Environment variables loaded.", flush=True)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

STT_MODEL = "whisper-large-v3"
CHAT_MODEL = "llama-3.3-70b-versatile"

TTS_VOICE_EN = "en-US-JennyNeural"
TTS_VOICE_HI = "hi-IN-SwaraNeural"

SAMPLE_RATE = 16000
CHANNELS = 1
MAX_TOKENS = 350

ENERGY_THRESHOLD = 0.010
SILENCE_AFTER_SPEECH = 1.5
PRE_ROLL_CHUNKS = 6
MIN_SPEECH_SECS = 0.5
CHUNK_SECS = 0.1

IDLE_TIMEOUT = 10.0
IDLE_POLL_TIMEOUT = 30.0

WAKE_WORDS = [
    "hello",
    "hey",
    "hello acrobot",
    "hey acrobot",
    "acrobot"
]

PDF_PATH = "Acropolis_Details.pdf"

SYSTEM_EN = (
    "You are AcroBot 2.2, the official AI assistant of "
    "Acropolis Institute of Technology and Research (AITR), Indore. "
    "Always prioritize information from the provided college knowledge base. "
    "If information is unavailable in the knowledge base, then use general knowledge carefully. "
    "Keep responses short, conversational, professional and accurate. "
    "No markdown or bullet points."
)

SYSTEM_HI = (
    "Aap AcroBot 2.2 hain, jo Acropolis Institute of Technology and Research "
    "(AITR), Indore ke official AI assistant hain. "
    "Hamesha pehle knowledge base ki information ka use karein. "
    "Agar information knowledge base mein available nahi ho tabhi general knowledge use karein. "
    "Jawab concise, professional aur conversational rakhein. "
    "Hamesha Roman Hindi mein jawab dein. "
    "Koi markdown ya bullet points nahi."
)

# ──────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────

class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"

# ──────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────

client = Groq(api_key=GROQ_API_KEY)

history = {
    "en": [],
    "hi": [],
}

print("Initializing pygame mixer...", flush=True)

pygame.mixer.init()

print("Pygame initialized successfully.", flush=True)

# ──────────────────────────────────────────────
# PDF RAG INITIALIZATION
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# PDF RAG INITIALIZATION
# ──────────────────────────────────────────────

print("\n[1/7] Starting AcroBot RAG System...", flush=True)

if not os.path.exists(PDF_PATH):

    print(f"\n❌ ERROR: PDF file not found -> {PDF_PATH}", flush=True)
    print("Place the PDF inside the same folder as this .py file.\n", flush=True)

    raise FileNotFoundError(PDF_PATH)

print("[2/7] PDF file detected successfully.", flush=True)

print("[3/7] Opening PDF document...", flush=True)

try:

    doc = fitz.open(PDF_PATH)

except Exception as e:

    print(f"\n❌ Failed to open PDF: {e}\n", flush=True)

    raise

print("[4/7] Extracting text from PDF...", flush=True)

pdf_text = ""

try:

    for page_num, page in enumerate(doc):

        print(f"   Reading page {page_num + 1}...", flush=True)

        page_text = page.get_text()

        if page_text:
            pdf_text += page_text

except Exception as e:

    print(f"\n❌ PDF text extraction failed: {e}\n", flush=True)

    raise

if not pdf_text.strip():

    print("\n❌ ERROR: No text found inside PDF.\n", flush=True)

    raise Exception("PDF text extraction returned empty data.")

print("[5/7] Creating text chunks...", flush=True)

chunk_size = 700

chunks = [
    pdf_text[i:i + chunk_size]
    for i in range(0, len(pdf_text), chunk_size)
]

print(f"   Total chunks created: {len(chunks)}", flush=True)

print("[6/7] Loading SentenceTransformer model...", flush=True)
print("   First launch may take 1-3 minutes.\n", flush=True)

try:

    embedder = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

except Exception as e:

    print(f"\n❌ Embedding model failed to load: {e}\n", flush=True)

    raise

print("✅ Embedding model loaded successfully.", flush=True)

print("[7/7] Creating FAISS vector database...", flush=True)

try:

    embeddings = embedder.encode(
        chunks,
        show_progress_bar=True
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

except Exception as e:

    print(f"\n❌ FAISS initialization failed: {e}\n", flush=True)

    raise

print("\n✅ Knowledge Base Loaded Successfully.", flush=True)
print("✅ AcroBot RAG System Ready.\n", flush=True)

# ──────────────────────────────────────────────
# RETRIEVAL
# ──────────────────────────────────────────────

def retrieve_context(query, k=4):

    query_embedding = embedder.encode([query])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        k
    )

    retrieved_chunks = [
        chunks[i]
        for i in indices[0]
    ]

    return "\n\n".join(retrieved_chunks)

# ──────────────────────────────────────────────
# VAD RECORDING
# ──────────────────────────────────────────────

def capture_speech(timeout: float) -> Optional[np.ndarray]:

    audio_q = queue.Queue()
    blocksize = int(SAMPLE_RATE * CHUNK_SECS)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    )

    stream.start()

    speech_buffer = []
    pre_buffer = []

    recording = False
    silence_start = None
    idle_clock = time.time()

    try:
        while True:

            try:
                chunk = audio_q.get(timeout=0.5)

            except queue.Empty:

                if not recording and time.time() - idle_clock >= timeout:
                    return None

                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= ENERGY_THRESHOLD:

                idle_clock = time.time()
                silence_start = None

                if not recording:
                    recording = True
                    speech_buffer = list(pre_buffer)

                speech_buffer.append(chunk)

            elif recording:

                speech_buffer.append(chunk)

                if silence_start is None:
                    silence_start = time.time()

                elif time.time() - silence_start >= SILENCE_AFTER_SPEECH:
                    break

            else:

                pre_buffer.append(chunk)

                if len(pre_buffer) > PRE_ROLL_CHUNKS:
                    pre_buffer.pop(0)

                if time.time() - idle_clock >= timeout:
                    return None

    finally:
        stream.stop()
        stream.close()

    if not speech_buffer:
        return None

    audio = np.concatenate(speech_buffer, axis=0)

    return audio if len(audio) >= SAMPLE_RATE * MIN_SPEECH_SECS else None

# ──────────────────────────────────────────────
# TRANSCRIBE
# ──────────────────────────────────────────────

def transcribe(audio: np.ndarray) -> Tuple[str, str]:

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as tmp:

        tmp_path = tmp.name

    sf.write(tmp_path, audio, SAMPLE_RATE)

    with open(tmp_path, "rb") as f:

        result = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f,
            response_format="verbose_json",
        )

    os.unlink(tmp_path)

    text = (result.text or "").strip()

    lang = (result.language or "en").strip().lower()

    if lang == "ur":
        lang = "hi"

    if lang not in ("hi", "en"):
        lang = "en"

    for ch in text:

        cp = ord(ch)

        if 0x0900 <= cp <= 0x097F:
            lang = "hi"
            break

        if 0x0600 <= cp <= 0x06FF:
            lang = "hi"
            break

    return text, lang

# ──────────────────────────────────────────────
# WAKE WORD
# ──────────────────────────────────────────────

def is_wake_word(text: str) -> bool:

    lower = text.lower().strip()

    return any(w in lower for w in WAKE_WORDS)

# ──────────────────────────────────────────────
# AI REPLY WITH RAG
# ──────────────────────────────────────────────

def get_ai_reply(user_text: str, lang: str) -> str:

    system = SYSTEM_HI if lang == "hi" else SYSTEM_EN

    lang_history = history[lang]

    pdf_context = retrieve_context(user_text)

    rag_prompt = f"""
PRIMARY SOURCE:
Use the following AITR knowledge base context first.

If the answer exists in the context,
answer ONLY using that information.

If the answer is unavailable,
then use your general knowledge carefully.

AITR KNOWLEDGE BASE:
{pdf_context}

USER QUESTION:
{user_text}
"""

    lang_history.append({
        "role": "user",
        "content": user_text
    })

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": system
            },
            {
                "role": "system",
                "content": rag_prompt
            },
            *lang_history,
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.4,
    )

    reply = response.choices[0].message.content.strip()

    lang_history.append({
        "role": "assistant",
        "content": reply
    })

    return reply

# ──────────────────────────────────────────────
# VOICE SELECTION
# ──────────────────────────────────────────────

def pick_voice(text: str, lang: str) -> str:

    if lang == "hi":
        return TTS_VOICE_HI

    for ch in text:

        cp = ord(ch)

        if 0x0900 <= cp <= 0x097F:
            return TTS_VOICE_HI

        if 0x0600 <= cp <= 0x06FF:
            return TTS_VOICE_HI

    return TTS_VOICE_EN

# ──────────────────────────────────────────────
# SPEAK
# ──────────────────────────────────────────────

async def _tts(text: str, path: str, voice: str):

    await edge_tts.Communicate(
        text,
        voice=voice
    ).save(path)

def speak(text: str, lang: str = "en"):

    voice = pick_voice(text, lang)

    print(f"🔊 Voice → {voice}")

    with tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False
    ) as tmp:

        tmp_path = tmp.name

    asyncio.run(_tts(text, tmp_path, voice))

    pygame.mixer.music.load(tmp_path)

    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)

    pygame.mixer.music.unload()

    os.unlink(tmp_path)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def print_banner():

    print("\n" + "=" * 56)
    print("  AcroBot 2.2 🤖 | AITR RAG Assistant")
    print("=" * 56)
    print("  PDF-first Retrieval System Enabled")
    print("  Ctrl+C to quit")
    print("=" * 56 + "\n")

def state_label(state: State) -> str:

    return {
        State.IDLE: "😴 IDLE",
        State.LISTENING: "👂 LISTENING",
        State.SPEAKING: "🔊 SPEAKING",
    }[state]

# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────

def main():

    print_banner()

    state = State.LISTENING

    reply = ""

    lang = "hi"

    speak(
        "mein AcroBot 2.2 hu, "
        "Acropolis Institute ka AI Assistant. "
        "Aap apna sawaal poochhiye.",
        lang="hi",
    )

    try:

        while True:

            if state == State.IDLE:

                print(
                    f"\n{state_label(state)} "
                    f"— say Hello to activate"
                )

                audio = capture_speech(
                    timeout=IDLE_POLL_TIMEOUT
                )

                if audio is None:
                    continue

                print("Checking wake word...")

                wake_text, _ = transcribe(audio)

                print(f"Heard: {wake_text}")

                if is_wake_word(wake_text):

                    state = State.LISTENING

                    speak(
                        "Haan, mein sun raha hoon.",
                        lang="hi",
                    )

                continue

            if state == State.LISTENING:

                print(
                    f"\n{state_label(state)}"
                )

                audio = capture_speech(
                    timeout=IDLE_TIMEOUT
                )

                if audio is None:

                    state = State.IDLE

                    speak(
                        "Main idle mode mein ja raha hoon. "
                        "Mujhe activate karne ke liye Hello kahiye.",
                        lang="hi",
                    )

                    continue

                print("Transcribing...")

                user_text, lang = transcribe(audio)

                if not user_text:
                    continue

                print(
                    f"You [{lang.upper()}] › {user_text}"
                )

                print("Thinking...")

                reply = get_ai_reply(
                    user_text,
                    lang
                )

                print(
                    f"AI [{lang.upper()}] › {reply}"
                )

                state = State.SPEAKING

                continue

            if state == State.SPEAKING:

                speak(reply, lang)

                state = State.LISTENING

                continue

    except KeyboardInterrupt:

        print("\nShutting down...")

if __name__ == "__main__":
    main()