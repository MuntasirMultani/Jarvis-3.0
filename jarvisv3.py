"""
============================================================
  🤖  AcroBot 2.2 — RAG-Powered Speech-to-Speech Chatbot
============================================================
  Stack:
    STT      → Groq Whisper (whisper-large-v3)
    RAG      → PDF chunks → TF-IDF + cosine similarity
    Fallback → DuckDuckGo web search (zero-cost, no API key)
    LLM      → Groq LLaMA 3.3 70B  (context-injected)
    TTS      → Microsoft Edge TTS   (100% free)

  RAG Priority:
    1️⃣  PRIMARY   — PDF knowledge base (AITR_RAG_KnowledgeBase.pdf)
    2️⃣  SECONDARY — Web search (DuckDuckGo) if PDF score < threshold
    3️⃣  FALLBACK  — LLM general knowledge if web also fails

  Language Support:
    English ↔ Hindi — auto-detected every turn

  State Machine:
    IDLE ──(wake word)──► LISTENING ──(speech)──► SPEAKING
      ▲                        │                      │
      └──────(10s silence)─────┘◄─────────────────────┘
============================================================
"""

# ── Standard library ──────────────────────────────────────
import os, re, time, queue, asyncio, tempfile, textwrap
from enum import Enum
from typing import Optional, Tuple, List

# ── Third-party ───────────────────────────────────────────
import numpy as np
import fitz                          # PyMuPDF — PDF text extraction
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sounddevice as sd
import soundfile as sf
from groq import Groq
import edge_tts
import pygame
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════
#  CONFIG — tweak these without touching any logic below
# ══════════════════════════════════════════════════════════

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

STT_MODEL    = "whisper-large-v3"
CHAT_MODEL   = "llama-3.3-70b-versatile"

TTS_VOICE_EN = "en-US-JennyNeural"   # male: en-US-GuyNeural
TTS_VOICE_HI = "hi-IN-SwaraNeural"   # male: hi-IN-MadhurNeural

SAMPLE_RATE  = 16_000
CHANNELS     = 1
MAX_TOKENS   = 300

# ── RAG settings ──────────────────────────────────────────
PDF_PATH        = "Acropolis_Details.pdf"   # put PDF in same folder as this .py
CHUNK_SIZE      = 300        # words per chunk (larger = more context per hit)
CHUNK_OVERLAP   = 50         # overlap between consecutive chunks (prevents boundary misses)
TOP_K           = 3          # how many chunks to pass as context to the LLM
PDF_THRESHOLD   = 0.10       # cosine score below this → skip PDF, go to web
                             # ↑ raise (0.20) to demand a stronger PDF match before using it
                             # ↓ lower (0.05) to almost always prefer PDF

# ── Web fallback settings ─────────────────────────────────
WEB_RESULTS     = 3          # number of DuckDuckGo snippets to fetch
WEB_TIMEOUT     = 5          # seconds before giving up on web call
WEB_KEYWORDS    = [          # if query contains any of these, ALWAYS try web too
    "today", "latest", "current", "now", "2025", "2026",
    "result", "merit list", "cutoff", "ranking",
]

# ── VAD tuning ────────────────────────────────────────────
ENERGY_THRESHOLD     = 0.010
SILENCE_AFTER_SPEECH = 1.5
PRE_ROLL_CHUNKS      = 6
MIN_SPEECH_SECS      = 0.5
CHUNK_SECS           = 0.1
IDLE_TIMEOUT         = 10.0
IDLE_POLL_TIMEOUT    = 30.0

# ── Wake words ────────────────────────────────────────────
WAKE_WORDS = ["hello", "hey", "hello acrobot", "hey acrobot", "acrobot"]

# ── System prompts ────────────────────────────────────────
_BASE_EN = (
    "Your name is AcroBot 2.2. You are the official AI assistant of "
    "Acropolis Institute of Technology and Research (AITR), Indore. "
    "Answer ONLY using the provided CONTEXT. If the context does not "
    "contain enough information, say so politely and direct the user to "
    "call 0731-4730000 or visit aitr.ac.in. "
    "Keep responses concise and conversational. No bullet points or markdown."
)
_BASE_HI = (
    "Aapka naam AcroBot 2.2 hai. Aap Acropolis Institute of Technology and "
    "Research (AITR), Indore ke official AI assistant hain. "
    "Jawab SIRF diye gaye CONTEXT se dein. Agar context mein jaankari nahi hai, "
    "toh politely batayein aur 0731-4730000 ya aitr.ac.in refer karein. "
    "Hamesha Roman/Latin script mein jawab dein — Devanagari bilkul mat use karein. "
    "Uttar chhota aur batcheet ke andaz mein rakhein. Koi bullet points ya markdown nahi."
)

def build_system(lang: str, context: str) -> str:
    base = _BASE_HI if lang == "hi" else _BASE_EN
    if context:
        return f"{base}\n\n--- CONTEXT ---\n{context}\n--- END CONTEXT ---"
    return base


# ══════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════

class State(Enum):
    IDLE      = "idle"
    LISTENING = "listening"
    SPEAKING  = "speaking"


# ══════════════════════════════════════════════════════════
#  RAG ENGINE
# ══════════════════════════════════════════════════════════

class RAGEngine:
    """
    Lightweight RAG over a local PDF.

    Pipeline:
      1. load_pdf()     → extract raw text from all pages via PyMuPDF
      2. _chunk()       → split text into overlapping word windows
      3. _build_index() → fit a TF-IDF matrix over all chunks
      4. retrieve()     → cosine-rank chunks for a query, return top-K + best score
    """

    def __init__(self):
        self.chunks:     List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix      = None          # sparse TF-IDF matrix (n_chunks × vocab)
        self.ready       = False

    # ── Public API ────────────────────────────────────────

    def load_pdf(self, path: str) -> bool:
        """
        Extract text from PDF, chunk it, and build the TF-IDF index.
        Returns True on success, False if the file is missing or empty.
        """
        if not os.path.exists(path):
            print(f"⚠️  RAG: PDF not found at '{path}' — will rely on web/LLM only.")
            return False

        print(f"📄 RAG: Loading '{path}' …", end=" ", flush=True)
        raw = self._extract_text(path)
        if not raw.strip():
            print("empty — skipping.")
            return False

        self.chunks = self._chunk(raw, CHUNK_SIZE, CHUNK_OVERLAP)
        self._build_index()
        self.ready  = True
        print(f"done. {len(self.chunks)} chunks indexed.")
        return True

    def retrieve(self, query: str) -> Tuple[str, float]:
        """
        Find the most relevant chunks for `query`.

        Returns:
            context (str)  — concatenated top-K chunks, ready to paste into prompt
            score   (float)— cosine similarity of the best chunk (0.0–1.0)
        """
        if not self.ready or not self.chunks:
            return "", 0.0

        q_vec  = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()

        # Pick top-K unique chunk indices sorted by score descending
        top_idx   = scores.argsort()[::-1][:TOP_K]
        best_score = float(scores[top_idx[0]])

        context = "\n\n".join(self.chunks[i] for i in top_idx if scores[i] > 0)
        return context, best_score

    # ── Internal helpers ──────────────────────────────────

    @staticmethod
    def _extract_text(path: str) -> str:
        """Use PyMuPDF to pull plain text from every page."""
        doc   = fitz.open(path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages)

    @staticmethod
    def _chunk(text: str, size: int, overlap: int) -> List[str]:
        """
        Split `text` into overlapping word windows.
        overlap ensures that sentences spanning chunk boundaries are captured.
        """
        words  = text.split()
        step   = max(1, size - overlap)
        chunks = []
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _build_index(self):
        """Fit TF-IDF vectorizer and compute the document-term matrix."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams → better phrase matching
            sublinear_tf=True,    # log-scale TF dampens very frequent terms
            min_df=1,
            stop_words="english",
        )
        self.matrix = self.vectorizer.fit_transform(self.chunks)


# ══════════════════════════════════════════════════════════
#  WEB SEARCH FALLBACK  (DuckDuckGo Instant Answer API)
# ══════════════════════════════════════════════════════════

def web_search(query: str) -> str:
    """
    Fetch a brief context string from DuckDuckGo's free Instant Answer API.
    Appends "Acropolis Institute Indore AITR" so results stay on-topic.
    Returns an empty string if the request fails or times out.
    """
    search_query = f"{query} Acropolis Institute Indore AITR"
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q":      search_query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            },
            timeout=WEB_TIMEOUT,
            headers={"User-Agent": "AcroBot/2.2"},
        )
        data     = resp.json()
        snippets = []

        # AbstractText — DuckDuckGo's summarised answer
        if data.get("AbstractText"):
            snippets.append(data["AbstractText"])

        # RelatedTopics — sub-topics with short text
        for topic in data.get("RelatedTopics", [])[:WEB_RESULTS]:
            text = topic.get("Text", "")
            if text:
                snippets.append(text)

        context = " ".join(snippets).strip()
        if context:
            print(f"   🌐 Web context fetched ({len(context)} chars)")
        else:
            print("   🌐 Web search returned no usable snippets.")
        return context

    except Exception as exc:
        print(f"   🌐 Web search failed: {exc}")
        return ""

def needs_web(query: str, score: float) -> bool:
    """
    Decide whether to also query the web.
    True when:
      - PDF score is below threshold (weak match), OR
      - Query explicitly asks for live/recent data
    """
    q = query.lower()
    time_sensitive = any(kw in q for kw in WEB_KEYWORDS)
    return score < PDF_THRESHOLD or time_sensitive


# ══════════════════════════════════════════════════════════
#  GROQ CLIENT  +  CONVERSATION HISTORY
# ══════════════════════════════════════════════════════════

client = Groq(api_key=GROQ_API_KEY)

# Keep per-language histories so the model stays in the right language
history: dict = {"en": [], "hi": []}

def get_ai_reply(user_text: str, lang: str, context: str) -> str:
    """
    Call Groq LLaMA with the injected RAG context.
    The system prompt already contains the retrieved chunks;
    the model is instructed to answer ONLY from that context.
    """
    system        = build_system(lang, context)
    lang_history  = history[lang]

    lang_history.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model    = CHAT_MODEL,
        messages = [
            {"role": "system", "content": system},
            *lang_history,
        ],
        max_tokens  = MAX_TOKENS,
        temperature = 0.4,   # lower temp → more factual, less creative
    )

    reply = response.choices[0].message.content.strip()
    lang_history.append({"role": "assistant", "content": reply})
    return reply


# ══════════════════════════════════════════════════════════
#  CONTEXT BUILDER  — glues PDF + web together
# ══════════════════════════════════════════════════════════

def build_context(query: str, rag: RAGEngine) -> Tuple[str, str]:
    """
    Returns (context_string, source_label) where source_label is one of:
        "PDF"  "PDF+Web"  "Web"  "None"

    Priority logic:
        1. Always try PDF first.
        2. If PDF score < PDF_THRESHOLD OR query is time-sensitive → also try web.
        3. Merge PDF and web contexts (PDF first so LLM trusts it more).
        4. If both fail → return empty context (LLM will politely say it doesn't know).
    """
    pdf_context, pdf_score = rag.retrieve(query)
    print(f"   📄 PDF score: {pdf_score:.3f}  (threshold={PDF_THRESHOLD})")

    web_context = ""
    source      = "None"

    if pdf_context and pdf_score >= PDF_THRESHOLD:
        source = "PDF"

    if needs_web(query, pdf_score):
        web_context = web_search(query)
        if web_context:
            source = "PDF+Web" if pdf_context else "Web"

    # Merge: PDF context comes first (higher trust), web appended as supplementary
    parts = []
    if pdf_context:
        parts.append(f"[From AITR Knowledge Base]\n{pdf_context}")
    if web_context:
        parts.append(f"[From Web]\n{web_context}")

    return "\n\n".join(parts), source


# ══════════════════════════════════════════════════════════
#  VAD RECORDING
# ══════════════════════════════════════════════════════════

def capture_speech(timeout: float) -> Optional[np.ndarray]:
    """
    Listens via microphone using energy-based Voice Activity Detection.

    Returns:
        np.ndarray  — audio when speech ends (silence >= SILENCE_AFTER_SPEECH s)
        None        — if no speech detected within `timeout` seconds
    """
    audio_q   = queue.Queue()
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

    speech_buffer: list            = []
    pre_buffer:    list            = []
    recording                      = False
    silence_start: Optional[float] = None
    idle_clock                     = time.time()

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
                idle_clock    = time.time()
                silence_start = None
                if not recording:
                    recording     = True
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


# ══════════════════════════════════════════════════════════
#  TRANSCRIBE  (3-layer language detection)
# ══════════════════════════════════════════════════════════

def transcribe(audio: np.ndarray) -> Tuple[str, str]:
    """
    Returns (transcript, lang_code) where lang_code ∈ {'en', 'hi'}.

    Layer 1: Whisper's own language tag (fast, occasionally wrong).
    Layer 2: Script scan — Devanagari/Arabic code-points → force 'hi'.
    Layer 3: Default to 'en'.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
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
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            lang = "hi"
            break

    return text, lang


# ══════════════════════════════════════════════════════════
#  WAKE WORD
# ══════════════════════════════════════════════════════════

def is_wake_word(text: str) -> bool:
    lower = text.lower().strip()
    return any(w in lower for w in WAKE_WORDS)


# ══════════════════════════════════════════════════════════
#  TTS
# ══════════════════════════════════════════════════════════

def pick_voice(text: str, lang: str) -> str:
    if lang == "hi":
        return TTS_VOICE_HI
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            return TTS_VOICE_HI
    return TTS_VOICE_EN


async def _tts(text: str, path: str, voice: str):
    await edge_tts.Communicate(text, voice=voice).save(path)


def speak(text: str, lang: str = "en"):
    voice = pick_voice(text, lang)
    print(f"   🔊 [{voice}] {textwrap.shorten(text, 80)}")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    asyncio.run(_tts(text, tmp_path, voice))
    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)
    pygame.mixer.music.unload()
    os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════

def print_banner(rag_ready: bool):
    status = "✅ PDF loaded" if rag_ready else "⚠️  PDF not found — web-only mode"
    print("\n" + "=" * 60)
    print("  AcroBot 2.2 🤖  |  Acropolis Institute, Indore")
    print("=" * 60)
    print(f"  RAG status : {status}")
    print(f"  PDF path   : {PDF_PATH}")
    print(f"  PDF thresh : {PDF_THRESHOLD}  (score below → web fallback)")
    print( "  States     :")
    print( "    👂 LISTENING  — auto-detects your voice")
    print(f"    😴 IDLE       — {int(IDLE_TIMEOUT)}s silence → idle")
    print( "    🔊 SPEAKING   — playing response")
    print( "  Ctrl+C to quit")
    print("=" * 60 + "\n")


def state_label(state: State) -> str:
    return {
        State.IDLE:      "😴 IDLE",
        State.LISTENING: "👂 LISTENING",
        State.SPEAKING:  "🔊 SPEAKING",
    }[state]


# ══════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════

def main():
    # ── Initialise pygame audio ───────────────────────────
    pygame.mixer.init()

    # ── Boot RAG engine ───────────────────────────────────
    rag = RAGEngine()
    rag.load_pdf(PDF_PATH)

    print_banner(rag.ready)

    state = State.LISTENING
    reply = ""
    lang  = "hi"

    # ── Opening greeting ──────────────────────────────────
    speak(
        "Namaste! Mein AcroBot 2.2 hu, Acropolis Institute ka AI Assistant. "
        "Aap admission, courses, departments, fees aur college se judi "
        "koi bhi jaankari poochh sakte hain. Mein aapki kaise madad kar sakta hu?",
        lang="hi",
    )

    try:
        while True:

            # ──────────────────────────────────────────────
            #  IDLE — wait silently for a wake word
            # ──────────────────────────────────────────────
            if state == State.IDLE:
                print(f"\n{state_label(state)}  — say 'Hello' to wake me up …")

                audio = capture_speech(timeout=IDLE_POLL_TIMEOUT)
                if audio is None:
                    continue

                print("🔍 Checking for wake word …")
                wake_text, _ = transcribe(audio)
                print(f"   Heard: {wake_text!r}")

                if is_wake_word(wake_text):
                    state = State.LISTENING
                    print("\n✅ Wake word detected!")
                    speak("Haan, mein sun raha hoon. Aap apna sawaal poochhiye.", lang="hi")
                else:
                    print("   Not a wake word — staying idle.")
                continue

            # ──────────────────────────────────────────────
            #  LISTENING — VAD; 10 s silence → IDLE
            # ──────────────────────────────────────────────
            if state == State.LISTENING:
                print(f"\n{state_label(state)}  "
                      f"— {int(IDLE_TIMEOUT)}s of silence → idle")

                audio = capture_speech(timeout=IDLE_TIMEOUT)

                if audio is None:
                    state = State.IDLE
                    print(f"\n⏱️  No speech — going idle.")
                    speak(
                        "Mein idle mode mein ja raha hoon. "
                        "Jab zaroorat ho 'Hello' kahiye.",
                        lang="hi",
                    )
                    continue

                # ── Transcribe ────────────────────────────
                print("🎙️  Transcribing …")
                user_text, lang = transcribe(audio)

                if not user_text:
                    print("⚠️  Could not understand — listening again.")
                    continue

                print(f"\n   You [{lang.upper()}] › {user_text}")

                # ── RAG: retrieve context ─────────────────
                print("🔎 Retrieving context …")
                context, source = build_context(user_text, rag)
                print(f"   📚 Source used: [{source}]")
                if context:
                    preview = context[:120].replace("\n", " ")
                    print(f"   Context preview: {preview} …")

                # ── LLM ───────────────────────────────────
                print("🤔 Generating reply …")
                reply = get_ai_reply(user_text, lang, context)
                print(f"   AI [{lang.upper()}] › {reply}")

                state = State.SPEAKING
                continue

            # ──────────────────────────────────────────────
            #  SPEAKING — play reply → back to LISTENING
            # ──────────────────────────────────────────────
            if state == State.SPEAKING:
                print(f"\n{state_label(state)}")
                speak(reply, lang)
                state = State.LISTENING
                continue

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down AcroBot 2.2 …")


if __name__ == "__main__":
    main()
