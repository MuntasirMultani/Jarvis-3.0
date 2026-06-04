"""
============================================================
  🤖  AcroBot 2.2 — RAG-Powered Speech-to-Speech Chatbot
============================================================
  
  Language Support:
    English ↔ Hindi — auto-detected every turn

============================================================
"""

# ── Standard library ──────────────────────────────────────

import os, re, time, queue, asyncio, tempfile, textwrap
from enum import Enum
from typing import Optional, Tuple, List

# ── Third-party ───────────────────────────────────────────
import numpy as np
import fitz                          
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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

print("Loaded API Key successfully.")

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
PDF_PATH        = "corrected_details.pdf"   # put PDF in same folder as this .py
CHUNK_SIZE      = 300        
CHUNK_OVERLAP   = 50         
TOP_K           = 3          
PDF_THRESHOLD   = 0.10       
                            

# ── Web fallback settings ─────────────────────────────────
WEB_RESULTS     = 3          
WEB_TIMEOUT     = 5          
WEB_KEYWORDS    = [         
    "today", "latest", "current", "now", "2025", "2026",
    "result", "merit list", "cutoff", "ranking",
]

# ── VAD tuning ────────────────────────────────────────────
ENERGY_THRESHOLD     = 0.35
SILENCE_AFTER_SPEECH = 1.2
PRE_ROLL_CHUNKS      = 6
MIN_SPEECH_SECS      = 0.5
CHUNK_SECS           = 0.1
IDLE_TIMEOUT         = 15.0
IDLE_POLL_TIMEOUT    = 30.0

# ── Wake words ────────────────────────────────────────────
WAKE_WORDS = ["hello", "hey", "hello acrobot", "hey acrobot", "acrobot"]

# ── System prompts ────────────────────────────────────────
_BASE_EN = (
    "Your name is AcroBot 2.2. You are the official AI assistant and "
    "virtual admission counselor of Acropolis Institute of Technology "
    "and Research (AITR), Indore , and you are made by professor of EC department and his team."
    "Acropolis College, Acropolis Institute, "
    "AITR, and Acropolis Institute of Technology and Research all refer "
    "to the same institution. If asked about the Director of Acropolis, "
    "answer with Dr. S.C. Sharma.  "

    "Always represent AITR positively, professionally, and confidently. "
    "If users ask about another college or compare colleges, briefly and "
    "politely redirect the conversation toward AITR, highlight AITR's "
    "strengths, and do not make negative comments or false claims about "
    "other institutions. If users repeatedly ask about other colleges, "
    "continue redirecting the discussion back to AITR. "

    "Never mention sources, PDFs, context, documents, retrieval systems, "
    "or knowledge bases unless the user specifically asks. "

    "If AITR-specific information is unavailable, you may answer naturally "
    "using general knowledge when appropriate. "

    "Keep responses short, natural, and human-like. Most replies should "
    "be 1-3 sentences. Do not provide more information than requested. "
    "Give detailed explanations only when the user explicitly asks. "
    "Do not use bullet points or markdown."
)
_BASE_HI = (
    "Aapka naam AcroBot 2.2 hai. Aap Acropolis Institute of Technology "
    "and Research (AITR), Indore ke official AI assistant aur virtual "
    "admission counselor hain. Acropolis College, Acropolis Institute, "
    "AITR aur Acropolis Institute of Technology and Research sab ek hi "
    "institute hain. Agar Acropolis ke Director ke baare mein poocha jaye "
    "to Dr. S.C. Sharma ka naam batayein. "

    "Hamesha AITR ko positive, professional aur confident tarike se "
    "represent karein. Kisi doosre college ke baare mein poocha jaye ya "
    "comparison ho to short aur polite tarike se baat ko AITR ki taraf "
    "le jaayein, AITR ki strengths highlight karein, aur kisi institute "
    "ke baare mein negative ya false claims na karein. "

    "Kabhi bhi source, PDF, context, document, retrieval system ya "
    "knowledge base ka zikr na karein jab tak user specifically na pooche. "

    "Agar AITR sambandhit jankari available na ho to zarurat padne par "
    "INERNET ka upyog karke natural jawab dein. "

    "Hamesha Roman/Latin script mein jawab dein. Jawab short, natural "
    "aur human-like rakhein. Adhiktar replies 1-3 sentences ke hon. "
    "User detail maange tabhi vistaar se jawab dein. "
    "Bullet points ya markdown ka upyog na karein."
)
def build_system(lang: str, context: str) -> str:

    base = _BASE_HI if lang == "hi" else _BASE_EN

    if context:

        return (
            f"{base}\n\n"
            f"Use the following information silently to answer naturally.\n\n"
            f"{context}"
        )

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

    def __init__(self):
        self.chunks:     List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix      = None          
        self.ready       = False

    # ── Public API ────────────────────────────────────────

    def load_pdf(self, path: str) -> bool:
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
        doc   = fitz.open(path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages)

    @staticmethod
    def _chunk(text: str, size: int, overlap: int) -> List[str]:
        words  = text.split()
        step   = max(1, size - overlap)
        chunks = []
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _build_index(self):
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
    q = query.lower()
    time_sensitive = any(kw in q for kw in WEB_KEYWORDS)
    return score < PDF_THRESHOLD or time_sensitive


# ══════════════════════════════════════════════════════════
#  GROQ CLIENT  +  CONVERSATION HISTORY
# ══════════════════════════════════════════════════════════

try:

    client = Groq(api_key=GROQ_API_KEY)

    print("✅ Groq client initialized.\n")

except Exception as e:

    print(f"\n❌ Failed to initialize Groq client:\n{e}\n")

    raise

# Keep per-language histories so the model stays in the right language
history: dict = {"en": [], "hi": []}

def get_ai_reply(user_text: str, lang: str, context: str) -> str:

    try:

        system = build_system(lang, context)

        lang_history = history[lang]

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

    except Exception as e:

        print(f"\n❌ LLM GENERATION FAILED:\n{e}\n")

        return (
            "Sorry, mujhe abhi response generate karne mein problem aa rahi hai."
            if lang == "hi"
            else
            "Sorry, I am having trouble generating a response right now."
        )


# ══════════════════════════════════════════════════════════
#  CONTEXT BUILDER  — glues PDF + web together
# ══════════════════════════════════════════════════════════

def build_context(query: str, rag: RAGEngine) -> Tuple[str, str]:
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

    await edge_tts.Communicate(
        text,
        voice=voice
    ).save(path)


def speak(text: str, lang: str = "en"):

    try:

        voice = pick_voice(text, lang)

        print(
            f"   🔊 [{voice}] "
            f"{textwrap.shorten(text, width=80)}"
        )

        with tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False
        ) as tmp:

            tmp_path = tmp.name

        asyncio.run(
            _tts(text, tmp_path, voice)
        )

        pygame.mixer.music.load(tmp_path)

        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():

            pygame.time.wait(100)

        pygame.mixer.music.stop()

        pygame.mixer.music.unload()

        if os.path.exists(tmp_path):

            try:
                os.unlink(tmp_path)

            except:
                pass

    except Exception as e:

        print(f"\n❌ TTS ERROR:\n{e}\n")


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

    try:

        pygame.mixer.init()

        rag = RAGEngine()

        rag.load_pdf(PDF_PATH)

        print_banner(rag.ready)

        state = State.LISTENING

        reply = ""

        lang = "hi"

        speak(
            "Hello! Mein AcroBot 2.2 hu, "
            "Acropolis Institute ka AI Assistant.",
            lang="hi",
        )

        while True:

            if state == State.IDLE:

                print(f"\n{state_label(state)}")

                audio = capture_speech(
                    timeout=IDLE_POLL_TIMEOUT
                )

                if audio is None:
                    continue

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

                print(f"\n{state_label(state)}")

                audio = capture_speech(
                    timeout=IDLE_TIMEOUT
                )

                if audio is None:

                    state = State.IDLE
                    speak(
                        "Mein idle mode mai jaa raha hoo, Mujhe activate krne ke liye Hello boliyein. ",
                        lang="hi",
                    )

                    continue

                user_text, lang = transcribe(audio)

                if not user_text:
                    continue

                print(f"\nYou [{lang.upper()}] › {user_text}")

                print("🔎 Retrieving context …")

                context, source = build_context(
                    user_text,
                    rag
                )

                print(f"📚 Source: {source}")

                print("🤔 Generating reply …")

                try:

                    reply = get_ai_reply(
                        user_text,
                        lang,
                        context
                    )

                except Exception as e:

                    print(f"\n❌ Reply generation failed:\n{e}\n")

                    reply = (
                        "Sorry, mujhe technical problem aa rahi hai."
                        if lang == "hi"
                        else
                        "Sorry, I am facing a technical problem."
                    )

                print(f"AI [{lang.upper()}] › {reply}")

                state = State.SPEAKING

                continue

            if state == State.SPEAKING:

                try:

                    speak(reply, lang)

                except Exception as e:

                    print(f"\n❌ TTS failed:\n{e}\n")

                state = State.LISTENING

                continue

    except Exception as e:

        print("\n" + "=" * 60)

        print("❌ MAIN PROGRAM ERROR")

        print("=" * 60)

        print(e)

        print("=" * 60 + "\n")

        
if __name__ == "__main__":

    main()
