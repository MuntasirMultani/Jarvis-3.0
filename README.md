# Jarvis-3.0
RAG based chatbot , updated version of Jarvis-2.0

# JarvisV3 - AI Speech-to-Speech RAG Chatbot Assistant 🤖

JarvisV3 is a multilingual AI voice assistant powered by:
- Groq Whisper (Speech-to-Text)
- Groq LLaMA 3.3 70B
- Edge-TTS (Text-to-Speech)
- PDF-based RAG System
- DuckDuckGo Web Search Fallback

The chatbot first checks the college PDF knowledge base and then uses web search if information is not found.

---

# Features

- Speech-to-Speech AI chatbot
- Hindi + English support
- PDF-first RAG retrieval
- Web search fallback
- Wake word support
- Smart silence detection
- Real-time voice interaction

---

# Folder Structure

```bash
JarvisV3/
│
├── jarvisv3.py
├── .env
├── Acropolis_Details.pdf
```

---

# Follow these steps to run

## Step 1: Create a virtual environment

```bash
python -m venv jarvisv3-env
```

---

## Step 2: Activate the virtual environment

### Windows

```bash
jarvisv3-env\Scripts\activate
```

### Linux / Mac

```bash
source jarvisv3-env/bin/activate
```

---

## Step 2.1: Windows PowerShell Fix

If activation is blocked on Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```bash
jarvisv3-env\Scripts\activate
```

---

## Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

# Step 4: Install dependencies

```bash
python -m pip install groq edge-tts pygame sounddevice soundfile numpy python-dotenv pymupdf requests scikit-learn
```

---

# Step 5: Create .env file

Create a `.env` file in the project folder and add:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

# Step 6: Add PDF Knowledge Base

Place your college PDF inside the project folder.

Example:

```text
Acropolis_Details.pdf
```

---

# Step 7: Run the chatbot

```bash
python jarvisv3.py
```

---

# Wake Words

You can activate the chatbot using:

- Hello
- Hey
- Hello Acrobot
- Acrobot

---

# Notes

- Internet connection is required
- First startup may take some time
- Works best with a microphone and speaker
- PDF is used as primary source of information
- Web search is used as secondary fallback
