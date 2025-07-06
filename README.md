# 🎙️ Voice Assistant RAG

> A fully interactive, voice-controlled AI assistant powered by Retrieval-Augmented Generation (RAG), Whisper speech-to-text, and ElevenLabs voice synthesis.

---

![RAG Voice Assistant Demo](https://img.shields.io/badge/LLM-Ollama-blue?style=flat-square)
![Built with Streamlit](https://img.shields.io/badge/Frontend-Streamlit-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)

---

## 🚀 Features

- 🔍 **RAG-based Q&A**: Ask voice questions, get accurate answers sourced from your documents.
- 🗂️ **Knowledge Base Setup**: Upload `.pdf`, `.txt`, or `.md` files and embed them using local FAISS vector store.
- 🎧 **Whisper Integration**: High-quality speech-to-text transcription using OpenAI's Whisper model.
- 🗣️ **ElevenLabs TTS**: Convert AI responses into realistic voice output.
- 🧠 **Chat Memory**: Keeps track of your conversation context.
- 🖥️ **Streamlit UI**: Clean, interactive web interface — no CLI needed.

---

## 🛠️ Tech Stack

| Layer       | Technology             |
|------------|-------------------------|
| Frontend   | Streamlit               |
| LLM        | `llama3.2` via Ollama |
| Embedding  | `nomic-embed-text` via Ollama |
| Vector DB  | FAISS                   |
| STT        | Whisper (`base` model)  |
| TTS        | ElevenLabs              |

---
# Clone the repo and navigate into it
git clone https://github.com/samay-jain/Voice_Assistant_RAG_System_using_LangChain_and_Streamlit

# Create and activate virtual environment
python -m venv venv && source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your ElevenLabs API key in a .env file
echo "ELEVEN_LABS_API_KEY=your_api_key_here" > .env

# Pull required Ollama models
ollama pull llama3.2 && ollama pull nomic-embed-text

# Run the app
streamlit run main.py

