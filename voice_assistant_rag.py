from voice_generator import VoiceGenerator
import tempfile
import os
import soundfile as sf
import sounddevice as sd
import whisper

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

class VoiceAssistantRAG:
    def __init__(self, elevenlabs_api_key):
        self.whisper_model = whisper.load_model("base")
        self.llm = ChatOllama(model="llama3.2", temperature=0)
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text", base_url="http://localhost:11434"
        )
        self.vector_store = None
        self.qa_chain = None
        self.sample_rate = 44100
        self.voice_generator = VoiceGenerator(elevenlabs_api_key)

    def setup_vector_store(self, vector_store):
        """Initialize the vector store and QA chain"""
        self.vector_store = vector_store

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=memory,
            verbose=True,
        )

    def record_audio(self, duration=5):
        """Record audio from microphone"""
        recording = sd.rec(
            int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1
        )
        sd.wait()
        return recording

    def transcribe_audio(self, audio_array):
        """Transcribe audio using Whisper"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name  # Store path before closing the file

        sf.write(temp_path, audio_array, self.sample_rate)  # Write outside 'with' to avoid locking

        try:
            result = self.whisper_model.transcribe(temp_path)
        finally:
            try:
                os.remove(temp_path)  # Safe deletion
            except Exception as e:
                print(f"Warning: Failed to delete temp audio file: {e}")

        return result["text"]


    def generate_response(self, query):
        """Generate response using RAG system"""
        if self.qa_chain is None:
            return "Error: Vector store not initialized"

        response = self.qa_chain.invoke({"question": query})
        return response["answer"]

    def text_to_speech(self, text: str, voice_name: str = None) -> str:
        """Convert text to speech"""
        return self.voice_generator.generate_voice_response(text, voice_name)
    