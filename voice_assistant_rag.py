from voice_generator import VoiceGenerator
import tempfile
import os
import soundfile as sf
import sounddevice as sd
import whisper

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import ChatOllama, OllamaEmbeddings

class VoiceAssistantRAG:
    """
    VoiceAssistantRAG is a voice-enabled Retrieval-Augmented Generation (RAG) system.
    It records user speech, transcribes it, retrieves relevant knowledge, and responds
    using a conversational LLM, optionally converting the response to speech.
    """

    def __init__(self, elevenlabs_api_key):
        """
        Initializes models for audio transcription, LLM, embeddings, and voice generation.

        Args:
            elevenlabs_api_key (str): API key for ElevenLabs voice synthesis.
        """
        self.whisper_model = whisper.load_model("base")  # Load Whisper model for speech-to-text
        self.llm = ChatOllama(model="llama3.2", temperature=0)  # Local LLM via Ollama
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )

        self.vector_store = None       # Will hold the FAISS vector store
        self.qa_chain = None           # Conversational QA chain
        self.sample_rate = 44100       # Audio recording sample rate (Hz)
        self.voice_generator = VoiceGenerator(elevenlabs_api_key)  # TTS generator

    def setup_vector_store(self, vector_store):
        """
        Initializes the vector store and creates a conversational retrieval chain.

        Args:
            vector_store: A FAISS vector store loaded with processed documents.
        """
        self.vector_store = vector_store

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=memory,
            verbose=True,
        )

    def record_audio(self, duration=5):
        """
        Records audio from the default microphone for a given duration.

        Args:
            duration (int): Duration of the recording in seconds.

        Returns:
            numpy.ndarray: Recorded audio samples.
        """
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1
        )
        sd.wait()  # Wait until recording is finished
        return recording

    def transcribe_audio(self, audio_array):
        """
        Transcribes recorded audio to text using Whisper.

        Args:
            audio_array: Numpy array of audio samples.

        Returns:
            str: Transcribed text.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name  # Store file path

        # Write audio data to temporary WAV file
        sf.write(temp_path, audio_array, self.sample_rate)

        try:
            result = self.whisper_model.transcribe(temp_path)
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Warning: Failed to delete temp audio file: {e}")

        return result["text"]

    def generate_response(self, query):
        """
        Uses the QA chain to generate a response to a user query.

        Args:
            query (str): User's question.

        Returns:
            str: Response from the LLM.
        """
        if self.qa_chain is None:
            return "Error: Vector store not initialized"

        response = self.qa_chain.invoke({"question": query})
        return response["answer"]

    def text_to_speech(self, text: str, voice_name: str = None) -> str:
        """
        Converts given text into speech using ElevenLabs voice generation.

        Args:
            text (str): Text to be converted to speech.
            voice_name (str, optional): Voice to be used.

        Returns:
            str: Path to the generated audio file.
        """
        return self.voice_generator.generate_voice_response(text, voice_name)
