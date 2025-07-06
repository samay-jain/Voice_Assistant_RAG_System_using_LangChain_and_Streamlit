# Import necessary modules
from document_processor import DocumentProcessor           # Custom module to handle document loading and processing
from voice_assistant_rag import VoiceAssistantRAG         # Custom module implementing the voice-enabled RAG system
import streamlit as st                                    # UI framework for web app
import tempfile                                            # To create temporary directories for uploaded files
import os                                                  # For file and environment operations
from dotenv import load_dotenv                             # To load environment variables from .env file

# Fix for a known issue in some environments with MKL libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def setup_knowledge_base():
    """
    Handles document upload, processing, and vector store creation.
    Allows users to upload multiple documents and converts them into embeddings
    for use in the Voice Assistant RAG system.
    """
    st.title("Knowledge Base Setup")

    doc_processor = DocumentProcessor()

    # Upload documents in supported formats
    uploaded_files = st.file_uploader(
        "Upload your documents", accept_multiple_files=True, type=["pdf", "txt", "md"]
    )

    # Process uploaded documents when button is clicked
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            temp_dir = tempfile.mkdtemp()  # Create temporary directory

            # Save uploaded files to temporary directory
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

            try:
                # Load and preprocess documents
                documents = doc_processor.load_documents(temp_dir)
                processed_docs = doc_processor.process_documents(documents)

                # Create a vector store from processed document chunks
                vector_store = doc_processor.create_vector_store(
                    processed_docs, "knowledge_base"
                )

                # Store vector store in session state
                st.session_state.vector_store = vector_store

                st.success(f"Processed {len(processed_docs)} document chunks!")

            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

            finally:
                # Clean up temporary directory and files
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)

    return None


def main():
    """
    Main entry point of the Streamlit application.
    Provides a two-page interface:
    1. Knowledge Base Setup
    2. Voice Assistant for querying the knowledge base
    """
    st.set_page_config(page_title="Voice RAG Assistant", layout="wide")

    # Load environment variables (e.g., API keys)
    load_dotenv()
    elevenlabs_api_key = os.getenv("ELEVEN_LABS_API_KEY")

    if not elevenlabs_api_key:
        st.error("Please set ELEVEN_LABS_API_KEY in your environment variables")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Setup Knowledge Base", "Voice Assistant"])

    if page == "Setup Knowledge Base":
        # Call the document processing workflow
        vector_store = setup_knowledge_base()
        if vector_store:
            st.session_state.vector_store = vector_store

    else:
        # Ensure that the knowledge base is setup before proceeding
        if "vector_store" not in st.session_state:
            st.error("Please setup knowledge base first!")
            return

        st.title("Voice Assistant RAG System")

        # Initialize the voice assistant with API key and vector store
        assistant = VoiceAssistantRAG(elevenlabs_api_key)
        assistant.setup_vector_store(st.session_state.vector_store)

        # Voice selection from available options
        try:
            available_voices = assistant.voice_generator.available_voices
            selected_voice = st.sidebar.selectbox(
                "Select Voice",
                available_voices,
                index=available_voices.index("Alice") if "Alice" in available_voices else 0,
            )
        except Exception as e:
            st.error(f"Error loading voices: {e}")
            selected_voice = "Alice"

        # Duration selector for voice recording
        duration = st.sidebar.slider("Recording Duration (seconds)", 1, 10, 5)

        col1, col2 = st.columns(2)

        # Button to start voice recording
        with col1:
            if st.button("Start Recording"):
                with st.spinner(f"Recording for {duration} seconds..."):
                    audio_data = assistant.record_audio(duration)
                    st.session_state.audio_data = audio_data
                    st.success("Recording completed!")

        # Button to process the recorded voice
        with col2:
            if st.button("Process Recording"):
                if "audio_data" not in st.session_state:
                    st.error("Please record audio first!")
                    return

                # Step 1: Transcribe audio to text
                with st.spinner("Transcribing..."):
                    query = assistant.transcribe_audio(st.session_state.audio_data)
                    st.write("You said:", query)

                # Step 2: Generate response using RAG
                with st.spinner("Generating response..."):
                    try:
                        response = assistant.generate_response(query)
                        st.write("Response:", response)
                        st.session_state.last_response = response
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        return

                # Step 3: Convert response to speech and play audio
                with st.spinner("Converting to speech..."):
                    audio_file = assistant.voice_generator.generate_voice_response(
                        response, selected_voice
                    )
                    if audio_file:
                        st.audio(audio_file)
                        os.unlink(audio_file)  # Clean up temp audio file
                    else:
                        st.error("Failed to generate voice response")

        # Optional: Show chat history for the session
        if "chat_history" in st.session_state:
            st.subheader("Chat History")
            for q, a in st.session_state.chat_history:
                st.write("Q:", q)
                st.write("A:", a)
                st.write("---")


# Entry point for the Streamlit app
if __name__ == "__main__":
    main()
