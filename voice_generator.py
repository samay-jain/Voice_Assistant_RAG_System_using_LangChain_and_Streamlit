import tempfile
from elevenlabs.client import ElevenLabs

class VoiceGenerator:
    """
    A class to generate speech audio from text using the ElevenLabs API.
    """

    def __init__(self, api_key):
        """
        Initializes the ElevenLabs client and sets up available voices.
        
        Args:
            api_key (str): Your ElevenLabs API key.
        """
        self.client = ElevenLabs(api_key=api_key)

        # Mapping of friendly voice names to their actual ElevenLabs voice IDs
        self.voice_map = {
            "Alice": "Xb7hH8MSUJpSbSDYk0k2",
            "Aria": "9BWtsMINqrJLrRacOk9x",
            "Bill": "pqHfZKP75CvOlQylNhV4",
            "Brian": "nPczCjzI2devNBz1zQrb",
            "Callum": "N2lVS1w4EtoT3dr4eOWO"
        }

        # List of available voice names for UI selection or defaulting
        self.available_voices = list(self.voice_map.keys())
        self.default_voice = "Alice"

    def generate_voice_response(self, text: str, voice_name: str = None) -> str:
        """
        Converts the given text into speech using a selected voice and saves it to a temporary MP3 file.
        
        Args:
            text (str): The text to be converted into speech.
            voice_name (str, optional): The name of the voice to use. Defaults to None (uses default voice).
        
        Returns:
            str: Path to the generated temporary MP3 file, or None if generation fails.
        """
        try:
            # Use provided voice or fall back to default
            selected_voice = voice_name or self.default_voice
            voice_id = self.voice_map.get(selected_voice)

            # Check if the voice exists in the map
            if not voice_id:
                raise ValueError(f"Voice '{selected_voice}' not found in voice map.")

            # Generate audio as a byte-stream generator from ElevenLabs API
            audio_generator = self.client.text_to_speech.convert(
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                text=text
            )

            # Concatenate all audio chunks into a single byte array
            audio_bytes = b"".join(audio_generator)

            # Write the audio bytes to a temporary .mp3 file and return the path
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                return temp_audio.name

        except Exception as e:
            # Log and return None on error
            print(f"Error generating voice response: {e}")
            return None
