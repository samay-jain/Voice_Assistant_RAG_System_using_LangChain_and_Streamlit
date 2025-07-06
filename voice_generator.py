import tempfile
from elevenlabs.client import ElevenLabs

class VoiceGenerator:
    def __init__(self, api_key):
        self.client = ElevenLabs(api_key=api_key)

        # Map display names to real voice IDs
        self.voice_map = {
            "Alice": "Xb7hH8MSUJpSbSDYk0k2",
            "Aria": "9BWtsMINqrJLrRacOk9x",
            "Bill": "pqHfZKP75CvOlQylNhV4",
            "Brian": "nPczCjzI2devNBz1zQrb",
            "Callum": "N2lVS1w4EtoT3dr4eOWO"
        }

        self.available_voices = list(self.voice_map.keys())
        self.default_voice = "Alice"

    def generate_voice_response(self, text: str, voice_name: str = None) -> str:
        try:
            selected_voice = voice_name or self.default_voice
            voice_id = self.voice_map.get(selected_voice)

            if not voice_id:
                raise ValueError(f"Voice '{selected_voice}' not found in voice map.")

            # Get audio as generator
            audio_generator = self.client.text_to_speech.convert(
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                text=text
            )

            # Convert generator to bytes
            audio_bytes = b"".join(audio_generator)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                return temp_audio.name

        except Exception as e:
            print(f"Error generating voice response: {e}")
            return None

