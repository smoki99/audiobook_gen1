#!/usr/bin/env python3
"""
Fish Speech TTS API Client
Einfache Klasse zum Generieren von deutscher Sprache mit Fish Speech
"""

import os
import requests
import ormsgpack
from pathlib import Path
from fish_speech.utils.file import audio_to_bytes, read_ref_text
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest


class FishSpeechTTS:
    def __init__(self, api_url="http://127.0.0.1:8080/v1/tts", api_key="YOUR_API_KEY"):
        """
        Initialize Fish Speech TTS Client
        
        Args:
            api_url: URL des Fish Speech API Servers
            api_key: API Key für Authentifizierung
        """
        self.api_url = api_url
        self.api_key = api_key
    
    def generate_speech(
        self, 
        text: str, 
        output_filename: str, 
        reference_audio: str = None, 
        reference_text: str = None,
        # Erweiterte Parameter für bessere Qualität
        temperature: float = 0.8,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        chunk_length: int = 300,
        max_new_tokens: int = 1024,
        output_format: str = "wav",
        seed: int = None
    ):
        """
        Generiere Sprache mit Fish Speech
        
        Args:
            text: Der zu synthetisierende Text
            output_filename: Dateiname für die Ausgabe (ohne Endung)
            reference_audio: Pfad zur Referenz-Audio-Datei
            reference_text: Text der Referenz-Audio
            temperature: Sampling Temperature (0.1-2.0, höher = mehr Variation)
            top_p: Top-p Sampling (0.1-1.0)
            repetition_penalty: Wiederholungsstrafe (1.0-2.0)
            chunk_length: Chunk-Länge für Synthese
            max_new_tokens: Maximale neue Tokens
            output_format: Ausgabeformat (wav, mp3, flac)
            seed: Seed für deterministische Generierung (None = zufällig)
        
        Returns:
            bool: True wenn erfolgreich, False wenn Fehler
        """
        
        # Referenz-Audio und Text verarbeiten
        if reference_audio and os.path.exists(reference_audio):
            try:
                byte_audio = audio_to_bytes(reference_audio)
                ref_text = reference_text if reference_text else ""
                references = [ServeReferenceAudio(audio=byte_audio, text=ref_text)]
                print(f"✅ Referenz-Audio geladen: {reference_audio}")
            except Exception as e:
                print(f"❌ Fehler beim Laden der Referenz-Audio: {e}")
                references = []
        else:
            references = []
            if reference_audio:
                print(f"⚠️  Referenz-Audio nicht gefunden: {reference_audio}")
        
        # Request-Daten zusammenstellen
        data = {
            "text": text,
            "references": references,
            "reference_id": None,
            "format": output_format,
            "max_new_tokens": max_new_tokens,
            "chunk_length": chunk_length,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "streaming": False,
            "use_memory_cache": "off",
            "seed": seed,
        }
        
        # Pydantic Validierung
        try:
            pydantic_data = ServeTTSRequest(**data)
        except Exception as e:
            print(f"❌ Fehler bei der Datenvalidierung: {e}")
            return False
        
        # API Request senden
        try:
            print(f"🚀 Generiere Sprache für: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            response = requests.post(
                self.api_url,
                data=ormsgpack.packb(pydantic_data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
                headers={
                    "authorization": f"Bearer {self.api_key}",
                    "content-type": "application/msgpack",
                },
                timeout=120  # 2 Minuten Timeout
            )
            
            if response.status_code == 200:
                # Audio-Datei speichern
                output_path = f"{output_filename}.{output_format}"
                with open(output_path, "wb") as audio_file:
                    audio_file.write(response.content)
                
                file_size = len(response.content) / 1024  # KB
                print(f"✅ Audio gespeichert: {output_path} ({file_size:.1f} KB)")
                return True
            else:
                print(f"❌ API Fehler {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("❌ Timeout: Server antwortet nicht rechtzeitig")
            return False
        except Exception as e:
            print(f"❌ Unerwarteter Fehler: {e}")
            return False


def main():
    """
    Beispiel-Verwendung des Fish Speech TTS Clients
    """
    
    # TTS Client initialisieren
    tts = FishSpeechTTS()
    
    # Test-Konfiguration
    tests = [
        {
            "text": "Hallo! Das ist ein Test der deutschen Sprachsynthese mit Fish Speech.",
            "output_filename": "connection_test_output/test_01_einfach",
            "reference_audio": None,
            "reference_text": None
        },
        {
            "text": "Guten Tag! Wie geht es Ihnen heute? Das Wetter ist wirklich schön!",
            "output_filename": "connection_test_output/test_02_freundlich", 
            "reference_audio": None,
            "reference_text": None,
            "temperature": 0.9,  # Mehr Variation
            "top_p": 0.85
        },
        {
            "text": "Dies ist ein Test mit einer Referenz-Stimme. Die Qualität sollte deutlich besser sein.",
            "output_filename": "connection_test_output/test_03_mit_referenz",
            "reference_audio": "voices/calm.wav",
            "reference_text": "Alles ist ruhig und friedlich, ganz entspannt und gelassen."
        }
    ]
    
    print("🎯 Starte Fish Speech TTS Tests...\n")
    
    # Prüfe ob API Server läuft
    try:
        response = requests.get("http://127.0.0.1:8080/", timeout=5)
        print("✅ API Server ist erreichbar\n")
    except:
        print("❌ API Server nicht erreichbar! Starte ihn mit:")
        print("python -m tools.api_server --llama-checkpoint-path 'checkpoints/openaudio-s1-mini' --decoder-checkpoint-path 'checkpoints/openaudio-s1-mini/codec.pth' --decoder-config-name modded_dac_vq")
        return
    
    # Tests durchführen
    for i, test in enumerate(tests, 1):
        print(f"📝 Test {i}/{len(tests)}:")
        
        # Kopiere Test-Parameter
        params = test.copy()
        
        # Führe TTS durch
        success = tts.generate_speech(**params)
        
        if success:
            print("🎉 Erfolgreich!\n")
        else:
            print("💥 Fehlgeschlagen!\n")


if __name__ == "__main__":
    main()