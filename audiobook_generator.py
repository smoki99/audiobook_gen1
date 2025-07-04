#!/usr/bin/env python3
"""
Audiobook Generator using Fish Speech TTS

This script processes text files, applies voice configurations from a JSON file,
and generates audio using the Fish Speech TTS API.
"""

import wave
import os
import sys
import json
import re
import glob
import random
import datetime
import csv
from pathlib import Path
from fish_speech_tts import FishSpeechTTS

# 1. Definieren der bekannten Emotionen, Tone Makers und Special Audio Effects
# Wir extrahieren sie direkt aus dem bereitgestellten Text 
# -> (scared) wurde entfernt, da es immer wieder zu problemen geführt hat
raw_tags_string = """
(angry) (sad) (excited) (surprised) (satisfied) (delighted)
(worried) (upset) (nervous) (frustrated) (depressed)
(empathetic) (embarrassed) (disgusted) (moved) (proud) (relaxed)
(grateful) (confident) (interested) (curious) (confused) (joyful)
(disdainful) (unhappy) (anxious) (indifferent)
(scornful) (panicked) (reluctant)
(disapproving) (negative) (denying) (serious)
(sarcastic) (conciliative) (comforting) (sincere) (sneering)
(hesitating) (yielding) (painful) (awkward) (amused)

Tone Makers:
(in a hurry tone) (shouting) (screaming) (whispering) (soft tone)

Special Audio effect:
(laughing) (chuckling) (crying loudly) (sighing) (panting)
(groaning) (crowd laughing) (background laughter) (audience laughing)
(break) (long-break)
"""

# Verwenden von Regular Expressions, um alle Inhalte innerhalb von Klammern zu finden
# und sie in ein Set umzuwandeln, um Duplikate zu vermeiden, dann in eine Liste
allowed_emotions = sorted(list(set(re.findall(r'\((.*?)\)', raw_tags_string))))


# Check for required libraries
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    print("⚠️ pydub not installed. Audio concatenation disabled.")
    print("To enable concatenation, install pydub: pip install pydub")
    HAS_PYDUB = False

def load_mapping_csv(csv_path):
    """Load word mappings from CSV file"""
    mapping = {}
    if not csv_path or not os.path.exists(csv_path):
        return mapping
        
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Try to detect if the CSV has headers
            sample = f.read(1024)
            f.seek(0)
            
            # Check if first line looks like headers
            first_line = f.readline().strip()
            f.seek(0)
            
            reader = csv.reader(f)
            
            # Skip header if it exists (contains non-quoted alphabetic content)
            if not first_line.startswith('"') and any(c.isalpha() for c in first_line.split(',')[0]):
                next(reader, None)  # Skip header row
                
            for row in reader:
                if len(row) >= 2:
                    # Clean quotes from CSV values
                    original = row[0].strip(' "')
                    mapped = row[1].strip(' "')
                    if original and mapped:
                        mapping[original] = mapped
                        
        print(f"✅ Loaded {len(mapping)} word mappings from {csv_path}")
        if len(mapping) > 0:
            # Show first few mappings as examples
            sample_mappings = list(mapping.items())[:3]
            for orig, mapped in sample_mappings:
                print(f"  - '{orig}' → '{mapped}'")
            if len(mapping) > 3:
                print(f"  ... and {len(mapping) - 3} more")
                
    except Exception as e:
        print(f"⚠️ Error loading mapping CSV: {e}")
        
    return mapping

def apply_text_mapping(text, mapping):
    """Apply word mappings to text"""
    if not mapping:
        return text
        
    # Apply mappings (case-sensitive, whole word matching)
    mapped_text = text
    for original, mapped in mapping.items():
        # Use word boundaries to ensure whole word matching
        pattern = r'\b' + re.escape(original) + r'\b'
        mapped_text = re.sub(pattern, mapped, mapped_text)
        
    return mapped_text
    
def extract_voice_tag(line):
    """Extract voice tag from text line and remove the tag"""
    voice_match = re.search(r'\[([^\[\]]+)\]', line)
    if voice_match:
        voice = voice_match.group(1).strip()
        # Remove the voice tag from the line
        clean_line = re.sub(r'\[[^\[\]]+\]', '', line).strip()
        return voice, clean_line
    return "default", line.strip()

def select_reference_audio(reference_config):
    """
    Selects a reference audio and corresponding text from the configuration.
    Supports both single reference and multiple references for random selection.
    
    Args:
        reference_config: Either a single reference object or a list of references
        
    Returns:
        tuple: (reference_audio_path, reference_text)
    """
    if not reference_config:
        return None, None
        
    # Handle list of references
    if isinstance(reference_config, list):
        if not reference_config:
            return None, None
            
        # Randomly select one reference from the list
        selected = random.choice(reference_config)
        
        # Handle different formats of the selected reference
        if isinstance(selected, dict) and "audio" in selected and "text" in selected:
            return selected["audio"], selected["text"]
        elif isinstance(selected, list) and len(selected) >= 2:
            return selected[0], selected[1]
        else:
            # If format is unknown, return None
            print(f"⚠️ Unknown reference format: {selected}")
            return None, None
    
    # Handle single reference as dict
    elif isinstance(reference_config, dict) and "audio" in reference_config and "text" in reference_config:
        return reference_config["audio"], reference_config["text"]
    
    # Handle single reference as string (assume it's just the audio path)
    elif isinstance(reference_config, str):
        return reference_config, None
        
    # Handle other formats (simple list of [audio_path, text])
    elif isinstance(reference_config, list) and len(reference_config) >= 2:
        return reference_config[0], reference_config[1]
        
    # Unknown format
    print(f"⚠️ Unknown reference format: {reference_config}")
    return None, None

def create_silent_wav(duration_ms: int, filename: str = None, channels: int = 1, sampwidth: int = 2, framerate: int = 44100):
    """
    Erstellt eine WAV-Datei mit digitaler Stille.

    Args:
        duration_ms (int): Die Dauer der Stille in Millisekunden.
        filename (str, optional): Der gewünschte Dateiname. 
                                Wenn nicht angegeben, wird ein Name wie 'stille_1000ms.wav' generiert.
        channels (int, optional): Anzahl der Kanäle (1 für Mono, 2 für Stereo). Standard ist 1.
        sampwidth (int, optional): Sample-Breite in Bytes (1 für 8-bit, 2 für 16-bit). Standard ist 2.
        framerate (int, optional): Die Abtastrate in Hz. Standard ist 44100.

    Returns:
        str: Der absolute Pfad zur erstellten Datei.
    """
    
    # Wenn kein Dateiname angegeben wurde, einen generieren
    if filename is None:
        filename = f"stille_{duration_ms}ms.wav"
    
    # Sicherstellen, dass die Dateiendung .wav ist
    if not filename.lower().endswith('.wav'):
        filename += '.wav'

    # --- Berechnungen ---
    duration_s = duration_ms / 1000.0
    num_frames = int(duration_s * framerate)
    
    # Erzeuge die stillen Bytes
    # b'\x00' repräsentiert ein Null-Byte, also Stille
    silent_bytes = b'\x00' * (num_frames * channels * sampwidth)
    
    # --- Datei erstellen ---
    try:
        with wave.open(filename, 'wb') as wav_file:
            # WAV-Header-Parameter setzen
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(framerate)
            wav_file.setnframes(num_frames)
            wav_file.setcomptype("NONE", "not compressed")

            # Die stillen Audio-Daten schreiben
            wav_file.writeframes(silent_bytes)
        
        # Gebe den vollständigen Pfad der erstellten Datei zurück
        return os.path.abspath(filename)

    except Exception as e:
        print(f"Fehler beim Erstellen der Datei '{filename}': {e}")
        return None

def concatenate_audio_files(audio_files, output_file, format="wav"):
    """Concatenate multiple audio files into a single file and return timing information"""
    if not HAS_PYDUB:
        print("❌ Audio concatenation failed: pydub not installed")
        return False, None
    
    if not audio_files:
        print("⚠️ No audio files to concatenate")
        return False, None
    
    try:
        # Load the first audio file
        combined = AudioSegment.from_file(audio_files[0], format=format)
        
        # Track timing information for SRT subtitles
        timing_info = []
        current_time = 0.0
        
        # Add timing for first file
        first_duration = len(combined) / 1000.0  # Convert ms to seconds
        timing_info.append({
            'start': current_time,
            'end': current_time + first_duration,
            'duration': first_duration
        })
        current_time += first_duration
        
        # Add a small pause between segments (500ms)
        random_duration = random.randint(250, 600)
        pause = AudioSegment.silent(duration=random_duration)
        pause_duration = random_duration / 1000.0  # Convert ms to seconds
        
        # Append the rest
        for audio_file in audio_files[1:]:
            segment = AudioSegment.from_file(audio_file, format=format)
            combined += pause + segment
            
            # Track timing including pause
            current_time += pause_duration
            segment_duration = len(segment) / 1000.0  # Convert ms to seconds
            timing_info.append({
                'start': current_time,
                'end': current_time + segment_duration,
                'duration': segment_duration
            })
            current_time += segment_duration
        
        # Export the combined file
        combined.export(output_file, format=format)
        
        file_size = os.path.getsize(output_file) / 1024  # KB
        print(f"✅ Combined audio saved: {output_file} ({file_size:.1f} KB)")
        return True, timing_info
    
    except Exception as e:
        print(f"❌ Error concatenating audio files: {e}")
        return False, None

def process_text_file(file_path, output_dir, voice_config, tts_client, word_mapping=None):
    """Process a single text file and generate audio for each line"""
    base_name = Path(file_path).stem
    print(f"\n📄 Processing file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    successful_lines = 0
    total_lines = 0
    generated_files = []
    output_format = voice_config.get("default", {}).get("output_format", "wav")
    
    for i, line in enumerate(lines):
        # Skip empty lines completely
        if not line.strip():
            continue
        
        total_lines += 1
        
        # Extract voice tag and clean line
        voice, clean_text_with_emotions = extract_voice_tag(line)

        # Remove unallowed emotions
        clean_text = clean_script_text(clean_text_with_emotions, allowed_emotions)

        # Skip if clean text is empty after tag removal
        if not clean_text:
            continue
        
        # Apply word mapping if provided
        if word_mapping:
            original_text = clean_text
            clean_text = apply_text_mapping(clean_text, word_mapping)
            if clean_text != original_text:
                print(f"🔄 Text mapping applied: '{original_text[:30]}...' → '{clean_text[:30]}...'")
        
        # Get voice configuration (use default if specified voice doesn't exist)
        voice_params = voice_config.get(voice, voice_config.get("default", {}))
        print(voice_params)

        # Make a copy of the parameters to avoid modifying the original
        params = voice_params.copy()
        
        # Get output format
        output_format = params.get("output_format", "wav")
        
        # Handle reference audio selection (single or multiple)
        reference_config = params.get("reference_audio")
        reference_text_config = params.get("reference_text")
        
        # If we have a reference_audio_set, that takes precedence
        if "reference_audio_set" in params:
            ref_audio, ref_text = select_reference_audio(params["reference_audio_set"])
            if ref_audio:
                params["reference_audio"] = ref_audio
                if ref_text:  # Only override if we have a text from the set
                    params["reference_text"] = ref_text
                    
                # Log which reference was selected
                print(f"🎤 Selected reference: {os.path.basename(ref_audio)}")
        
        # Generate output filename
        output_filename = f"{output_dir}/{base_name}_{i:03d}"
        final_output_path = f"{output_filename}.{output_format}"
        
        # Get voice comment for logging if available
        voice_comment = voice_params.get("comment", "")
        if voice_comment:
            voice_info = f"{voice} ({voice_comment})"
        else:
            voice_info = voice
            
        print(f"\n🎙️ Line {i+1}/{len(lines)} - Voice: {voice_info}")
        print(f"📝 Text: \"{clean_text[:50]}{'...' if len(clean_text) > 50 else ''}\"")
        
        # Ensure parameter names match the function signature
        # Fix any potential naming mismatches
        if "ref_audio" in params:
            params["reference_audio"] = params.pop("ref_audio")
        if "ref_text" in params:
            params["reference_text"] = params.pop("ref_text")
    

        # Remove parameters that shouldn't be passed to TTS function
        for param_to_remove in ["speed", "reference_audio_set", "comment"]:
            if param_to_remove in params:
                params.pop(param_to_remove)
        
        # --- Spezialfall für '...' Pausen ---
        if clean_text.strip() == "...":
            print(f"⏱️ Detected silence marker '...'. Generating 2000ms silent WAV file.")
            # Eine längere Pause für die Ellipse
            created_path = create_silent_wav(duration_ms=700, filename=final_output_path)
            if created_path:
                successful_lines += 1 # Zählt als erfolgreiche "Zeile"
                generated_files.append(final_output_path)
            continue


        # Generate speech with appropriate parameters
        try:
            # Generate directly to output file
            success = tts_client.generate_speech(
                text=clean_text,
                output_filename=output_filename,
                **params
            )
            
            if success:
                successful_lines += 1
                generated_files.append(final_output_path)
                
        except TypeError as e:
            print(f"❌ Parameter error: {e}")
            print("⚠️ Attempting with default parameters only")
            try:
                # Try again with minimal parameters
                success = tts_client.generate_speech(
                    text=clean_text,
                    output_filename=output_filename
                )
                if success:
                    successful_lines += 1
                    generated_files.append(final_output_path)
            except Exception as e2:
                print(f"❌ Failed with default parameters too: {e2}")
    
    print(f"\n✅ Completed file {file_path}: {successful_lines}/{total_lines} lines processed successfully")
    
    # Concatenate all generated audio files
    if successful_lines > 0 and HAS_PYDUB and generated_files:
        print(f"\n🔄 Concatenating {successful_lines} audio segments into a single file...")
        concat_output = f"{output_dir}/{base_name}_concat.{output_format}"
        concatenation_success, timing_info = concatenate_audio_files(generated_files, concat_output, format=output_format)
        
        # Note: audiobook_generator.py doesn't track subtitle texts, so no SRT generation here
        # To enable SRT generation, the process_text_file function would need to be updated
        # to track subtitle texts like in improved_audiobook_generator.py
    
    return successful_lines, total_lines


def clean_script_text(text_to_clean, known_tags):
    """
    Entfernt alle Klammerausdrücke (z.B. '(invalid_tag)'),
    es sei denn, der Inhalt der Klammern ist in der Liste der bekannten Tags.

    Args:
        text_to_clean (str): Der Eingabetext, der bereinigt werden soll.
        known_tags (list): Eine Liste von Strings, die die erlaubten Tag-Inhalte sind.

    Returns:
        str: Der bereinigte Text.
    """
    # Regulärer Ausdruck, um (beliebiger_inhalt) zu finden
    # Der Inhalt in den Klammern wird in Gruppe 1 erfasst
    pattern = re.compile(r'\((.*?)\)')

    def replace_match(match):
        # Den Inhalt innerhalb der Klammern extrahieren (Gruppe 1)
        tag_content = match.group(1)
        # Überprüfen, ob der Inhalt in unseren bekannten Tags ist
        if tag_content in known_tags:
            # Wenn ja, behalte den Original-Tag bei (Gruppe 0 ist der gesamte Treffer inklusive Klammern)
            return match.group(0)
        else:
            # Wenn nein, entferne den Tag, indem ein leerer String zurückgegeben wird
            return ""

    # Führe die Ersetzung durch, wobei die Funktion replace_match für jeden Treffer aufgerufen wird
    cleaned_text = pattern.sub(replace_match, text_to_clean)
    
    # Remove all but one
    matches = re.findall(r'\([^)]*\)', cleaned_text)

    # Keep the first (...) group only
    if matches:
        first = matches[0]
        # Remove all (...) groups
        cleaned_text = re.sub(r'\([^)]*\)', '', cleaned_text)
        # Strip extra spaces and prepend the first (...) group
        cleaned_text = first + ' ' + re.sub(r'\s+', ' ', cleaned_text).strip()
    else:
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

def natural_sort_key(s):
    """
    Sort strings containing numbers naturally (e.g., chapter1.txt comes before chapter10.txt)
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def get_current_time_utc():
    """Get current UTC time in YYYY-MM-DD HH:MM:SS format"""
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def main():
    """Main function to process text files and generate audio"""
    # Check command line arguments
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python audiobook_generator.py <input_dir> <output_dir> <voice_config_file> [mapping.csv]")
        print("\nArguments:")
        print("  input_dir        - Directory containing text files to process")
        print("  output_dir       - Directory where audio files will be saved")
        print("  voice_config_file - JSON file with voice configurations")
        print("  mapping.csv      - (Optional) CSV file with word mappings")
        print("\nExample:")
        print("  python audiobook_generator.py ./texts ./output voices.json")
        print("  python audiobook_generator.py ./texts ./output voices.json mapping.csv")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    voice_config_file = sys.argv[3]
    mapping_csv = sys.argv[4] if len(sys.argv) == 5 else None
    
    # Check if directories exist
    if not os.path.isdir(input_dir):
        print(f"❌ Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load word mapping if provided
    word_mapping = load_mapping_csv(mapping_csv) if mapping_csv else {}
    
    # Load voice configuration
    try:
        with open(voice_config_file, 'r', encoding='utf-8') as f:
            voice_config = json.load(f)
        print(f"✅ Loaded voice configuration from {voice_config_file}")
        
        # Count voices with comments
        voices_with_comments = sum(1 for k, v in voice_config.items() if "comment" in v)
        print(f"🎙️ Available voices: {len(voice_config.keys())} ({voices_with_comments} with comments)")
        
        # Print voice names
        print(f"🔹 Voice list: {', '.join(voice_config.keys())}")
        
        # Print reference sets info
        for voice_name, config in voice_config.items():
            if "reference_audio_set" in config:
                ref_set = config["reference_audio_set"]
                if isinstance(ref_set, list):
                    print(f"  - '{voice_name}' has {len(ref_set)} reference variations")
    except Exception as e:
        print(f"❌ Error loading voice configuration: {e}")
        sys.exit(1)
    
    # Initialize TTS client
    tts_client = FishSpeechTTS()
    
    # Find all text files in input directory
    text_files = glob.glob(f"{input_dir}/*.txt")
    
    if not text_files:
        print(f"❌ No text files found in {input_dir}")
        sys.exit(1)
    
    # Sort text files naturally
    text_files.sort(key=natural_sort_key)
    
    print(f"🔍 Found {len(text_files)} text file(s) to process")
    print(f"📋 Processing order: {', '.join([os.path.basename(f) for f in text_files])}")
    
    if not HAS_PYDUB:
        print("⚠️ pydub nicht installiert. Audio-Zusammenführung wird übersprungen.")
    
    # Process each text file
    total_successful = 0
    total_lines = 0
    
    for file_path in text_files:
        file_successful, file_total = process_text_file(file_path, output_dir, voice_config, tts_client, word_mapping)
        total_successful += file_successful
        total_lines += file_total
    
    # Print summary
    print("\n📊 Zusammenfassung:")
    print(f"🔹 {len(text_files)} Datei(en) verarbeitet")
    print(f"🔹 {total_successful}/{total_lines} Audiodateien generiert")
    if word_mapping:
        print(f"🔹 {len(word_mapping)} Wort-Mappings angewendet")
    if HAS_PYDUB:
        print(f"🔹 {len(text_files)} zusammengeführte Audiodateien erstellt")
    print(f"🔹 Ausgabeverzeichnis: {output_dir}")
    print(f"\n📅 Abgeschlossen am: {get_current_time_utc()}")

if __name__ == "__main__":
    main()