#!/usr/bin/env python3
"""
Audiobook Generator using Fish Speech TTS with Quality Improvement

This script processes text files, applies voice configurations from a JSON file,
and generates audio using the Fish Speech TTS API. It includes quality evaluation
to select the best audio from multiple generations with German language support.
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
import tempfile
import argparse
from pathlib import Path
from fish_speech_tts import FishSpeechTTS

# Try importing quality evaluation libraries
try:
    import librosa
    import numpy as np
    HAS_LIBROSA = True
    print("✅ Librosa loaded for audio analysis")
except ImportError:
    print("⚠️ librosa not installed. Quality evaluation will be limited.")
    print("To enable better quality evaluation, install: pip install librosa numpy")
    HAS_LIBROSA = False
except Exception as e:
    print(f"⚠️ Error initializing librosa: {e}")
    HAS_LIBROSA = False

# Try importing speech recognition for pronunciation check
try:
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    HAS_SPEECH_RECOGNITION = True
    print("✅ Speech Recognition loaded for pronunciation analysis")
except ImportError:
    print("⚠️ speech_recognition not installed. Pronunciation evaluation will be disabled.")
    print("To enable pronunciation evaluation, install: pip install SpeechRecognition")
    HAS_SPEECH_RECOGNITION = False
except Exception as e:
    print(f"⚠️ Error initializing speech recognition: {e}")
    HAS_SPEECH_RECOGNITION = False

# Check for pydub
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    print("⚠️ pydub not installed. Audio concatenation disabled.")
    print("To enable concatenation, install pydub: pip install pydub")
    HAS_PYDUB = False

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

def format_srt_timestamp(seconds):
    """Format seconds as SRT timestamp (HH:MM:SS,MS)"""
    # Handle floating-point precision issues by rounding to 3 decimal places
    seconds = round(seconds, 3)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def clean_subtitle_text(text):
    """Clean text for subtitle display by removing emotion tags and formatting"""
    # Remove emotion tags like (angry), (sad), etc.
    # Use a more robust approach to handle nested parentheses
    while '(' in text and ')' in text:
        # Find the first opening parenthesis
        start = text.find('(')
        if start == -1:
            break
        
        # Find the matching closing parenthesis
        count = 1
        end = start + 1
        while end < len(text) and count > 0:
            if text[end] == '(':
                count += 1
            elif text[end] == ')':
                count -= 1
            end += 1
        
        # If we found a matching closing parenthesis, remove the content
        if count == 0:
            text = text[:start] + text[end:]
        else:
            # If no matching closing parenthesis, just remove the opening one
            text = text[:start] + text[start+1:]
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Filter out silence markers
    if text == "...":
        return ""
    
    return text

def create_srt_subtitles(subtitle_texts, timing_info, output_srt_path):
    """Create SRT subtitle file with accurate timestamps
    
    Args:
        subtitle_texts: List of text content for each audio segment
        timing_info: List of timing dictionaries with start, end, duration
        output_srt_path: Path for the output SRT file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not subtitle_texts or not timing_info:
        print("⚠️ No subtitle texts or timing information provided")
        return False
    
    if len(subtitle_texts) != len(timing_info):
        print(f"⚠️ Mismatch: {len(subtitle_texts)} subtitle texts vs {len(timing_info)} timing entries")
        return False
    
    try:
        srt_entries = []
        
        for i, (subtitle_text, timing) in enumerate(zip(subtitle_texts, timing_info)):
            # Clean subtitle text
            clean_text = clean_subtitle_text(subtitle_text)
            if not clean_text:
                # Skip empty text entries
                continue
            
            # Format timestamps
            start_timestamp = format_srt_timestamp(timing['start'])
            end_timestamp = format_srt_timestamp(timing['end'])
            
            # Create SRT entry
            srt_entry = f"{len(srt_entries) + 1}\n{start_timestamp} --> {end_timestamp}\n{clean_text}\n"
            srt_entries.append(srt_entry)
        
        # Write SRT file
        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_entries))
        
        print(f"✅ SRT subtitles created: {output_srt_path} ({len(srt_entries)} entries)")
        return True
    
    except Exception as e:
        print(f"❌ Error creating SRT subtitles: {e}")
        return False

def check_audio_completeness(audio_path, expected_duration_estimate):
    """
    Check if an audio file seems complete based on its duration.
    
    Args:
        audio_path: Path to the audio file
        expected_duration_estimate: Estimated expected duration in seconds
        
    Returns:
        float: Completeness score between 0.0 and 1.0
    """
    try:
        if HAS_PYDUB:
            # Get actual duration using pydub
            audio = AudioSegment.from_file(audio_path)
            actual_duration = len(audio) / 1000  # Convert ms to seconds
            
            # Calculate completeness score
            if actual_duration >= expected_duration_estimate:
                # Audio is at least as long as expected - good!
                return min(1.0, actual_duration / (expected_duration_estimate * 1.5))
            else:
                # Audio is shorter than expected - might be cut off
                completeness_ratio = actual_duration / expected_duration_estimate
                return max(0.1, completeness_ratio)  # Min score 0.1
        else:
            # Fallback method - check file size
            file_size = os.path.getsize(audio_path)
            expected_min_size = expected_duration_estimate * 16000  # Rough estimate
            
            if file_size >= expected_min_size:
                return 1.0
            else:
                return max(0.1, file_size / expected_min_size)
    
    except Exception as e:
        print(f"⚠️ Error checking audio completeness: {e}")
        return 0.5  # Default middle value

def check_audio_continuity(audio_path):
    """
    Check for breaks or unexpected silences in the audio.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        float: Continuity score between 0.0 and 1.0
    """
    try:
        if HAS_LIBROSA:
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=None)
            
            # Calculate RMS energy
            hop_length = 512
            frame_length = 2048
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Detect silent regions
            silent_threshold = 0.01
            is_silent = rms < silent_threshold
            
            # Calculate proportion of silent frames and count silence blocks
            if len(rms) > 0:
                silent_ratio = sum(is_silent) / len(rms)
                
                # Convert frame-level silence into time blocks
                silence_blocks = []
                in_silence = False
                start_frame = 0
                
                for i, silent in enumerate(is_silent):
                    if silent and not in_silence:
                        # Start of silence
                        in_silence = True
                        start_frame = i
                    elif not silent and in_silence:
                        # End of silence
                        in_silence = False
                        duration_frames = i - start_frame
                        # Only count if longer than 0.3s (adjust threshold as needed)
                        if (duration_frames * hop_length / sr) > 0.3:
                            silence_blocks.append((start_frame, i))
                
                # Handle case where audio ends in silence
                if in_silence:
                    duration_frames = len(is_silent) - start_frame
                    if (duration_frames * hop_length / sr) > 0.3:
                        silence_blocks.append((start_frame, len(is_silent)))
                
                # Analyze silence distribution
                num_blocks = len(silence_blocks)
                
                # Calculate continuity score
                if num_blocks == 0:
                    # No significant silence - perfect continuity
                    return 1.0
                elif num_blocks == 1:
                    # One silence block might be ok if it's at the end
                    if silence_blocks[0][0] > len(rms) * 0.8:
                        # Silence is near the end - probably fine
                        return 0.9
                    else:
                        # Silence in middle or beginning - suspicious
                        return 0.5
                else:
                    # Multiple silence blocks - might indicate breaks
                    # More blocks = worse score
                    return max(0.1, 1.0 - (num_blocks * 0.15))
            else:
                return 0.5  # Default for empty audio
        elif HAS_PYDUB:
            # Simpler method with pydub - check for very quiet sections
            audio = AudioSegment.from_file(audio_path)
            
            # Split into 100ms chunks and check loudness
            chunk_size = 100  # ms
            chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
            
            # Count very quiet chunks (potentially breaks)
            quiet_threshold = -40  # dBFS
            quiet_chunks = sum(1 for chunk in chunks if chunk.dBFS < quiet_threshold)
            
            if len(chunks) > 0:
                quiet_ratio = quiet_chunks / len(chunks)
                
                # Calculate continuity score
                if quiet_ratio < 0.1:
                    # Few quiet chunks - good continuity
                    return 1.0
                elif quiet_ratio < 0.2:
                    # Some quiet chunks - decent continuity
                    return 0.8
                elif quiet_ratio < 0.3:
                    # More quiet chunks - questionable continuity
                    return 0.5
                else:
                    # Too many quiet chunks - poor continuity
                    return max(0.1, 1.0 - quiet_ratio)
            else:
                return 0.5  # Default for empty audio
        else:
            # No audio analysis libraries available
            return 0.7  # Default reasonable value
    
    except Exception as e:
        print(f"⚠️ Error checking audio continuity: {e}")
        return 0.5  # Default middle value

def check_pronunciation(audio_path, original_text):
    """
    Check pronunciation quality using speech recognition (German language)
    
    Args:
        audio_path: Path to the audio file
        original_text: Original text that should be spoken
        
    Returns:
        float: Pronunciation score between 0.0 and 1.0
    """
    if not HAS_SPEECH_RECOGNITION:
        return 0.7  # Default reasonable score if speech recognition is not available
    
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            
            # Use German language recognition
            recognized_text = recognizer.recognize_google(audio_data, language="de-DE")
            
            # Normalize both texts for comparison
            original_normalized = re.sub(r'[^\w\s]', '', original_text.lower())
            recognized_normalized = re.sub(r'[^\w\s]', '', recognized_text.lower())
            
            # Print the recognized text for debugging
            print(f"  🔊 Recognized text: '{recognized_normalized[:50]}...'")
            
            # Compare words in both texts
            original_words = set(original_normalized.split())
            recognized_words = set(recognized_normalized.split())
            
            if len(original_words) > 0:
                # Calculate word overlap as a proportion
                common_words = original_words.intersection(recognized_words)
                return min(1.0, len(common_words) / len(original_words))
            
            return 0.7  # Default if no words to compare
    
    except Exception as e:
        # Speech recognition often fails, don't print the error
        return 0.7  # Default if recognition fails

def evaluate_audio_quality(audio_path, original_text):
    """
    Evaluate the quality of an audio file based on multiple metrics.
    
    Args:
        audio_path (str): Path to the audio file
        original_text (str): Original text that should be spoken in the audio
        
    Returns:
        dict: Dictionary with quality scores
    """
    scores = {
        'completeness': 0.0,  # Detects if audio cuts off prematurely
        'continuity': 0.0,    # Detects breaks in the middle
        'pronunciation': 0.7, # Pronunciation accuracy (default to reasonable value)
        'total': 0.0          # Weighted average
    }
    
    if not os.path.exists(audio_path):
        return scores
    
    try:
        # Basic quality check - file exists and has non-zero size
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            return scores  # File exists but is empty
        
        # Calculate expected minimum duration based on text length
        # This is a rough estimation: average reading speed is ~150 words per minute or 2.5 words/second
        word_count = len(original_text.split())
        expected_duration = max(1.0, word_count / 2.5)  # At least 1 second
        
        # Check completeness
        scores['completeness'] = check_audio_completeness(audio_path, expected_duration)
        
        # Check continuity
        scores['continuity'] = check_audio_continuity(audio_path)
        
        # Check pronunciation
        scores['pronunciation'] = check_pronunciation(audio_path, original_text)
        
        # Calculate weighted total score
        # Higher weights on completeness and continuity since they're most important
        scores['total'] = (
            0.5 * scores['completeness'] + 
            0.3 * scores['continuity'] + 
            0.2 * scores['pronunciation']
        )
        
    except Exception as e:
        print(f"❌ Error evaluating audio quality: {e}")
        # Set default scores
        scores['completeness'] = 0.5
        scores['continuity'] = 0.5
        scores['pronunciation'] = 0.7
        scores['total'] = 0.55
    
    return scores

def generate_with_quality_control(tts_client, text, output_filename, params, num_retries=3):
    """
    Generate speech with multiple attempts and select the best quality.
    
    Args:
        tts_client: The FishSpeechTTS client
        text (str): The text to convert to speech
        output_filename (str): Base filename without extension
        params (dict): Parameters for the TTS engine
        num_retries (int): Number of generation attempts
        
    Returns:
        tuple: (success, final_output_path)
    """
    if num_retries < 1:
        num_retries = 1
    
    best_score = -1.0
    best_audio_path = None
    temp_files = []
    output_format = params.get("output_format", "wav")
    final_output_path = f"{output_filename}.{output_format}"
    
    print(f"🔄 Generating {num_retries} samples to find best quality...")
    
    for attempt in range(num_retries):
        # Generate a temporary filename for this attempt
        temp_output_filename = f"{output_filename}_attempt_{attempt}"
        temp_output_path = f"{temp_output_filename}.{output_format}"
        temp_files.append(temp_output_path)
        
        try:
            # Generate speech for this attempt
            success = tts_client.generate_speech(
                text=text,
                output_filename=temp_output_filename,
                **params
            )
            
            if success and os.path.exists(temp_output_path):
                # Evaluate the quality of this attempt
                scores = evaluate_audio_quality(temp_output_path, text)
                
                print(f"  Attempt {attempt+1}/{num_retries}: " + 
                      f"Scores[Total={scores['total']:.2f}, " +
                      f"Completeness={scores['completeness']:.2f}, " +
                      f"Continuity={scores['continuity']:.2f}, " +
                      f"Pronunciation={scores['pronunciation']:.2f}]")
                
                # Update best audio if this one is better
                if scores['total'] > best_score:
                    best_score = scores['total']
                    best_audio_path = temp_output_path
            else:
                print(f"  ❌ Attempt {attempt+1}/{num_retries}: Failed to generate audio")
        
        except Exception as e:
            print(f"  ❌ Attempt {attempt+1}/{num_retries}: Error: {e}")
    
    # Copy the best audio to the final output path
    if best_audio_path:
        try:
            # Use pydub to copy if available (preserves audio format better)
            if HAS_PYDUB:
                audio = AudioSegment.from_file(best_audio_path)
                audio.export(final_output_path, format=output_format)
            else:
                # Fallback to simple file copy
                with open(best_audio_path, 'rb') as src, open(final_output_path, 'wb') as dst:
                    dst.write(src.read())
                    
            print(f"✅ Selected best audio (score: {best_score:.2f}) and saved to: {final_output_path}")
            
            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file) and temp_file != final_output_path:
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                        
            return True, final_output_path
        
        except Exception as e:
            print(f"❌ Error copying best audio: {e}")
            
            # If we can't copy, but the best file exists, use it directly
            if os.path.exists(best_audio_path):
                return True, best_audio_path
    
    return False, None

def process_text_file(file_path, output_dir, voice_config, tts_client, word_mapping=None, num_retries=3, no_overwrite=False):
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
    subtitle_texts = []  # Track text content for subtitles
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
        
        # Check if file already exists and skip if no-overwrite is enabled
        if no_overwrite and os.path.exists(final_output_path):
            print(f"⏭️ Skipping existing file: {final_output_path}")
            successful_lines += 1
            generated_files.append(final_output_path)
            subtitle_texts.append(clean_text_with_emotions)  # Store original text with emotions for subtitles
            continue
        
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
            # Check if file already exists and skip if no-overwrite is enabled
            if no_overwrite and os.path.exists(final_output_path):
                print(f"⏭️ Skipping existing silence file: {final_output_path}")
                successful_lines += 1
                generated_files.append(final_output_path)
                subtitle_texts.append("...")  # Add ellipsis for silence marker
                continue
            # Eine längere Pause für die Ellipse
            created_path = create_silent_wav(duration_ms=700, filename=final_output_path)
            if created_path:
                successful_lines += 1 # Zählt als erfolgreiche "Zeile"
                generated_files.append(final_output_path)
                subtitle_texts.append("...")  # Add ellipsis for silence marker
            continue


        # Generate speech with quality control
        try:
            success, audio_path = generate_with_quality_control(
                tts_client=tts_client,
                text=clean_text,
                output_filename=output_filename,
                params=params,
                num_retries=num_retries
            )
            
            if success and audio_path:
                successful_lines += 1
                generated_files.append(audio_path)
                subtitle_texts.append(clean_text_with_emotions)  # Store original text with emotions for subtitles
                
        except TypeError as e:
            print(f"❌ Parameter error: {e}")
            print("⚠️ Attempting with default parameters only")
            try:
                # Try again with minimal parameters
                success, audio_path = generate_with_quality_control(
                    tts_client=tts_client,
                    text=clean_text,
                    output_filename=output_filename,
                    params={},
                    num_retries=num_retries
                )
                
                if success and audio_path:
                    successful_lines += 1
                    generated_files.append(audio_path)
                    subtitle_texts.append(clean_text_with_emotions)  # Store original text with emotions for subtitles
            except Exception as e2:
                print(f"❌ Failed with default parameters too: {e2}")
    
    print(f"\n✅ Completed file {file_path}: {successful_lines}/{total_lines} lines processed successfully")
    
    # Concatenate all generated audio files
    if successful_lines > 0 and HAS_PYDUB and generated_files:
        print(f"\n🔄 Concatenating {successful_lines} audio segments into a single file...")
        concat_output = f"{output_dir}/{base_name}_concat.{output_format}"
        concatenation_success, timing_info = concatenate_audio_files(generated_files, concat_output, format=output_format)
        
        # Create SRT subtitles if concatenation was successful
        if concatenation_success and timing_info and subtitle_texts:
            srt_output = f"{output_dir}/{base_name}_concat.srt"
            create_srt_subtitles(subtitle_texts, timing_info, srt_output)
    
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
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def main():
    """Main function to process text files and generate audio"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate audiobooks using Fish Speech TTS with quality evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python improved_audiobook_generator.py ./texts ./output voices.json
  python improved_audiobook_generator.py ./texts ./output voices.json --mapping mapping.csv --retries 5
  python improved_audiobook_generator.py ./texts ./output voices.json --no-overwrite
        """
    )
    
    parser.add_argument("input_dir", help="Directory containing text files to process")
    parser.add_argument("output_dir", help="Directory where audio files will be saved")
    parser.add_argument("voice_config_file", help="JSON file with voice configurations")
    parser.add_argument("--mapping", metavar="CSV_FILE", help="CSV file with word mappings")
    parser.add_argument("--retries", type=int, default=3, metavar="N", 
                       help="Number of generation attempts per line (default: 3)")
    parser.add_argument("--no-overwrite", action="store_true", 
                       help="Skip generation for existing WAV files")
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    voice_config_file = args.voice_config_file
    mapping_csv = args.mapping
    num_retries = args.retries
    no_overwrite = args.no_overwrite
    
    print(f"ℹ️ Number of generation attempts per line: {num_retries}")
    if no_overwrite:
        print("ℹ️ No-overwrite mode enabled: existing WAV files will be skipped")
    
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
        file_successful, file_total = process_text_file(
            file_path, 
            output_dir, 
            voice_config, 
            tts_client, 
            word_mapping,
            num_retries,
            no_overwrite
        )
        total_successful += file_successful
        total_lines += file_total
    
    # Print summary
    print("\n📊 Zusammenfassung:")
    print(f"🔹 {len(text_files)} Datei(en) verarbeitet")
    print(f"🔹 {total_successful}/{total_lines} Audiodateien generiert")
    print(f"🔹 {num_retries} Generierungsversuche pro Textzeile")
    if word_mapping:
        print(f"🔹 {len(word_mapping)} Wort-Mappings angewendet")
    if HAS_PYDUB:
        print(f"🔹 {len(text_files)} zusammengeführte Audiodateien erstellt")
    print(f"🔹 Ausgabeverzeichnis: {output_dir}")
    print(f"\n📅 Abgeschlossen am: {get_current_time_utc()}")

if __name__ == "__main__":
    main()