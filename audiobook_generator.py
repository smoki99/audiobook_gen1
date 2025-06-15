#!/usr/bin/env python3
"""
Audiobook Generator using Fish Speech TTS

This script processes text files, applies voice configurations from a JSON file,
and generates audio using the Fish Speech TTS API.
"""

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

# Check for required libraries
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    print("⚠️ pydub not installed. Audio concatenation disabled.")
    print("To enable concatenation, install pydub: pip install pydub")
    HAS_PYDUB = False

def load_mapping_csv(csv_path):
    mapping = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Annahme: CSV hat Spalten 'raw_text' und 'mapped_text'
            mapping[row['raw_text'].strip()] = row['mapped_text'].strip()
    return mapping
    
def extract_voice_tag(line):
    """Extract voice tag from text line and remove the tag"""
    # Erfasst Buchstaben, Zahlen, Unterstriche UND deutsche Umlaute
    voice_match = re.search(r'\[([a-zA-Z0-9-_äöüÄÖÜß\s]+)\]', line)
    if voice_match:
        voice = voice_match.group(1).strip()
        # Remove the voice tag from the line
        clean_line = re.sub(r'\[[a-zA-Z0-9-_äöüÄÖÜß\s]+\]', '', line).strip()
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

def concatenate_audio_files(audio_files, output_file, format="wav"):
    """Concatenate multiple audio files into a single file"""
    if not HAS_PYDUB:
        print("❌ Audio concatenation failed: pydub not installed")
        return False
    
    if not audio_files:
        print("⚠️ No audio files to concatenate")
        return False
    
    try:
        # Load the first audio file
        combined = AudioSegment.from_file(audio_files[0], format=format)
        
        # Add a small pause between segments (500ms)
        pause = AudioSegment.silent(duration=500)
        
        # Append the rest
        for audio_file in audio_files[1:]:
            segment = AudioSegment.from_file(audio_file, format=format)
            combined += pause + segment
        
        # Export the combined file
        combined.export(output_file, format=format)
        
        file_size = os.path.getsize(output_file) / 1024  # KB
        print(f"✅ Combined audio saved: {output_file} ({file_size:.1f} KB)")
        return True
    
    except Exception as e:
        print(f"❌ Error concatenating audio files: {e}")
        return False

def process_text_file(file_path, output_dir, voice_config, tts_client):
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
        voice, clean_text = extract_voice_tag(line)

        # Skip if clean text is empty after tag removal
        if not clean_text:
            continue
        
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
        concatenate_audio_files(generated_files, concat_output, format=output_format)
    
    return successful_lines, total_lines

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
    if len(sys.argv) != 4:
        print("Usage: python audiobook_generator.py <input_dir> <output_dir> <voice_config_file>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    voice_config_file = sys.argv[3]
    
    # Check if directories exist
    if not os.path.isdir(input_dir):
        print(f"❌ Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
        file_successful, file_total = process_text_file(file_path, output_dir, voice_config, tts_client)
        total_successful += file_successful
        total_lines += file_total
    
    # Print summary
    print("\n📊 Zusammenfassung:")
    print(f"🔹 {len(text_files)} Datei(en) verarbeitet")
    print(f"🔹 {total_successful}/{total_lines} Audiodateien generiert")
    if HAS_PYDUB:
        print(f"🔹 {len(text_files)} zusammengeführte Audiodateien erstellt")
    print(f"🔹 Ausgabeverzeichnis: {output_dir}")
    print(f"\n📅 Abgeschlossen am: {get_current_time_utc()}")

if __name__ == "__main__":
    main()