#date: 2025-12-26T16:55:49Z
#url: https://api.github.com/gists/29d791e637ad896caa5c4f18c9f9467c
#owner: https://api.github.com/users/Auax

import os
import whisperx
import torch
from deep_translator import GoogleTranslator
import datetime

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

def format_timestamp(seconds: float) -> str:
    """
    Converts seconds (float) to SRT timestamp format: HH:MM:SS,mmm
    """
    td = datetime.timedelta(seconds=seconds)
    # Total seconds
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds_part = total_seconds % 60
    milliseconds = int(td.microseconds / 1000)
    
    return f"{hours:02}:{minutes:02}:{seconds_part:02},{milliseconds:03}"

def write_srt(segments, filename, key="text"):
    """
    Writes segments to an SRT file.
    key: the key in the segment dict to use for the subtitle text (e.g., 'text' or 'translation')
    """
    print(f"Writing {filename}...")
    with open(filename, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment.get(key, "").strip()
            
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")
    print(f"Saved: {filename}")

def main():
    print(f"------------------------------------------------")
    print(f"WhisperX Local Subtitle Generator")
    print(f"Device: {DEVICE}")
    print(f"------------------------------------------------")

    # 1. Inputs
    video_path = input("Enter the path to the video file: ").strip().strip('"')
    if not os.path.exists(video_path):
        print("Error: File not found.")
        return

    model_input = input("Enter Whisper model size (tiny, base, small, medium, large-v2, large-v3) [default: small]: ").strip()
    model_size = model_input if model_input else "small"

    target_lang_input = input("Enter target language code (e.g. en, es, fr) [default: en]: ").strip()
    target_lang = target_lang_input if target_lang_input else "en"

    print(f"------------------------------------------------")
    print(f"Selected Model: {model_size}")
    print(f"Target Language: {target_lang}")
    print(f"------------------------------------------------")

    # 2. Load Model
    print("Loading Transcription Model... (This may take a minute)")
    try:
        model = whisperx.load_model(model_size, DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"Error loading model '{model_size}': {e}")
        return
    print("Model Loaded successfully!")

    # 3. Transcribe
    print(f"Transcribing {os.path.basename(video_path)}...")
    audio = whisperx.load_audio(video_path)
    result = model.transcribe(audio, batch_size=16)
    
    source_lang = result["language"]
    print(f"Transcription complete (Detected: {source_lang}). Aligning...")

    # 4. Align
    model_a, metadata = whisperx.load_align_model(language_code=source_lang, device=DEVICE)
    aligned_result = whisperx.align(
        result["segments"], 
        model_a, 
        metadata, 
        audio, 
        device=DEVICE, 
        return_char_alignments=False
    )
    
    del model_a
    torch.cuda.empty_cache()

    # 5. Prepare Logic for Translation
    print(f"Translating segments from '{source_lang}' to '{target_lang}'...")
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    
    # Process segments
    processed_segments = []
    
    for segment in aligned_result["segments"]:
        original_text = segment["text"].strip()
        translated_text = original_text # Default to original if same language

        if source_lang != target_lang:
            try:
                translated_text = translator.translate(original_text)
            except Exception as e:
                print(f"Translation warning for segment: {e}")
                translated_text = original_text # Fallback

        processed_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": original_text,
            "translation": translated_text
        })

    # 6. Save SRT Files
    base_name = os.path.splitext(video_path)[0]
    
    # Original Language SRT
    original_srt_path = f"{base_name}_{source_lang}.srt"
    write_srt(processed_segments, original_srt_path, key="text")

    # Translated SRT (if different)
    if source_lang != target_lang:
        english_srt_path = f"{base_name}_{target_lang}.srt"
        write_srt(processed_segments, english_srt_path, key="translation")
    else:
        print(f"Source language is already {target_lang}, skipping translated SRT.")

    print("Done!")

if __name__ == "__main__":
    main()