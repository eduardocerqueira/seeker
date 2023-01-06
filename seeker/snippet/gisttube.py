#date: 2023-01-06T16:32:24Z
#url: https://api.github.com/gists/b340339dda0e865960d2fd864aa12280
#owner: https://api.github.com/users/suraj813

import os
import re
import csv
import torch
import click
from typing import List

import whisper
from yt_dlp import YoutubeDL
from transformers import AutoTokenizer, pipeline

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

NLP_ARCH = 'philschmid/bart-large-cnn-samsum'

def create_safe_filename(unsafe_string):
    """Replace all non-alphanumeric characters with underscores and return the modified string."""
    safe_string = re.sub(r'[^\w\s]', '_', unsafe_string)
    safe_string = re.sub(r'_+', '_', safe_string)
    safe_string = safe_string.strip('_')
    return safe_string

def save_transcript_to_csv(asr_result, file_path):
    """
    Save transcription to a CSV file.
    
    Parameters:
        asr_result: The transcription data to save.
        file_path (str): The file path of the CSV file.
    """
    field_names = ['start', 'end', 'text', 'summary']
    with open(file_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for entry in asr_result:
            writer.writerow({k:entry[k] for k in field_names if k in entry.keys()})

def load_csv_to_transcript(file_path):
    """
    Load transcription from a CSV file.
    
    Parameters:
        file_path (str): The file path of the CSV file.
    
    Returns:
        list: The transcription data.
    """
    asr_result = []
    with open(file_path, "r") as csv_file:
        # Create a CSV reader
        reader = csv.DictReader(csv_file)
        for row in reader:
            asr_result.append(row)
    return asr_result

def download_youtube_video(video_url: str):
    """
    Download audio from a YouTube video.
    
    Parameters:
        video_url (str): The URL of the YouTube video.
    
    Returns:
        str: The file path of the audio file.
    """
    # Create folder for assets
    with YoutubeDL({'quiet':True}) as ydl:
        dir_name = ydl.prepare_filename(ydl.extract_info(video_url, download=False)).split('.')[0]
    dir_name = create_safe_filename(dir_name)
    audio_path = f"{dir_name}/audio.mp3"

    # Download audio file if it does not exist
    if not os.path.exists(audio_path):
        YDL_OPTS = {
            'quiet':True,
            'format': 'bestaudio/best',
            'outtmpl': audio_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '128',
        }]}
        with YoutubeDL(YDL_OPTS) as ydl:
            ydl.download([video_url])
        
    return audio_path


def transcribe_audio(audio_path: str, asr_model):
    """
    Transcribe an audio file using the provided ASR model.
    
    Parameters:
        audio_path (str): The file path of the audio file.
        asr_model: The ASR model to use for transcription.
    
    Returns:
        str: The file path of the transcript.
    """
    # Get the directory name and transcript file path
    dir_name = os.path.dirname(audio_path)
    transcript_file = f"{dir_name}/transcript.csv"

    # Transcribe audio if transcript file does not exist
    if not os.path.exists(transcript_file):        
        with torch.no_grad():
            # Transcribe audio file and get transcription segments
            asr_result = asr_model.transcribe(audio_path)['segments']
        # Save transcription to a CSV file
        save_transcript_to_csv(asr_result, transcript_file)
        
    # Load transcription from CSV file
    asr_result = load_csv_to_transcript(transcript_file)
    return transcript_file    

def tokenize_with_timestamps(asr_result: "**********":
    """
    Tokenize transcribed text, chunking into segments of length NLP_MAXLEN 
    while preserving timestamps from ASR transcription.
    
    Parameters:
        asr_result (List[dict]): The transcription segments. Each dict should have a 'text' and 
            'start' and 'end' keys for the transcription text and start and end times, respectively.
    
    Returns:
        List[dict]: "**********"
            and 'end' keys for the sentence text, tokenized form, and start and end times, respectively.
    """
    # Initialize NLP tokenizer and maximum length of tokens
    NLP_TOKENIZER = "**********"
    NLP_MAXLEN = "**********"
    # Initialize list to store tokenized sentences and current sentence data
    sent_tokens = "**********"
    d = {}
    # Tokenize each transcription segment
    for seg in asr_result:
        seg_tokens = "**********"=False)['input_ids']
        # If current tokens plus segment tokens exceed maximum length, add current sentence data to list
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"l "**********"e "**********"n "**********"( "**********"s "**********"e "**********"g "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********"  "**********"+ "**********"  "**********"l "**********"e "**********"n "**********"( "**********"d "**********". "**********"g "**********"e "**********"t "**********"( "**********"' "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"' "**********", "**********"  "**********"[ "**********"] "**********") "**********") "**********"  "**********"> "**********"= "**********"  "**********"N "**********"L "**********"P "**********"_ "**********"M "**********"A "**********"X "**********"L "**********"E "**********"N "**********": "**********"
            sent_tokens.append(d)
            d = {}
        # If current sentence data does not have a start time, set it
        if 'start' not in d.keys(): 
            d['start'] = seg['start']
        # Add segment text and tokens to current sentence data
        curr_tokens = "**********"
        curr_text = d.get('text', "")
        d['text'] = curr_text + seg['text']
        d['tokens'] = "**********"
        d['end'] = seg['end']

    # Add final sentence data to list
    sent_tokens.append(d)
    return sent_tokens

def get_transcript_summary(transcript_file: str, summary_lengths: int):
    """
    Generate a summary of a transcript and save it to a file.
    
    Parameters:
        transcript_file (str): The file path of the transcript.
        summary_lengths (int): The maximum length of the summary.
    
    Returns:
        str: The file path of the summary.
    """
    # Load the transcript from a file
    transcript = load_csv_to_transcript(transcript_file)
    
    # Tokenize the transcript and add timestamps
    timestamped_sentences = "**********"
    
    print("[NLP] Generating summary of transcription")
    
    # Extract the sentences from the timestamped transcript
    sentences = [sent['text'] for sent in timestamped_sentences]
    
    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model=NLP_ARCH)
    
    # Generate summaries for the sentences
    summaries = summarizer(sentences, max_length=summary_lengths, min_length=20, do_sample=False)
    
    # Add the summaries to the original timestamped sentences
    for a, b in zip(timestamped_sentences, summaries):
        a['summary'] = b['summary_text']
    
    # Save the timestamped summaries to a file
    summary_file = os.path.dirname(transcript_file)+"/summary.csv"
    save_transcript_to_csv(timestamped_sentences, summary_file)
    print("[NLP] Video summary saved at ", summary_file)

    return summary_file

def print_timestamped_summaries(summary_file):
    """
    Print timestamped summaries from a summary file.
    
    Parameters:
        summary_file (str): The file path of the summary file.
    """
    # Load the summary segments from the file
    segments = load_csv_to_transcript(summary_file)
    
    # Iterate through the segments and print their start and end times and summaries
    for seg in segments:
        print(f"{format_time(seg['start'])} - {format_time(seg['end'])}")
        print(seg['summary'])
        print()

def format_time(t):
    """
    Convert a time in seconds to a string in HH:MM:SS format.
    
    Parameters:
        t (str): The time in seconds.
    
    Returns:
        str: The time in HH:MM:SS format.
    """
    t = round(float(t))
    hh = t // 3600
    t %= 3600
    mm = t // 60
    ss = t % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

@click.command()
@click.option('--youtube_url', '-u', required=True, help='The URL of the YouTube video to summarize')
@click.option('--asr_model_size', '-m', default='small.en', type=click.Choice(['tiny.en', 'base.en', 'small.en', 'medium.en']), help='The size of the ASR model to use')
@click.option('--summary_lengths', '-l', default=128, help='Segment summary max length. Choose a larger value for longer and conversation-dense videos')
def main(youtube_url, asr_model_size, summary_lengths):
    # Load ASR model
    asr_model = whisper.load_model(asr_model_size).to(device)
    # Download and transcribe audio from YouTube video
    audio = download_youtube_video(youtube_url)
    transcript = transcribe_audio(audio, asr_model)
    # Generate summary of transcription
    summary = get_transcript_summary(transcript, summary_lengths)
    # Print timestamped summaries
    print_timestamped_summaries(summary)

if __name__=="__main__":
    main()