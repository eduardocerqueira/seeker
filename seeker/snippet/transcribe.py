#date: 2024-12-17T16:58:35Z
#url: https://api.github.com/gists/5593ae6f12d3ff48df503c06489c627e
#owner: https://api.github.com/users/ghadena

# transcribing mp3 
import time

def transcribe_audio(job_name, s3_uri, primary_language="en-US"):
    """
    Transcribes an audio file from S3 using AWS Transcribe and returns the transcript URI.
    
    Parameters:
        job_name (str): Unique name for the transcription job.
        s3_uri (str): S3 URI of the audio file to transcribe.
        primary_language (str): Primary language code (default: "en-US").
        secondary_language (str): Secondary language code (default: "he-IL").
        
    Returns:
        str: URI of the transcription JSON file.
    """
    transcribe_client = boto3.client('transcribe')

    # Start the transcription job
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': s3_uri},
        MediaFormat='mp3',
        LanguageCode=primary_language,  # Specify the file format
        Settings={
            'ShowSpeakerLabels': True,
            'MaxSpeakerLabels': 10
        }
    )

    # Wait for the transcription job to complete
    while True:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        print(f"Waiting for {job_name} to complete...")
        time.sleep(5)

    # Check the final status
    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        print(f"Transcription completed. Transcript URI: {transcript_uri}")
        return transcript_uri
    else:
        raise RuntimeError(f"Transcription job {job_name} failed.")
    