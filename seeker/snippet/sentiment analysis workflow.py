#date: 2024-12-17T16:54:35Z
#url: https://api.github.com/gists/67b1fc0cd46ae0cd73fc42c5101f421c
#owner: https://api.github.com/users/ghadena

# running sentiment analysis for amnesty data 

with open("amnesty.json", "r") as file:
    t2 = json.load(file)

t2['results']['audio_segments'][0]['speaker_label']
t2['results']['audio_segments'][0]['transcript']

#List comprehension to extract ID, transcript, and speaker label from json 
audio_segments = t2['results']['audio_segments']

extracted_data = [
    {'ID': segment['id'], 
     'Transcript': segment['transcript'], 
     'Speaker': segment['speaker_label']}
    for segment in audio_segments
]

# Group transcripts by speaker
from collections import defaultdict
grouped_by_speaker = defaultdict(list)

for entry in extracted_data:
    grouped_by_speaker[entry['Speaker']].append(entry['Transcript'])

# Combine all sentences per speaker into a single string
speaker_transcriptions = {speaker: " ".join(transcripts) 
                          for speaker, transcripts in grouped_by_speaker.items()}

# Split transcriptions for each speaker into chunks
speaker_chunks = {speaker: split_text(transcript) 
                  for speaker, transcript in speaker_transcriptions.items()}

# Initialize AWS Comprehend
comprehend = boto3.client('comprehend', region_name='us-east-1')

# Function to analyze sentiment
def analyze_sentiment(text):
    response = comprehend.detect_sentiment(Text=text, LanguageCode='en')
    return response['Sentiment'], response['SentimentScore']

# Analyze sentiment for each speaker's chunks
speaker_sentiments = {}

for speaker, chunks in speaker_chunks.items():
    sentiments = []
    for chunk in chunks:
        sentiment, sentiment_score = analyze_sentiment(chunk)
        sentiments.append({'Sentiment': sentiment, 'Score': sentiment_score})
    speaker_sentiments[speaker] = sentiments

for speaker, sentiment_list in speaker_sentiments.items():
    print(f"Sentiments for {speaker}:")
    for idx, result in enumerate(sentiment_list):
        print(f"  Chunk {idx + 1}: Sentiment = {result['Sentiment']}, Scores = {result['Score']}")

output_data = []

# Loop through the sentiment results and collect data
for speaker, sentiment_list in speaker_sentiments.items():
    for idx, result in enumerate(sentiment_list):
        output_data.append({
            'Speaker': speaker,
            'Chunk': idx + 1,
            'Sentiment': result['Sentiment'],
            'Positive': result['Score']['Positive'],
            'Negative': result['Score']['Negative'],
            'Neutral': result['Score']['Neutral'],
            'Mixed': result['Score']['Mixed']
        })

# Convert the list to a DataFrame
df2 = pd.DataFrame(output_data)

# Save the DataFrame to a CSV file
df2.to_csv('speaker_sentiment_results_amnesty.csv', index=False)

print("Sentiment results saved to 'speaker_sentiment_results_amnesty.csv'")

amnesty_speaker_avg_scores = {}

# Loop through the results and calculate averages
for speaker, results in speaker_sentiments.items():
    avg_positive = np.mean([r['Score']['Positive'] for r in results])
    avg_negative = np.mean([r['Score']['Negative'] for r in results])
    avg_neutral = np.mean([r['Score']['Neutral'] for r in results])
    
    # Save the averages into the dictionary
    amnesty_speaker_avg_scores[speaker] = {
        'Avg Positive': round(avg_positive, 2),
        'Avg Negative': round(avg_negative, 2),
        'Avg Neutral': round(avg_neutral, 2)
    }

# Display the stored results
print(amnesty_speaker_avg_scores)