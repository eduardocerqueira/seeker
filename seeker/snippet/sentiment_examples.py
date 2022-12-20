#date: 2022-12-20T16:59:12Z
#url: https://api.github.com/gists/f56db8b8d66348e23de9cc07bf631280
#owner: https://api.github.com/users/neil-assembly

import json
import re

f = open('transcript.json')
transcript = json.load(f)
sentiment_results = transcript['sentiment_analysis_results']

# NOTE: Optionally, you can choose to filter out sentiment results returned by our model that have low confidence scores.
# sentiment_results = [sa for sa in sentiment_results if sa['confidence'] > 0.70]

# Sentiment score and confidence rating per speaker block of the transcript
def sentiment_per_utterance_block():
  print('••• SENTIMENT PER UTTERANCE BLOCKS •••')
  utterances = transcript['utterances']

  for utt in utterances:

    # NOTE: Our sentiment model returns sentiment values per sentence.
    # So we will need to split the utterance block into sentences first.

    sentences = re.split('(?<=[.!?])\s+', utt['text'])

    utt_sentiments = []
    for sentence in sentences:
      try:
        sentence_match = [sa for sa in sentiment_results if sentence.strip() in sa['text']][0]
        if sentence_match:
          utt_sentiments.append(sentence_match)
      except:
        continue

    print(f'UTTERANCE BLOCK: {utt["text"]}')
    sentiment_results_positive = [sa for sa in utt_sentiments if sa['sentiment'] == 'POSITIVE']
    sentiment_results_negative = [sa for sa in utt_sentiments if sa['sentiment'] == 'NEGATIVE']

    percentage_positive = len(sentiment_results_positive) / len(utt_sentiments)
    percentage_negative = len(sentiment_results_negative) / len(utt_sentiments)

    print(f'{round(percentage_positive*100)}% of the utterance block was positive')
    print(f'{round(percentage_negative*100)}% of the utterance block was negative')
    print(f'{round(average_confidence_rating(utt_sentiments)*100)}% average confidence rating')
    print('\n')

# Overall call sentiment score and rating
def sentiment_per_transcript():
  sentiment_results_positive = [sa for sa in sentiment_results if sa['sentiment'] == 'POSITIVE']
  sentiment_results_negative = [sa for sa in sentiment_results if sa['sentiment'] == 'NEGATIVE']
  sentiment_results_neutral = [sa for sa in sentiment_results if sa['sentiment'] == 'NEUTRAL']

  percentage_positive = len(sentiment_results_positive) / len(sentiment_results)
  percentage_negative = len(sentiment_results_negative) / len(sentiment_results)

  print('••• SENTIMENT PER TRANSCRIPT •••')
  print(f'{round(percentage_positive*100)}% of all utterances were positive')
  print(f'{round(percentage_negative*100)}% of all utterances were negative')
  print(f'{len(sentiment_results_positive)} POSITIVE – {len(sentiment_results_negative)} NEGATIVE – {len(sentiment_results_neutral)} NEUTRAL')
  print(f'{round(average_confidence_rating(sentiment_results)*100)}% average confidence rating')
  print('\n')

# Agent sentiment score
def sentiment_per_agent():
  # NOTE: From the CallRail example files, the Agent is generally listed as Speaker B and the Customer is generally listed as Speaker C.

  sentiment_results_agent = [sa for sa in sentiment_results if sa['speaker'] == 'B']

  sentiment_results_positive = [sa for sa in sentiment_results_agent if sa['sentiment'] == 'POSITIVE']
  sentiment_results_negative = [sa for sa in sentiment_results_agent if sa['sentiment'] == 'NEGATIVE']
  sentiment_results_neutral = [sa for sa in sentiment_results_agent if sa['sentiment'] == 'NEUTRAL']

  percentage_positive = len(sentiment_results_positive) / len(sentiment_results_agent)
  percentage_negative = len(sentiment_results_negative) / len(sentiment_results_agent)

  print('••• AVERAGE AGENT SENTIMENT •••')
  print(f'{round(percentage_positive*100)}% of Agent utterances were positive')
  print(f'{round(percentage_negative*100)}% of Agent utterances were negative')
  print(f'{len(sentiment_results_positive)} POSITIVE – {len(sentiment_results_negative)} NEGATIVE – {len(sentiment_results_neutral)} NEUTRAL')
  print(f'{round(average_confidence_rating(sentiment_results_agent)*100)}% average confidence rating')
  print('\n')

# Customer sentiment score
def sentiment_per_customer():
  sentiment_results_customer = [sa for sa in sentiment_results if sa['speaker'] == 'C']

  sentiment_results_positive = [sa for sa in sentiment_results_customer if sa['sentiment'] == 'POSITIVE']
  sentiment_results_negative = [sa for sa in sentiment_results_customer if sa['sentiment'] == 'NEGATIVE']
  sentiment_results_neutral = [sa for sa in sentiment_results_customer if sa['sentiment'] == 'NEUTRAL']

  percentage_positive = len(sentiment_results_positive) / len(sentiment_results_customer)
  percentage_negative = len(sentiment_results_negative) / len(sentiment_results_customer)

  print('••• AVERAGE CUSTOMER SENTIMENT •••')
  print(f'{round(percentage_positive*100)}% of Customer utterances were positive')
  print(f'{round(percentage_negative*100)}% of Customer utterances were negative')
  print(f'{len(sentiment_results_positive)} POSITIVE – {len(sentiment_results_negative)} NEGATIVE – {len(sentiment_results_neutral)} NEUTRAL')
  print(f'{round(average_confidence_rating(sentiment_results_customer)*100)}% average confidence rating')
  print('\n')

def average_confidence_rating(sa_results):
  confidence_total = sum(sa['confidence'] for sa in sa_results)
  return confidence_total / len(sa_results)

sentiment_per_utterance_block()
sentiment_per_transcript()
sentiment_per_agent()
sentiment_per_customer()