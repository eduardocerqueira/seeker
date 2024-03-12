#date: 2024-03-12T16:56:09Z
#url: https://api.github.com/gists/b0427b83db51d529d6b9d3aa099f8c4e
#owner: https://api.github.com/users/willwade

import requests

# Azure subscription key and service region
subscription_key = 'YourAzureSubscriptionKey'
service_region = 'YourServiceRegion'

# Set up the TTS endpoint
tts_endpoint = f'https://{service_region}.tts.speech.microsoft.com/cognitiveservices/v1'

# Set up the headers for the HTTP request
headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    'Content-Type': 'application/ssml+xml',
    'X-Microsoft-OutputFormat': 'audio-16khz-32kbitrate-mono-mp3'
}

# The SSML document, including IPA notation for the word "dog"
ssml = """
<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
    <voice name='en-US-AriaNeural'>
        <phoneme alphabet='ipa' ph='dɔg'>dog</phoneme>
    </voice>
</speak>
"""

# Make the HTTP request to the Azure TTS service
response = requests.post(tts_endpoint, headers=headers, data=ssml)

# Check if the request was successful
if response.status_code == 200:
    # Save the audio to a file
    with open('output.mp3', 'wb') as audio_file:
        audio_file.write(response.content)
    print("Audio saved to output.mp3")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
