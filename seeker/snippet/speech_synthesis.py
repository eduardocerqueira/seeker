#date: 2023-03-31T17:07:16Z
#url: https://api.github.com/gists/ad795867cfd3005ff03a9ea5ef2d7dff
#owner: https://api.github.com/users/vulture0902

'''
Synthesize to speaker output
https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/get-started-text-to-speech?tabs=linux%2Cterminal&pivots=programming-language-python#synthesize-to-speaker-output

# Language and voice support for the Speech service
https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support?tabs=tts
'''

import os
import azure.cognitiveservices.speech as speechsdk

# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
speech_config = speechsdk.SpeechConfig('xxxxxxxxxxxxxxxxxxx', 'japanwest')
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

# The language of the voice that speaks.
speech_config.speech_synthesis_voice_name='km-KH-PisethNeural'

speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

# Get text from the console and synthesize to the default speaker.
text = 'ខ្ញុំទៅផ្សារ'

speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
    print("Speech synthesized for text [{}]".format(text))
elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
    cancellation_details = speech_synthesis_result.cancellation_details
    print("Speech synthesis canceled: {}".format(cancellation_details.reason))
    if cancellation_details.reason == speechsdk.CancellationReason.Error:
        if cancellation_details.error_details:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")