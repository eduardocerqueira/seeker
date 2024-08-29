#date: 2024-08-29T16:50:18Z
#url: https://api.github.com/gists/e1cc87e813835f151f8a342a16764b25
#owner: https://api.github.com/users/aphexlog

import json
import boto3
import logging
import pyaudio
from botocore.exceptions import ClientError
from mypy_boto3_bedrock_runtime.client import BedrockRuntimeClient as BedrockClient
from typing import cast
from pydub import AudioSegment
from io import BytesIO

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DragonChatHandler:
    def __init__(self, region: str):
        self.client = cast(BedrockClient, boto3.client("bedrock-runtime", region_name=region))
        self.polly_client = boto3.client('polly', region_name=region)

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"n "**********"e "**********"r "**********"a "**********"t "**********"e "**********"_ "**********"m "**********"e "**********"s "**********"s "**********"a "**********"g "**********"e "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"m "**********"o "**********"d "**********"e "**********"l "**********"_ "**********"i "**********"d "**********", "**********"  "**********"s "**********"y "**********"s "**********"t "**********"e "**********"m "**********"_ "**********"p "**********"r "**********"o "**********"m "**********"p "**********"t "**********", "**********"  "**********"m "**********"e "**********"s "**********"s "**********"a "**********"g "**********"e "**********"s "**********", "**********"  "**********"m "**********"a "**********"x "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********": "**********"
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": "**********"
            "system": system_prompt,
            "messages": messages
        })

        try:
            response = self.client.invoke_model(body=body, modelId=model_id)
            response_body = json.loads(response.get('body').read())
            return response_body
        except ClientError as err:
            message = err.response["Error"]["Message"]
            logger.error("A client error occurred: %s", message)
            raise

    def send_message(self, message, model="anthropic.claude-3-haiku-20240307-v1:0"):
        user_message = {"role": "user", "content": message}
        messages = [user_message]
        system_prompt = "Please respond to the user's message."
        max_tokens = "**********"

        return self.generate_message(model, system_prompt, messages, max_tokens)

    def start_conversation(self, initial_message):
        return self.send_message(initial_message)

    def continue_conversation(self, message):
        return self.send_message(message)

    def speak_response(self, response_string):
        response = self.polly_client.synthesize_speech(
            Text=response_string,
            OutputFormat='mp3',
            VoiceId='Joanna'
        )

        if "AudioStream" in response:
            # Convert MP3 to PCM using pydub
            audio_stream = response['AudioStream'].read()
            sound = AudioSegment.from_mp3(BytesIO(audio_stream))
            raw_data = sound.raw_data
            sample_width = sound.sample_width
            channels = sound.channels
            frame_rate = sound.frame_rate

            # Play the audio
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(sample_width),
                            channels=channels,
                            rate=frame_rate,
                            output=True)

            stream.write(raw_data)

            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    chat_handler = DragonChatHandler("us-east-1")

    # Continue the conversation in a loop
    while True:
        user_input = input("Input message: ")
        response = chat_handler.continue_conversation(user_input)
        response_string = response["content"][0]["text"]

        print(f"{response_string}\n")
        chat_handler.speak_response(response_string)
