#date: 2024-02-21T17:02:28Z
#url: https://api.github.com/gists/f64a4b29ce9f4b7c18e796c55bcdb610
#owner: https://api.github.com/users/mvandermeulen

import os
import json
import sys
import time
import threading

import redis
import tkinter as tk
import azure.cognitiveservices.speech as speechsdk
from openai import OpenAI

CONST_REDIS_HOST = "127.0.0.1"
CONST_REDIS_PORT = 6379

CONST_OPENAI_API_KEY = "sk-XXX"
SPEECH_KEY='xxx'
SPEECH_REGION='germanywestcentral'

client = OpenAI(api_key=CONST_OPENAI_API_KEY)
redis_client = redis.StrictRedis(host=CONST_REDIS_HOST, port=CONST_REDIS_PORT, db=0)


CONST_INST_ROLE = '''
You are an expert Software Engineer in providing Linux shell commands and writing code in different programming languages.
Follow every direction here when crafting your response:

Use natural, conversational language that is clear and easy to follow (short sentences, simple words).

Your response MUST be always in a valid JSON format ONLY string, having only one key containing the code or shell command.
If the response code or shell command has multiple lines, it must be encoded as a single line string using '\n' delimiter.
If your response contains message outside of the code, put all those messages as comments in the JSON code.

```
Example #1
User: How can I run a shell command using Python?
Response:
{
    "code": "import os\\nos.system('ls -al')"
}

Example #2
User: What is the command to show open ports in linux?
Response:
{
    "code": netstat -natp"
}
```
'''


CONST_ENHANCEMENT_ROLE = '''
You are an expert Software Engineer in providing Linux shell commands and writing code in different programming languages.
Follow every direction here when crafting your response:

You will help user to modify the code or shell commands enclosed within the triple backticks.

Your response MUST be always in a valid JSON format ONLY string, having only one key containing the code or shell command.
If the response code or shell command has multiple lines, it must be encoded as a single line string using '\n' delimiter.
If your response contains message outside of the code, put all those messages as comments in the JSON code.

---
Example #1
User:
Help me to modify this command to list files in a neat format that is readable by human.

```
ls
```
Response:
{
    "code": "ls -alh"
}


Example #2
User:
Change this Python code to display current date time as output.
```
print("Hello world")
```

Response:
{
    "code": "from datetime import datetime \\nprint(datetime.now())"
}

---
'''


CONST_ROUTING_ROLE = '''
You are an AI assistant listening to the user. You analyze the user message to categorize whether it is one of the below.

1. Instruction prompt - User is asking for help to provide some code or shell commands.
2. Enhancement prompt - User is asking to editing an existing piece of code that the user will provide in the next message.
3. Other prompt - User is talking to other people and you can ignore the message

Follow every direction here when crafting your response:

Your response MUST be always a single string only. [INSTRUCTION, ENHANCEMENT or OTHERS]

```
Example #1:
User: Hello, this is a demo video that I'm recording to show how we can perform prompt routing technique
Response:
OTHERS

Example #2:
User: How can I check my existing iptables rules in Linux?
Response:
INSTRUCTION

Example #3:
User: Help me to modify line 5 to line 16 by changing the nested if-else code using a switch statement.
Response:
ENHANCEMENT
```

'''

def create_centered_bottom_window(text):
    window = tk.Tk()
    window.configure(bg="black")
    window.overrideredirect(True)

    label = tk.Label(window, text=text, fg="white", bg="black", font=("Helvetica", 30))
    label.pack(padx=0, pady=0)

    # I've added minus 1920 due to having dual screen, where my backup display is in 1920x1080 res
    screen_width = window.winfo_screenwidth() - 1920
    screen_height = window.winfo_screenheight()

    label_width = label.winfo_reqwidth()
    label_height = label.winfo_reqheight()

    window_width = min(label_width, screen_width)
    window_height = min(label_height, screen_height)
    window.geometry(f"{window_width}x{window_height}")

    x_position = (screen_width - window_width) // 2
    y_position = screen_height - window_height - 20

    window.geometry(f"+{x_position}+{y_position}")
    window.update_idletasks()
    return window

def clear_msg(window):
    if window:
        try:
            window.destroy()
        except:
            pass

def recognize_from_microphone():
    os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language="en-US"
    #speech_config.speech_recognition_language="zh-CN"
    #speech_config.speech_recognition_language="de-DE"

    audio_config = speechsdk.audio.AudioConfig(device_name='pipewire')
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    speech_recognizer.recognizing.connect(recognizing)
    speech_recognizer.recognized.connect(recognized)
    #speech_recognizer.session_started.connect(lambda evt: print('Speech recognition started'))
    #speech_recognizer.session_stopped.connect(lambda evt: print('Speech recognition stopped'))
    speech_recognizer.canceled.connect(lambda evt: print('Speech recognition error!'))
    #speech_recognizer.session_stopped.connect(stop_cb)
    #speech_recognizer.canceled.connect(stop_cb)
    speech_recognizer.start_continuous_recognition()

    try:
        while True:
            time.sleep(.5)
    except:
        speech_recognizer.stop_continuous_recognition()

def recognizing(evt=None):
    redis_client.publish(
        "recognizing",
        evt.result.text.strip()
    )

def recognized(evt=None):
    redis_client.publish(
        "recognized",
        evt.result.text.strip()
    )


def stop_cb(evt=None):
    speech_recognizer.stop_continuous_recognition()


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    return completion.choices[0].message.content


def parse_response(response):
    try:
        return json.loads(response)
    except:
        return None


def processor():
    window = None
    window = create_centered_bottom_window("Starting code assist...")
    subscriber = redis_client.pubsub()
    subscriber.subscribe("recognized")
    subscriber.subscribe("recognizing")

    time.sleep(2)
    clear_msg(window)
    window = create_centered_bottom_window("Redis is connected. Listening...")

    while True:
        azure_msg = subscriber.get_message()

        if azure_msg and azure_msg["type"] == "message":
            msg = azure_msg["data"].decode("utf-8")
            channel = azure_msg["channel"].decode("utf-8")

            if channel == "recognized" or "recognizing":
                clear_msg(window)
                window = create_centered_bottom_window(msg)

                if channel == "recognized":
                    messages = [{'role':'system', 'content': CONST_ROUTING_ROLE}]
                    user_input = {'role':'user', 'content': msg}
                    messages.append(user_input)

                    clear_msg(window)
                    window = create_centered_bottom_window("***ChatGPT is thinking...***")
                    response = get_completion_from_messages(messages)

                    process_response(window, msg, response)
        time.sleep(0.2)


def process_response(window, msg, response):
    if response == "INSTRUCTION":
        messages = [{'role':'system', 'content': CONST_INST_ROLE}]
        user_input = {'role':'user', 'content': msg}
        messages.append(user_input)

        clear_msg(window)
        window = create_centered_bottom_window("***Instruction prompt received. ChatGPT is thinking...***")
        response = get_completion_from_messages(messages)
        parsed_obj = parse_response(response)
        clear_msg(window)

        if parsed_obj:
            f = open("response.txt", "w")
            f.write(parsed_obj['code'])
            f.close()
        else:
            window = create_centered_bottom_window("Response does not comply with expected format. Try again")


    elif response == "ENHANCEMENT":
        f = open("response.txt", "r")
        code = f.read()
        f.close()

        messages = [{'role':'system', 'content': CONST_ENHANCEMENT_ROLE}]
        new_prompt = f"{msg}\n\n```{code}```"
        user_input = {'role':'user', 'content': new_prompt}
        messages.append(user_input)

        clear_msg(window)
        window = create_centered_bottom_window("***Enhancement prompt received. ChatGPT is thinking...***")
        response = get_completion_from_messages(messages)
        parsed_obj = parse_response(response)
        clear_msg(window)

        if parsed_obj:
            f = open("response.txt", "w")
            f.write(parsed_obj['code'])
            f.close()
        else:
            window = create_centered_bottom_window("Response does not comply with expected format. Try again")

    else:
        clear_msg(window)
        window = create_centered_bottom_window("***Others. Speech ignored...***")




if __name__ == "__main__":
    print("Tkinter window initialized, switching output to Tk window as subtitle...")
    threading.Thread(target=processor).start()
    recognize_from_microphone()
