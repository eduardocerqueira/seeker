#date: 2026-01-01T17:14:01Z
#url: https://api.github.com/gists/b58beeffda0cdd92445216bf91412bc8
#owner: https://api.github.com/users/darazacc828-eng

import os
import subprocess
import speech_recognition as sr
import openai
import time

# ===== CONFIG =====
openai.api_key = "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"  # <-- Apni OpenAI key yahan lagayein
language = "ur-PK"

def speak(text):
    os.system(f'espeak "{text}"')

def listen(timeout=5):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.listen(source, timeout=timeout)
            command = r.recognize_google(audio, language=language)
            print(f"You said: {command}")
            return command.lower()
        except:
            return ""

def ask_openai(question):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=question,
            max_tokens= "**********"
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"OpenAI error: {str(e)}"

def lock_phone():
    os.system("input keyevent 26")
    os.system("termux-vibrate -d 1000")
    print("Phone locked due to unauthorized access!")

def verify_owner():
    while True:
        command = listen()
        if "sir" in command or "ali" in command:
            speak("Ji Sir Ali, system unlocked.")
            break
        elif command != "":
            speak("Unauthorized voice detected! Locking system.")
            lock_phone()
        else:
            print("Waiting for owner...")

def run_command(command):
    if "wifi on" in command:
        os.system("termux-wifi-enable true")
        speak("WiFi on kar diya")
    elif "wifi off" in command:
        os.system("termux-wifi-enable false")
        speak("WiFi off kar diya")
    elif "battery" in command:
        battery = subprocess.getoutput("termux-battery-status")
        speak(f"Battery status: {battery}")
    elif "exit" in command:
        speak("Alvida Sir")
        exit()
    else:
        speak(ask_openai(command))

memory = {}
def learn_command(command):
    if command not in memory:
        memory[command] = time.time()
        print(f"Learned new command: {command}")

speak("Jarvis tayyar hai, Sir Ali")
verify_owner()

while True:
    command = listen(timeout=10)
    if command:
        learn_command(command)
        run_command(command)