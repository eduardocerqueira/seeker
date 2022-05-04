#date: 2022-05-04T17:00:46Z
#url: https://api.github.com/gists/11b631a2be1328ab28006177b4d5c5a9
#owner: https://api.github.com/users/insightsbees

def translation_func(input_language, output_language, text):
    translator = Translator()
    translation = translator.translate(text, src=input_language, dest=output_language)
    translation_text = translation.text
    tts = gTTS(translation_text, lang=output_language,  slow=True)
    try:
        audio_file = text[0:20]
    except:
        audio_file = "audio"
    tts.save(f"temp_folder/{audio_file}.mp3")
    return audio_file, translation_text