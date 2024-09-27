#date: 2024-09-27T16:47:15Z
#url: https://api.github.com/gists/1eb46571837cfd0fb223cf58d5461ba6
#owner: https://api.github.com/users/Supra-San

import telebot
import openai
import requests

# Replace with your API key
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
TELEGRAM_API_TOKEN = "**********"

# OpenAI client initialization
openai.api_key = OPENAI_API_KEY

# Telegram bot initialization
bot = "**********"

# Function to generate images
def generate_image(prompt):
    response = openai.Image.create(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    return response.data[0].url

# Function to generate text
def generate_text(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens= "**********"
    )
    return response.choices[0].message['content'].strip()

# Display initial message
@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_message = "Please use /generate-image or /generate-text and continue with the text to be processed. As a generative ai agent, I will create content for you."
    bot.reply_to(message, welcome_message)

# Handle text messages
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    if message.text.startswith('/generate-image '):
        prompt = message.text[len('/generate-image '):]
        try:
            image_url = generate_image(prompt)
            response = requests.get(image_url)
            if response.status_code == 200:
                bot.send_photo(message.chat.id, response.content, caption="Result Image")
            else:
                bot.reply_to(message, "Failed to Process.")
        except Exception as e:
            bot.reply_to(message, f"There is an error: {e}")

    elif message.text.startswith('/generate-text '):
        prompt = message.text[len('/generate-text '):]
        try:
            text_response = generate_text(prompt)
            bot.reply_to(message, text_response)
        except Exception as e:
            bot.reply_to(message, f"There is an error: {e}")

    else:
        bot.reply_to(message, "Please use /generate-image or /generate-text followed by the text to be generated.")

# Run the bot continuously
bot.polling() generated.")

# Run the bot continuously
bot.polling()