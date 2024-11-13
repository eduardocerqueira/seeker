#date: 2024-11-13T17:09:10Z
#url: https://api.github.com/gists/2acd9475fa9d8491ce94a1729f42003e
#owner: https://api.github.com/users/me-suzy

import openai
import os
import requests
from datetime import datetime

# OpenAI API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Cheia API OpenAI nu a fost setată în variabilele de mediu.")
openai.api_key = OPENAI_API_KEY

def test_generate_image(prompt):
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="url",
        )
        image_url = response['data'][0]['url']
        image_response = requests.get(image_url)

        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"test_image_{timestamp}.png"

        with open(image_path, 'wb') as f:
            f.write(image_response.content)

        print(f"Image saved as: {image_path}")
    except Exception as e:
        print(f"Error generating image: {str(e)}")

# Prompt de test
test_prompt = "A photorealistic image of a middle-aged male athlete running in a marathon, shot on Canon 5D Mark IV."
test_generate_image(test_prompt)
