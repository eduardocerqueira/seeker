#date: 2025-08-28T17:09:06Z
#url: https://api.github.com/gists/87df97d7f6512481a6bd08ef9935375c
#owner: https://api.github.com/users/yhyang201

from openai import OpenAI

port = 30000

client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
                    },
                },
            ],
        }
    ],
    max_tokens= "**********"
)

print(response.choices[0].message.content)