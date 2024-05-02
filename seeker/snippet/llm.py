#date: 2024-05-02T16:59:31Z
#url: https://api.github.com/gists/4be889c448a26b7c4df669af4d496496
#owner: https://api.github.com/users/tyschacht

import base64
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from modules import parsers
import openai

# Load environment variables from .env file
load_dotenv()
api_key = os.environ["GOOGLE_API_KEY"]

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize Google API Client
genai.configure(api_key=api_key)


def gpro_1_5_prompt(prompt) -> str:
    """
    Generates content based on the provided prompt using the Gemini 1.5 API model and returns the text part of the first candidate's content.

    Args:
    - prompt (str): The prompt to generate content for.

    Returns:
    - str: The text part of the first candidate's content from the generated response.
    """
    model_name = "models/gemini-1.5-pro-latest"
    gen_config = genai.GenerationConfig()
    model = genai.GenerativeModel(model_name=model_name)
    response = model.generate_content(prompt, request_options={})
    return response.candidates[0].content.parts[0].text


def gpro_1_5_prompt_with_model(prompt, pydantic_model: BaseModel) -> BaseModel:
    """
    Generates content based on the provided prompt using the Gemini 1.5 API model and returns the text part of the first candidate's content.

    Args:
    - prompt (str): The prompt to generate content for.

    Returns:
    - str: The text part of the first candidate's content from the generated response.
    """
    model_name = "models/gemini-1.5-pro-latest"
    gen_config = genai.GenerationConfig()
    model = genai.GenerativeModel(model_name=model_name)
    response = model.generate_content(prompt, request_options={})
    response_text = response.candidates[0].content.parts[0].text
    if "```json" in response_text:
        return pydantic_model.model_validate(
            parsers.parse_json_from_gemini(response_text)
        )
    else:
        return pydantic_model.model_validate_json(response_text)


def gpt4t_w_vision_json_prompt(
    prompt: str,
    model: str = "gpt-4-turbo-2024-04-09",
    instructions: str = "You are a helpful assistant that response in JSON format.",
    pydantic_model: BaseModel = None,
) -> str:
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": instructions,  # Added instructions as a system message
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format={"type": "json_object"},
    )

    response_text = response.choices[0].message.content
    print(f"Text LLM response: {response_text}")

    as_model = pydantic_model.model_validate_json(response_text)

    return as_model


def gpt4t_w_vision(
    prompt: str,
    model: str = "gpt-4-turbo-2024-04-09",
    instructions: str = "You are a helpful assistant.",
) -> str:
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": instructions,  # Added instructions as a system message
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    response_text = response.choices[0].message.content
    return response_text


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def gpt4t_w_vision_image_with_model(
    prompt: str,
    file_path: str,
    model: str = "gpt-4-turbo-2024-04-09",
    instructions: str = "You are a helpful assistant that specializes in image analysis.",
    pydantic_model: BaseModel = None,
):

    file_extension = file_path.split(".")[-1]

    base64_image = encode_image(file_path)

    print("base64_image", base64_image)

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": instructions,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{file_extension};base64,{base64_image}"
                        },
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
    )

    print("response", response)

    response_text = response.choices[0].message.content

    print("response_text", response_text)

    parsed_response = pydantic_model.model_validate_json(response_text)

    return parsed_response
