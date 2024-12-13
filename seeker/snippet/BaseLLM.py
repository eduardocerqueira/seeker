#date: 2024-12-13T17:10:57Z
#url: https://api.github.com/gists/ba169bf18cf4093d11f1cf563564d98c
#owner: https://api.github.com/users/sadbro

from enum import Enum
from typing import Callable
import requests
from PIL import Image
from io import BytesIO
import base64

class BaseLLMType(Enum):
    Llava = 0
    Phi3 = 1
    Flux = -1
    SDXL = -2

class QueryResponseType(Enum):
    Text = 0
    Image = 1

def _get_prompt_kernel(llm_type: BaseLLMType, response_type: QueryResponseType) -> Callable:
    if response_type == QueryResponseType.Text:
        match llm_type:
            case BaseLLMType.Llava:
                return lambda context, prompt: (
                    "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant".format(context, prompt)
                )
            case BaseLLMType.Phi3:
                return lambda context, prompt: (
                    "<|system|>\n{}<|end|>\n<|user|>\n{}<|end|>\n<|assistant|>".format(context, prompt)
                )
    elif response_type == QueryResponseType.Image:
        match llm_type:
            case BaseLLMType.Llava:
                return lambda context, base_64_encoded, prompt: (
                    "<|im_start|>system\n{}<|im_end|><|im_start|>user\n![]({})\n{}<|im_end|><|im_start|>assistant\n".format(
                        context, base_64_encoded, prompt
                    )
                )
            case l if l.value < 0:
                return lambda prompt: prompt

def _get_options_kernel(llm_type: BaseLLMType, response_type: QueryResponseType) -> Callable:
    if response_type == QueryResponseType.Text:
        match llm_type:
            case l if l.value >= 0:
                return lambda max_tokens, temperature: "**********"
                    {
                        "max_new_tokens": "**********"
                        "temperature": temperature,
                    }
                )
    elif response_type == QueryResponseType.Image:
        match llm_type:
            case l if l.value < 0:
                return lambda img_size, guidance_scale, num_inference_steps: (
                    {
                        "img_size": img_size,
                        "guidance_scale": guidance_scale,
                        "num_inference_steps": num_inference_steps,
                    }
                )

def _get_image_prompt(file_path: str, max_size: int = 200) -> str:
    image = Image.open(file_path)
    ratio = min(max_size / image.size[0], max_size / image.size[1])
    new_size = tuple([int(x * ratio) for x in image.size])
    image = image.resize(new_size, Image.Resampling.LANCZOS).convert('RGB')

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=50)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return "data:image/jpeg;base64,{}".format(base64_image)

class BaseLLMException(Exception):
    pass

class BaseLLM:
    def __init__(self, ep_url: str, api_key: str):
        self._endpoint = ep_url
        self._api_key = api_key
        self.headers = None
        self.options = None

    def get_base_query_headers(self, response_type: QueryResponseType) -> dict:
        base_response = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer {}'.format(self._api_key)
        }
        if response_type == QueryResponseType.Text:
            return {**base_response, 'Accept': 'application/json'}
        elif response_type == QueryResponseType.Image:
            return base_response

class LlavaClient(BaseLLM):

    def __init__(self, ep_url: str, api_key: str):
        super().__init__(ep_url, api_key)
        self.payload = None

    def set_headers(self, response_type: QueryResponseType = QueryResponseType.Text):
        self.headers = self.get_base_query_headers(response_type)

    def set_options(self, max_tokens: "**********": float, response_type: QueryResponseType = QueryResponseType.Text):
        self.options = _get_options_kernel(BaseLLMType.Llava, response_type)(
            max_tokens, temperature
        )

    def set_payload(self, context: str, prompt: str, image_path: str, response_type: QueryResponseType = QueryResponseType.Text):
        match response_type:
            case QueryResponseType.Text:
                self.payload = {
                    "inputs": _get_prompt_kernel(BaseLLMType.Llava, response_type)(context, prompt),
                    "parameters": self.options
                }
            case QueryResponseType.Image:
                self.payload = {
                    "inputs": _get_image_prompt(image_path),
                    "parameters": self.options
                }

    def get_response(self):
        return requests.post(
            self._endpoint,
            headers=self.headers,
            json=self.payload,
        )

class Phi3Client(BaseLLM):

    def __init__(self, ep_url: str, api_key: str):
        super().__init__(ep_url, api_key)
        self.payload = None

    def set_headers(self):
        self.headers = self.get_base_query_headers(QueryResponseType.Text)

    def set_options(self, max_tokens: "**********": float):
        self.options = _get_options_kernel(BaseLLMType.Phi3, QueryResponseType.Text)(
            max_tokens, temperature
        )

    def set_payload(self, context: str, prompt: str):
        self.payload = {
            "inputs": _get_prompt_kernel(BaseLLMType.Phi3, QueryResponseType.Text)(context, prompt),
            "parameters": self.options
        }

    def get_response(self):
        return requests.post(
            self._endpoint,
            headers=self.headers,
            json=self.payload,
        )

class FluxClient(BaseLLM):

    def __init__(self, ep_url: str, api_key: str):
        super().__init__(ep_url, api_key)
        self.payload = None

    def set_headers(self):
        self.headers = self.get_base_query_headers(QueryResponseType.Image)

    def set_options(self, img_size: int, guidance_scale: float, num_inference_steps: int):
        self.options = _get_options_kernel(BaseLLMType.Flux, QueryResponseType.Image)(
            img_size, guidance_scale, num_inference_steps
        )

    def set_payload(self, prompt: str):
        self.payload = {
            "prompt": _get_prompt_kernel(BaseLLMType.Flux, QueryResponseType.Image)(prompt),
            **self.options
        }

    def get_response(self):
        return requests.post(
            self._endpoint,
            headers=self.headers,
            json=self.payload,
        )

    def save_response(self, file_path: str):
        response = self.get_response()
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print("Image saved as {}".format(file_path))
        else:
            raise BaseLLMException(response.text)

class SDXLClient(BaseLLM):

    def __init__(self, ep_url: str, api_key: str):
        super().__init__(ep_url, api_key)
        self.payload = None

    def set_headers(self):
        self.headers = self.get_base_query_headers(QueryResponseType.Image)

    def set_options(self, img_size: int, guidance_scale: float, num_inference_steps: int):
        self.options = _get_options_kernel(BaseLLMType.SDXL, QueryResponseType.Image)(
            img_size, guidance_scale, num_inference_steps
        )

    def set_payload(self, prompt: str):
        self.payload = {
            "prompt": _get_prompt_kernel(BaseLLMType.SDXL, QueryResponseType.Image)(prompt),
            **self.options
        }

    def get_response(self):
        return requests.post(
            self._endpoint,
            headers=self.headers,
            json=self.payload,
        )

    def save_response(self, file_path: str):
        response = self.get_response()
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print("Image saved as image.png")
        else:
            raise BaseLLMException(response.text)
