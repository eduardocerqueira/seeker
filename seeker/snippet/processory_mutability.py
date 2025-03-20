#date: 2025-03-20T17:07:39Z
#url: https://api.github.com/gists/f003323ac4c8940f779f44a24b815ff7
#owner: https://api.github.com/users/nph4rd

# using `transformers==4.49.0` there's a input side-effect when calling the processor
# this example shows the case for Qwen2.5-VL

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

messages1 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
messages2 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "is there a person here?"},
        ],
    }
]

messages = [messages1, messages2]

print("="*40)
print("MULTIPLE MESSAGES")
texts = "**********"=False, add_generation_prompt=True) for msg in messages]
image_inputs, video_inputs = process_vision_info(messages)
texts_copy = texts.copy()
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
print(f"modified `texts`? {not texts == texts_copy}")

print("="*40)
print("SINGLE MESSAGE")
text = "**********"=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages1)
text_copy = text
inputs = processor(
    text=text,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
print(f"modified `text`? {not text == text_copy}")rn_tensors="pt",
)
print(f"modified `text`? {not text == text_copy}")