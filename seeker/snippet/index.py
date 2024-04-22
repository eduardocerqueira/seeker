#date: 2024-04-22T17:06:18Z
#url: https://api.github.com/gists/e96ef7a22df8ecb4eff210f56f5470de
#owner: https://api.github.com/users/AnirudhhSharma

from flask import Flask, request, jsonify
import base64
import os
import time
import warnings, logging
from rich import print, inspect, pretty
import urllib.request
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr
from ast import literal_eval
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)

bnb_config = BitsAndBytesConfig(
    llm_int8_enable_fp32_cpu_offload=True,
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# control model memory allocation between devices for low GPU resource (0,cpu)
device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": 0,
    "transformer.h": 0,
    "transformer.ln_f": 0,
    "model.embed_tokens": "**********"
    "model.layers": 0,
    "model.norm": 0,
}
device = "cuda" if torch.cuda.is_available() else "cpu"

# model use for inference
model_id = "mychen76/mistral7b_ocr_to_json_v1"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map=device_map,
)
# tokenizer
tokenizer = "**********"=True)

app = Flask(__name__)


@app.route("/prompt", methods=["POST"])
def prompt():
    image_data = request.json["image"]

    # Decode the base64 image data
    image_data = base64.b64decode(image_data)

    # Generate a unique filename for the image
    filename = "image.jpg"

    # Save the image to disk
    with open(filename, "wb") as f:
        f.write(image_data)

    receipt_image = Image.open(filename)
    receipt_image_array = np.array(receipt_image.convert("RGB"))

    paddleocr = PaddleOCR(
        lang="en", ocr_version="PP-OCRv4", show_log=False, use_gpu=False
    )

    def paddle_scan(paddleocr, img_path_or_nparray):
        result = paddleocr.ocr(img_path_or_nparray, cls=True)
        result = result[0]
        boxes = [line[0] for line in result]  # boundign box
        txts = [line[1][0] for line in result]  # raw text
        scores = [line[1][1] for line in result]  # scores
        return txts, result

    receipt_texts, receipt_boxes = paddle_scan(paddleocr, receipt_image_array)
    prompt = f"""### Instruction:
    You are POS receipt data expert, parse, detect, recognize and convert following receipt OCR image result into structure receipt data object.
    Don't make up value not in the Input. Output must be a well-formed JSON object. Only include place, date and an array containing items with name and price. ```json
    ### Input:
    {receipt_boxes}
    ### Output:
    """
    with torch.inference_mode():
        inputs = "**********"="pt", truncation=True).to(device)
        outputs = model.generate(
            **inputs, max_new_tokens= "**********"
        )  ##use_cache=True, do_sample=True,temperature=0.1, top_p=0.95
        result_text = "**********"
        print(result_text)  # Debugging purposes
        output_json = json.loads(result_text.split("```")[2][4:])
        return jsonify(output_json)

app.run(host='0.0.0.0', port=5000)

# all done
# api calls should now work
# but very slow ~ 10 mins for one prompt# api calls should now work
# but very slow ~ 10 mins for one prompt