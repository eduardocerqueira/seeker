#date: 2025-03-26T16:57:21Z
#url: https://api.github.com/gists/678ea2ce4b4c9532e4beb5cd28304fa2
#owner: https://api.github.com/users/dongluw

import base64
import logging
import os
import subprocess

from multimodal_tokeniser.continuous import Tokeniser
from datatools import __version__
from PIL import Image

from vllm import LLM, SamplingParams, TokensPrompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# tokenizer_path = f"gs: "**********"
tokenizer_path = "gs: "**********"

def image_to_base64(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, 'rb') as image_file:
        byte_string = image_file.read()
        base64_encoded = base64.b64encode(byte_string)
        return f"data:image/jpeg;base64,{base64_encoded.decode('utf-8')}"

def get_pil_image(image_path):
    with open(image_path, 'rb') as image_file:
        image = Image.open(image_file)
        # Make a copy before the file closes
        image_copy = image.copy()
    return image_copy

def resize_image(image_path):
    from multimodal_tokeniser.continuous import ImageEncoder
    encoder = ImageEncoder(
                min_image_size=512,
                max_image_size=512,
                downsampling_ratio=16,
                max_crops=12,
            )
    img = get_pil_image(image_path)
    print(img.height, img.width)
    imgs = encoder.encode(img)
    res = {key: [im[key] for im in imgs] for key in imgs[0]}
    return res['image'], res['size'], res['original_size']

def model_path_setup():
    """Fixture to ensure model directory exists and download if needed."""
    model_dir = os.path.abspath("../dx31ultp")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        logger.info("Created directory: %s", model_dir)

        gcs_path = "gs://cohere-dev-central-2/saurabh_data/vision/hf_exports/dx31ultp/*"
        try:
            logger.info("Downloading model from %s to %s...", gcs_path, model_dir)
            subprocess.check_call(["gsutil", "-m", "cp", "-r", gcs_path, model_dir],
                                timeout=600)
            logger.info("Download completed successfully.")
        except Exception as e:
            logger.error("Failed to download model: %s", e)
            pytest.fail(f"Failed to download model: {e}")
    
    return model_dir

 "**********"d "**********"e "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********"( "**********") "**********": "**********"
    """Fixture to initialize the tokenizer."""
    return TemplatedBPTokenizer(
        tokenizer_path,
        chat_template_name= "**********"
        chat_preamble="",
        img_size=364,
        img_patch_size=14*2,
        max_splits_per_img=12,
        image_text_ordering="image_first"
    )

def llm(model_path_setup):
    """Fixture to initialize the LLM."""
    try:
        print(model_path_setup)
        import torch
        print(torch.cuda.is_available()) 
        llm_instance = LLM(model=model_path_setup, max_model_len=8192)
        logger.info("LLM initialized successfully.")
        return llm_instance
    except Exception as e:
        logger.error("Failed to initialize LLM: %s", e)
        pytest.fail(f"Failed to initialize LLM: {e}")

def sampling_params():
    """Fixture to define sampling parameters."""
    return SamplingParams(top_k= "**********"=200)

 "**********"d "**********"e "**********"f "**********"  "**********"e "**********"n "**********"c "**********"o "**********"d "**********"e "**********"_ "**********"w "**********"i "**********"t "**********"h "**********"_ "**********"t "**********"u "**********"r "**********"n "**********"s "**********"( "**********"m "**********"s "**********"g "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********"_ "**********"t "**********"x "**********"t "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********"_ "**********"i "**********"m "**********"g "**********") "**********": "**********"
    token_ids = tokenizer_img.encode(msg)['token_ids'].tolist()[1: "**********"
    encoded_txt = "**********"=False, ignore_oov=False)

    conversation = [
        {"role": "User", "message": [
            {"text": encoded_txt},
        ]}
    ]
    return tokenizer_txt.encode_turns(conversation)
    

 "**********"d "**********"e "**********"f "**********"  "**********"t "**********"e "**********"s "**********"t "**********"_ "**********"m "**********"o "**********"d "**********"e "**********"l "**********"_ "**********"o "**********"u "**********"t "**********"p "**********"u "**********"t "**********"s "**********"( "**********"l "**********"l "**********"m "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********", "**********"  "**********"s "**********"a "**********"m "**********"p "**********"l "**********"i "**********"n "**********"g "**********"_ "**********"p "**********"a "**********"r "**********"a "**********"m "**********"s "**********") "**********": "**********"
    """Test model outputs using tokenizer approach."""
    
    # conversation1 = [
    #     {"role": "User", "message": [
    #         {"text": "Describe this image"},
    #         {"url": image_to_base64("vllm-cohere/tests/test_images/image_resized.png")}
    #     ]}
    # ]

    # conversation2 = [
    #     {"role": "User", "message": [
    #         {"url": image_to_base64("vllm-cohere/tests/test_images/v1_93.jpg")},
    #         {"text": "Print the exact text of this image"}
    #     ]}
    # ]

    # conversation3 = [
    #     {"role": "User", "message": [
    #         {"text": "How many moons does Mars have?"}
    #     ]}
    # ]

    # # Tokenize conversations
    # token_ids0, _ = "**********"=True)
    # token_ids1, _ = "**********"=True)
    # token_ids2, _ = "**********"=True)
    
    msg_1 = [
        {"text": "Describe this image"},
        {"image_sizes": resize_image("vllm-cohere/tests/test_images/image_resized.png")[1]}
    ]
    # Tokenize conversations
    # token_ids_1 = "**********"
    token_ids_1 = "**********"
    # Prepare engine inputs
    engine_input_1 = "**********"
        prompt_token_ids= "**********"
        multi_modal_data={"image": [image_to_base64("vllm-cohere/tests/test_images/image_resized.png")]}
    )
    
    msg_2 = [
        {"image_sizes": resize_image("vllm-cohere/tests/test_images/v1_93.jpg")[1]},  #720 960
        {"text": "The text shown in this image is:"}
    ]

    # Tokenize conversations
    # token_ids_2 = "**********"
    token_ids_2 = "**********"
    # Prepare engine inputs
    engine_input_2 = "**********"
        prompt_token_ids= "**********"
        multi_modal_data={"image": [image_to_base64("vllm-cohere/tests/test_images/v1_93.jpg")]}
    )
    
    msg_3 = [
        {"image_sizes": resize_image("vllm-cohere/tests/test_images/v1_93_resized.jpg")[1]},  #720 960
        {"text": "The text shown in this image is:"}
    ]

    # Tokenize conversations
    # token_ids_3 = "**********"
    token_ids_3 = "**********"
    # Prepare engine inputs
    engine_input_3 = "**********"
        prompt_token_ids= "**********"
        multi_modal_data={"image": [image_to_base64("vllm-cohere/tests/test_images/v1_93_resized.jpg")]}
    )

    # msg_3 = [{"text": "How many moons does Mars have?"}]
    # token_ids_3 = "**********"
    # engine_input_3 = "**********"=token_ids_3)

    # Generate outputs
    outputs = llm.generate(
        [engine_input_1, engine_input_2, engine_input_3],
        sampling_params=sampling_params,
    )

    # Expected outputs
    expected_outputs = [
        ("<|START_RESPONSE|>The image features two adorable golden retriever puppies "
         "sitting side by side on a vibrant green lawn. Their fur is a warm golden shade, "
         "and both have their mouths open, revealing their pink tongues in a joyful expression. "
         "The grass beneath them is dotted with colorful wildflowers, creating a picturesque "
         "and cheerful scene. The puppies appear to be looking at something off-camera, their "
         "eyes focused and curious. The overall atmosphere of the image is playful and serene, "
         "capturing the innocence and charm of young dogs in a natural setting.<|END_RESPONSE|>"),

        ("<|START_RESPONSE|>Connecticut Law of 1642\nIf any man or woman be a witch that is, hath or consulteth with a familiar spirit: "
         "they shall be put to death.<|END_RESPONSE|>"),

        ("<|START_RESPONSE|>Mars has two moons, Phobos and Deimos. These moons are small and irregularly shaped, with Phobos being the "
         "larger of the two. They are believed to be captured asteroids that were pulled into orbit by Mars' gravitational pull.<|END_RESPONSE|>")
    ]

    # Verify outputs
    for idx, (output, expected) in enumerate(zip(outputs, expected_outputs)):
        generated_text = output.outputs[0].text
        print(f"Output {idx}: {generated_text}")
        print(f"Expected {idx}: {expected}")
        # assert generated_text == expected, f"Output {idx} does not match expected output"    
    
    
if __name__ == "__main__":
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import torch

    model_id = "CohereForAI/aya-vision-8b"
    
    from datatools.tokenizer.bpe import TemplatedBPTokenizer
    tokenizer_txt = "**********"
        tokenizer_path,
        chat_template_name= "**********"
        chat_preamble="",
        img_size=364,
        img_patch_size=14*2,
        max_splits_per_img=12,
        image_text_ordering="image_first"
    )

    tokenizer = "**********"=512, max_image_size=512, downsample_ratio=16, max_crops=12)
    msg_1 = [
        {"image_sizes": resize_image("vllm-cohere/tests/test_images/v1_93.jpg")[1]},
        {"text": "Print the exact text of this image"},
    ]
    token_ids = tokenizer.encode(msg_1)['token_ids'].tolist()[1: "**********"

    token_ids_turn = "**********"
    breakpoint()
    
    processor = AutoProcessor.from_pretrained(model_id)
    # llm_instance = LLM(model="ckpts/dx31ultp", max_model_len=8192, enforce_eager=True)
    llm_instance = LLM(model="ckpts/mm_with_dummy_siglip", max_model_len=8192, tensor_parallel_size=4, enforce_eager=True, gpu_memory_utilization=0.99, quantization="fp8")
    logger.info("LLM initialized successfully.")
    sampling_params = "**********"=1, max_tokens=128)
    test_model_outputs(llm_instance, tokenizer, sampling_params)