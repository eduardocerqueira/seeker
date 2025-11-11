#date: 2025-11-11T17:04:42Z
#url: https://api.github.com/gists/0929ec3f3392d933fd64d5b194b2702b
#owner: https://api.github.com/users/jcaip

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig, Qwen3VLMoeForConditionalGeneration, AutoProcessor

from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    PerRow,
    FqnToConfig,
    quantize_
)

# Configure logging to see warnings and debug information
logging.basicConfig(
    level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s"
)

# Enable specific loggers that might contain the serialization warnings
logging.getLogger("transformers").setLevel(logging.INFO)
logging.getLogger("torchao").setLevel(logging.INFO)
logging.getLogger("safetensors").setLevel(logging.INFO)
logging.getLogger("huggingface_hub").setLevel(logging.INFO)

config = Float8DynamicActivationFloat8WeightConfig(
    granularity=PerRow(),
)
expert_config = Float8DynamicActivationFloat8WeightConfig(
    granularity=[PerRow(), PerRow(1)],
)

# only quantize language model
quant_config = FqnToConfig({
    r"re:language_model.*.gate_up_proj": expert_config,
    r"re:language_model.*.down_proj": expert_config,
    r"re:language_model.*.q_proj": config,
    r"re:language_model.*.k_proj": config,
    r"re:language_model.*.v_proj": config,
})
quantization_config = TorchAoConfig(quant_type=quant_config)
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    dtype="auto",
    device_map="auto",
    quantization_config=quantization_config
)

for name, p in model.named_parameters():
    print(name, type(p))

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

messages = [
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

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize= "**********"
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

# Inference: Generation of the output
generated_ids = "**********"=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens= "**********"=False
)
print(output_text)zation_spaces=False
)
print(output_text)