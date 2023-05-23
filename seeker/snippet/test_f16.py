#date: 2023-05-23T16:46:15Z
#url: https://api.github.com/gists/2959ff31db3346482df1fe58ae8a5c39
#owner: https://api.github.com/users/pashu123

import torch
import torch_mlir
from shark.shark_importer import import_with_fx
import os
import sys
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)



def compile_via_shark(model, inputs):
    is_f16 = True 
    input_mask = [False, False]
    bytecode = import_with_fx(model, inputs, is_f16=True, f16_input_mask=input_mask)
    with open(os.path.join("vicuna_fp16.mlir"), "wb") as mlir_file:
        mlir_file.write(bytecode[0])


    from shark.shark_inference import SharkInference
    shark_module = SharkInference(
        mlir_module=bytecode[0], device="cuda", mlir_dialect="tm_tensor",
    )
    # extra_args = ['--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-flow-detach-elementwise-from-named-ops,iree-flow-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=32}))', '--iree-spirv-index-bits=64']
    shark_module.compile(extra_args=[])
    return shark_module


tokenizer = "**********"
    "TheBloke/vicuna-7B-1.1-HF",
    use_fast=False
)

 "**********"c "**********"l "**********"a "**********"s "**********"s "**********"  "**********"S "**********"t "**********"o "**********"p "**********"O "**********"n "**********"T "**********"o "**********"k "**********"e "**********"n "**********"s "**********"( "**********"S "**********"t "**********"o "**********"p "**********"p "**********"i "**********"n "**********"g "**********"C "**********"r "**********"i "**********"t "**********"e "**********"r "**********"i "**********"a "**********") "**********": "**********"
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

prompt = f"{system_prompt}<|USER|>What's your mood today?<|ASSISTANT|>"

inputs = "**********"="pt")

inputs_model = (inputs["input_ids"], inputs["attention_mask"])

class SLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            "TheBloke/vicuna-7B-1.1-HF"
        )

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)[0]


slm_model = SLM()

# res_pytorch = slm_model(inputs_model[0], inputs_model[1])
shark_unet = compile_via_shark(slm_model, inputs_model)

