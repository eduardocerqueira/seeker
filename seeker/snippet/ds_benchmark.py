#date: 2023-06-22T17:09:36Z
#url: https://api.github.com/gists/d4c5e7e834822bba824e3017760426c0
#owner: https://api.github.com/users/jsheng-jian

import os
import time
import argparse
import deepspeed
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTNeoModel,
)

MODEL_CLASSES = {
    "gpt2": "**********"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=20, help="length of generated text")
    parser.add_argument("--prompt_length", type=int, default=20)
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--print_output", action="store_true", help="Print the generated text")

    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--p", type=float, default=0)

    args = parser.parse_args()

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = "**********"
    tokenizer = "**********"
    model = model_class.from_pretrained(args.model_name_or_path)
    model.eval()

    # deepspeed
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    model = deepspeed.init_inference(model,
                                     mp_size=world_size,
                                     dtype=torch.half,
                                     replace_with_kernel_inject=True)
    model = model.module

    fname = open('inputs.txt', "r")
    prompt_text = fname.readline()
    eprompt = "**********"=args.prompt_length, truncation=True, return_tensors="pt")
    if args.batch_size > 1:
        eprompt = eprompt.repeat(args.batch_size, 1)
    # print(f'######## Promp.shape: {eprompt.shape}')
    input_ids = eprompt.to(model.device)
    attention_mask = torch.ones_like(input_ids).to(model.device)

    def gpt_generate_fn():
        output_sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id= "**********"
            min_new_tokens= "**********"
            max_new_tokens= "**********"
        )
        return output_sequences

    # Measure inference time.
    iterations = 10
    for _ in range(iterations):
        gpt_generate_fn()
    t0 = time.time()
    for _ in range(iterations):
        output_sequences = gpt_generate_fn()
    time_elapsed = time.time() - t0
    print(f'[INFO] GPT time costs: {time_elapsed * 1000 / iterations:.2f} ms')

    if local_rank == 0:
        if args.print_output:
            print(f'[Context]\n {tokenizer.decode(eprompt[0])}')
            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                print(f'\n[Output]{generated_sequence_idx}\n')
                # Decode text
                text = "**********"=True)
                print(text)

if __name__ == "__main__":
    main()ecode text
                text = "**********"=True)
                print(text)

if __name__ == "__main__":
    main()   print(text)

if __name__ == "__main__":
    main()