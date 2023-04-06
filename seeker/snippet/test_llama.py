#date: 2023-04-06T16:53:02Z
#url: https://api.github.com/gists/c7472f5087ed38ee73043ba5f4103b27
#owner: https://api.github.com/users/shoaibahmed

from transformers import LlamaForCausalLM, LlamaTokenizer


 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"n "**********"e "**********"r "**********"a "**********"t "**********"e "**********"_ "**********"c "**********"o "**********"m "**********"p "**********"l "**********"e "**********"t "**********"i "**********"o "**********"n "**********"s "**********"( "**********"m "**********"o "**********"d "**********"e "**********"l "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********", "**********"  "**********"p "**********"r "**********"o "**********"m "**********"p "**********"t "**********", "**********"  "**********"m "**********"a "**********"x "**********"_ "**********"n "**********"e "**********"w "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"= "**********"1 "**********"0 "**********"0 "**********") "**********": "**********"
    model.eval()
    inputs = "**********"='pt')
    outputs = "**********"=10, do_sample=True, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    decoded_text = "**********"
    print("Decoded text:", decoded_text)


num_params = "7B"
assert num_params in ["7B", "13B", "33B", "65B"], num_params

root_dir = f"./hf_checkpoints/{num_params}/"
tokenizer = "**********"
model = LlamaForCausalLM.from_pretrained(root_dir)

examples = [
    "I can't believe that you are such a",
    "The most important part of the world is",
    "In summer 1990,",
    "The most important event in the history of our universe is",
]

for prompt in examples:
    generate_completions(model, tokenizer, prompt)
