#date: 2023-06-30T16:52:06Z
#url: https://api.github.com/gists/cab7915d4719bdb0b695c98e800bbf23
#owner: https://api.github.com/users/JonathanFly

### Implementation of Prompt Alternating for LLMs


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

 "**********"d "**********"e "**********"f "**********"  "**********"p "**********"r "**********"o "**********"m "**********"p "**********"t "**********"_ "**********"a "**********"l "**********"t "**********"e "**********"r "**********"n "**********"a "**********"t "**********"i "**********"n "**********"g "**********"( "**********"p "**********"r "**********"o "**********"m "**********"p "**********"t "**********", "**********"  "**********"i "**********"n "**********"s "**********"e "**********"r "**********"t "**********"_ "**********"p "**********"o "**********"s "**********"i "**********"t "**********"i "**********"o "**********"n "**********", "**********"  "**********"a "**********"l "**********"t "**********"e "**********"r "**********"n "**********"a "**********"t "**********"e "**********"_ "**********"p "**********"r "**********"o "**********"m "**********"p "**********"t "**********"s "**********", "**********"  "**********"n "**********"u "**********"m "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********": "**********"
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = "**********"

    prompt_tokens = "**********"="pt")
    output_tokens = "**********"
    
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"_ "**********"  "**********"i "**********"n "**********"  "**********"r "**********"a "**********"n "**********"g "**********"e "**********"( "**********"n "**********"u "**********"m "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********": "**********"
        alternate_index = "**********"
        alternate = alternate_prompts[alternate_index]
        
        alternate_tokens = "**********"="pt")

        print(prompt_tokens[: "**********":insert_position])
        print(alternate_tokens)
        print(output_tokens[: "**********":])
        
        input_ids = torch.cat((prompt_tokens[: "**********":insert_position] "**********"alternate_tokens "**********"output_tokens[: "**********"insert_position:]) "**********"dim=-1)
        next_token = model.generate(input_ids, max_length=input_ids.shape[1] + 1, do_sample = True)[: "**********"
        
        output_tokens = "**********"=-1)
        
    generated_text = "**********"=True)
    return generated_text

prompt = "This is a apple"
insert_position = 3
alternate_prompts = ["blue", "red", "yellow"]
num_tokens = "**********"

result = "**********"
print(result)
