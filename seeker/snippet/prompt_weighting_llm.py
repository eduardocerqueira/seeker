#date: 2023-06-30T16:52:06Z
#url: https://api.github.com/gists/cab7915d4719bdb0b695c98e800bbf23
#owner: https://api.github.com/users/JonathanFly

### Implementing Automatic1111 style attention weights
### Note, GPT2 is very tempermental with this technique, seems to need a high temperature for even close to coherent output

import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

 "**********"d "**********"e "**********"f "**********"  "**********"m "**********"o "**********"d "**********"i "**********"f "**********"y "**********"_ "**********"a "**********"t "**********"t "**********"e "**********"n "**********"t "**********"i "**********"o "**********"n "**********"_ "**********"m "**********"a "**********"s "**********"k "**********"( "**********"p "**********"r "**********"o "**********"m "**********"p "**********"t "**********", "**********"  "**********"m "**********"o "**********"d "**********"e "**********"l "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********") "**********": "**********"
    tokens = "**********"
    attention_modifiers = []
    add_space = False

 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"n "**********"  "**********"r "**********"e "**********". "**********"s "**********"p "**********"l "**********"i "**********"t "**********"( "**********"r "**********"' "**********"\ "**********"( "**********"| "**********"\ "**********") "**********"' "**********", "**********"  "**********"p "**********"r "**********"o "**********"m "**********"p "**********"t "**********") "**********": "**********"
        if ': "**********":
            word, modifier = token.split(': "**********"
            modifier = float(modifier.strip())
        else:
            word = "**********"
            modifier = 1.0

        current_tokens = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"a "**********"d "**********"d "**********"_ "**********"s "**********"p "**********"a "**********"c "**********"e "**********"  "**********"a "**********"n "**********"d "**********"  "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********": "**********"
            tokens.append('Ġ')  # Space token for GPT-2
            attention_modifiers.append(1.0)
        tokens.extend(current_tokens)
        attention_modifiers.extend([modifier] * len(current_tokens))
        add_space = True

    attention_mask = torch.tensor([attention_modifiers])
    input_ids = "**********"

    return input_ids, attention_mask


 "**********"d "**********"e "**********"f "**********"  "**********"c "**********"u "**********"s "**********"t "**********"o "**********"m "**********"_ "**********"g "**********"e "**********"n "**********"e "**********"r "**********"a "**********"t "**********"e "**********"( "**********"p "**********"r "**********"o "**********"m "**********"p "**********"t "**********", "**********"  "**********"m "**********"o "**********"d "**********"e "**********"l "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********", "**********"  "**********"* "**********"* "**********"k "**********"w "**********"a "**********"r "**********"g "**********"s "**********") "**********": "**********"
    input_ids, attention_mask = "**********"
    print(attention_mask)

    # Set the modified attention mask
    model.config.attention_probs_dropout_prob = 0.0

    with torch.no_grad():
        output_sequences = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    return tokenizer.decode(output_sequences[0], skip_special_tokens= "**********"

model_name = "gpt2"
tokenizer = "**********"
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = "The (large house:1.0001) was situated on a hill. The buildings were made in an enormous block by the three towers of the four houses, with high ceilings of over one hundred and eight inches. They were built with stones and wood and all are from small scale timber."

generated_text = "**********"= True, temperature = 20.0, max_length=200)
print(generated_text)
