#date: 2024-05-07T17:00:36Z
#url: https://api.github.com/gists/0e357aecfe51bbd53a4c41457c29d484
#owner: https://api.github.com/users/brandon-lockaby

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"

# I want a base model and this is instruct-tuned, but it will fit on my gpu
model_path = "microsoft/Phi-3-mini-128k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=DEV,
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = "**********"





class MultiClassifier():
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"_ "**********"i "**********"n "**********"i "**********"t "**********"_ "**********"_ "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"d "**********"e "**********"v "**********", "**********"  "**********"m "**********"o "**********"d "**********"e "**********"l "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********", "**********"  "**********"p "**********"r "**********"o "**********"m "**********"p "**********"t "**********", "**********"  "**********"c "**********"l "**********"a "**********"s "**********"s "**********"_ "**********"n "**********"a "**********"m "**********"e "**********"s "**********") "**********": "**********"
        self.__dict__.update(locals())
        input_ids = "**********"="pt").to(self.dev)
        self.kv_cache = self.model(input_ids, return_dict=True).past_key_values
        self.yes = " yes"
        self.no = " no"
        self.yes_id = "**********"
        self.no_id = "**********"

    def classify(self, held_out_example, return_probs=False):
        output_class_list = []
        output_probs = {}
        prompt = f"{self.prompt}{held_out_example}"
        kv_cache = self.kv_cache
        for class_name in self.class_names:
            prompt += f"\n{class_name}:"
            # todo: "**********"
            input_ids = "**********"="pt").to(self.dev)
            outputs = self.model(input_ids, past_key_values=kv_cache, return_dict=True)
            kv_cache = outputs.past_key_values
            probs = torch.nn.functional.softmax(outputs.logits[-1, -1], dim=-1)
            yes_prob = probs[self.yes_id].item()
            no_prob = probs[self.no_id].item()
            if yes_prob >= no_prob:
                prompt += self.yes
                output_class_list.append(class_name)
            else:
                prompt += self.no
            output_probs[class_name] = {"yes": yes_prob, "no": no_prob}
        return (output_class_list, output_probs) if return_probs else output_class_list


prompt = """Text: An apple a day keeps the doctor away.
Apples: yes
Oranges: no

Text: I am learning to tie my shoe.
Apples: no
Oranges: no

Text: I ate an apple and then a few oranges.
Apples: yes
Oranges: yes

Text: Do you sell chocolate oranges?
Apples: no
Oranges: yes

Text: """

class_names = ["Apples", "Oranges"]

classifier = "**********"

print(classifier.classify("Cigarettes are simply the best.", return_probs=True))

