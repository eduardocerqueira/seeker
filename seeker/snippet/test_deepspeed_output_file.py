#date: 2022-09-30T17:30:50Z
#url: https://api.github.com/gists/6372f42b0887194cf302d2c0293032c2
#owner: https://api.github.com/users/bstee615

#%%
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from deepspeed.profiling.flops_profiler import get_model_profile


 "**********"d "**********"e "**********"f "**********"  "**********"b "**********"e "**********"r "**********"t "**********"_ "**********"i "**********"n "**********"p "**********"u "**********"t "**********"_ "**********"c "**********"o "**********"n "**********"s "**********"t "**********"r "**********"u "**********"c "**********"t "**********"o "**********"r "**********"( "**********"b "**********"a "**********"t "**********"c "**********"h "**********"_ "**********"s "**********"i "**********"z "**********"e "**********", "**********"  "**********"s "**********"e "**********"q "**********"_ "**********"l "**********"e "**********"n "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********") "**********": "**********"
    fake_seq = ""
    for _ in range(seq_len - 2): "**********"
      fake_seq += "**********"
    inputs = "**********"
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * batch_size)
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs


with torch.cuda.device(0):
    tokenizer = "**********"
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    batch_size = 4
    seq_len = 128
    enable_profile = True
    if enable_profile:
      flops, macs, params = get_model_profile(
          model,
          kwargs= "**********"
          print_profile=True,
          detailed=True,
          output_file="profile.txt",
      )
    else:
      inputs = "**********"
      outputs = model(inputs)
