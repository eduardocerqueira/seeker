#date: 2022-01-28T17:11:17Z
#url: https://api.github.com/gists/2b60dad8390f0afbd54270353c3dd633
#owner: https://api.github.com/users/vkhangpham

from transformers import AutoTokenizer, AutoModelForTokenClassification

checkpoint = 'cahya/xlm-roberta-base-indonesian-NER'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)