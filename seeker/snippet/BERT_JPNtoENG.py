#date: 2023-02-16T16:43:48Z
#url: https://api.github.com/gists/fc4b6bce731df7f9fe45cd9b72ba04d7
#owner: https://api.github.com/users/Absolute-Value

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# GPUが使える場合はGPUを使用
　if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'
print(f'device : {device}')

# tokenizerとモデルを読み込む
tokenizer = "**********"
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
model.to(device)

# 翻訳をする関数を定義
def translation(japanese_text = "こんにちは、私の名前は太郎です。"):
  # 日本語文章をトークナイズする
  input_ids = "**********"="pt")

  # BERTを使用して日本語から英語に翻訳する
  output = model.generate(input_ids.to(device))

  # 翻訳された英語文章をデコードする
  english_text = "**********"=True)
  print(english_text)
  
translation(input('英語に翻訳したい日本語を入力してください：')=True)
  print(english_text)
  
translation(input('英語に翻訳したい日本語を入力してください：')