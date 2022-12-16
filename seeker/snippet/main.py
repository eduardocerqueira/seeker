#date: 2022-12-16T16:59:53Z
#url: https://api.github.com/gists/9e6a771165220b42665b66df88e4a374
#owner: https://api.github.com/users/McCdama

from transformers import pipeline
import codecs
import re


analyzer = pipeline(
    task='text-classification',
    model="cmarkea/distilcamembert-base-sentiment",
    tokenizer= "**********"
)

output = open('output.txt', 'w')

with codecs.open(r'myfile.txt', encoding='latin1') as f:
    for line in f.readlines():
        result = analyzer(
                re.split('\.+\s', line),
                top_k=1
        )
        out = line,"-->",result 
        print(out, file=output)    print(out, file=output)