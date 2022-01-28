#date: 2022-01-28T17:11:56Z
#url: https://api.github.com/gists/fe09baba21997bd112c9c695940ae2e0
#owner: https://api.github.com/users/vkhangpham

label_list = ['O', 'B-POI', 'I-POI', 'B-STR', 'I-STR']
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {v: k for k, v in id2label.items()}