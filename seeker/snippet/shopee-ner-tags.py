#date: 2022-01-28T17:08:12Z
#url: https://api.github.com/gists/c40813a16255f9bbe0e22ef69741e91a
#owner: https://api.github.com/users/vkhangpham

words = raw_train[0]["tokens"]
labels = raw_train[0]["ner_tags"]
line1 = ""
line2 = ""
for word, label in zip(words, labels):
    full_label = label_list[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)