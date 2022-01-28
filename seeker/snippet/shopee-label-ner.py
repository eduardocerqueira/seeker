#date: 2022-01-28T17:02:34Z
#url: https://api.github.com/gists/61c367ed1bef6c4a4f239e0be3fcd2fd
#owner: https://api.github.com/users/vkhangpham

def label_ner(example):
    tokens = example['tokens']
    labels = example['labels']
    found_poi, found_str = False, False
    for idx in range(len(tokens)):
        if tokens[idx] in example['POI']:
            if not found_poi:
                labels[idx] = 'B-POI'
                found_poi = True
            else:
                labels[idx] = 'I-POI'
        if tokens[idx] in example['STR']:
            if not found_str:
                labels[idx] = 'B-STR'
                found_str = True
            else:
                labels[idx] = 'I-STR'
    return {
        'labels': labels,
        'ner_tags': [label2id[label] for label in labels]
    }
 
raw_train = raw_train.map(label_ner)
raw_test = raw_test.map(label_ner)