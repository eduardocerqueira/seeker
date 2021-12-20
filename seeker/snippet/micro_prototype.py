#date: 2021-12-20T17:18:14Z
#url: https://api.github.com/gists/6304007a098eba52ccdf94415bacf372
#owner: https://api.github.com/users/abhitopia

from collections import Counter

from vowpalwabbit import pyvw
from typing import List

dialogue1 = [
    ('Customer', ["Hello,",
     "I ordered an item.",
     "I don't like it. I want refund.",
     "Best,",
     "David"]),
    ('Agent', ["Hi David,",
     "I am Omkar.",
     "I will be helping you today.",
     "Let me process the refund for you",
     "Kind regards,",
     "Omkar"]),
]

dialogue2 = [
    ('Customer', ["Hello there,",
     "I want refund for my order.",
     "The item came in broken and doesn't work at all.",
     "Best regards,",
     "John"]),
    ('Agent', ["Hi John",
     "My name is Abhi and I will help you today with your query",
     "I will go ahead and process the refund for you",
     "Kind regards,",
     "Abhi"]),
]

dialogue3 = [
    ('Customer', ["Hello there,",
     "I would like to return an item I ordered last weekend because it broken and doesn't work.",
     "Regards,",
     "Bob"]),
    ('Agent', ["Hi Bob",
     "My name in Abhi and I will help you today with your query."
     "I am sorry to hear that.",
     "I will go ahead and process the refund for you",
     "Kind regards,",
     "Abhi"]),
]


class QGram:
    def __init__(self, q=3):
        self.q = q
        self.label_map = {}

    def __call__(self, text: str, label=False) -> List[str]:

        result = Counter()
        for i in range(self.q-1, len(text)):
            feature = text[i - self.q + 1: i+1]
            feature = feature.replace(' ', '$')
            if not feature in self.label_map:
                self.label_map[feature] = len(self.label_map)+1

            if label:
                result[self.label_map[feature]] += 1.0
            else:
                result[feature] += 1.0

        # total = sum(result.values())
        # for k, v in result.items():
        #     result[k] = v/total

        return result


def create_examples(dialogue, qgram):
    so_far = []
    examples = []

    for speaker, response in dialogue:
        for s in response:
            if speaker == 'Agent':
                examples.append((qgram(' '.join(so_far), label=False), qgram(s, label=True)))
            so_far.append(s)

    return examples

qgram = QGram()

train_examples = sum([create_examples(d, qgram) for d in [dialogue1, dialogue2]], [])
test_examples = sum([create_examples(d, qgram) for d in [dialogue3]], [])

num_labels = len(qgram.label_map)


def convert_to_vw_samples(example, namespace='n', test=False):
    inp, label = example

    inp_str = []
    for k, v in inp.items():
        inp_str.append(f"{k}:{v}")

    if test:
        return [f'|{namespace} {" ".join(inp_str)}\n']

    samples = []
    for k, v in label.items():
        if v == 2:
            debug = 1
        for i in range(int(v)):
            sample = f'{k} |{namespace} {" ".join(inp_str)}\n'
            samples.append(sample)

    # label_str = sorted(label_str)
    # final = f'{",".join([str(k) for k in label_str])} |{namespace} {" ".join(inp_str)}\n'
    return samples

train_examples = sum([convert_to_vw_samples(e) for e in train_examples], [])
test_examples = sum([convert_to_vw_samples(e, test=True) for e in test_examples], [])

"|", ":", " "

model = pyvw.vw(arg_str=f"--probabilities --oaa {num_labels} --loss_function=logistic -qnn", quiet=False)

# train_examples = [
#   "0 | price:.23 sqft:.25 age:.05 2006",
#   "1 | price:.18 sqft:.15 age:.35 1976",
#   "0 | price:.53 sqft:.32 age:.87 1924",
# ]

for _ in range(1):
    for example in train_examples:
        model.learn(example)

for t in test_examples:
    prediction = model.predict(t)
    print(prediction)

# prediction = model.predict(test_example)
# print(prediction)