#date: 2022-08-01T17:20:19Z
#url: https://api.github.com/gists/5b4c9cf75a0c70f4fcd4938de850782e
#owner: https://api.github.com/users/AfvanMoopen

from random import choice,choices 
from nltk.tokenize import WhitespaceTokenizer
from collections import defaultdict , Counter 


class TextGenerator:
	def __init__(self):
		self.path = input()
		self.tokens = "**********"
		self.ngrams = self.ngrams()

	def read_file(self):
		with open(self.path , encoding="utf-8") as file:
			text = file.read()
		return text 

 "**********"	 "**********"d "**********"e "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"( "**********"s "**********"e "**********"l "**********"f "**********") "**********": "**********"
		text = self.read_file()
		tk = "**********"
		return tk.tokenize(text)

	def ngrams(self):
		ngrams = defaultdict(Counter)
 "**********"	 "**********"	 "**********"f "**********"o "**********"r "**********"  "**********"i "**********"  "**********"i "**********"n "**********"  "**********"r "**********"a "**********"n "**********"g "**********"e "**********"( "**********"l "**********"e "**********"n "**********"( "**********"s "**********"e "**********"l "**********"f "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********"  "**********"- "**********"  "**********"2 "**********") "**********": "**********"
			ngrams[" ".join(self.tokens[i : "**********"
		return ngrams 

	def find_start(self):
		starts = [pair for pair in list(self.ngrams) if pair.split()[0][0].isupper() and pair.split()[0][-1] not in "!?."]
		return choice(starts).split()

	def make_sentence(self , n = 5):
		result = []
		while True:
			if len(result) == 0:
				result = self.find_start()
			w1 = " ".join(result[-2:])
			w2 = choices(list(self.ngrams[w1].keys()) , list(self.ngrams[w1].values()))
			result.append(*w2)

			if w[0][-1] in "!?.":
				if len(result) < n:
					result = []
					continue
				return " ".join(result)
if __name__ = "__main__":
	text_gen = TextGenerator()
	for _ in range(10):
		print(text_gen.make_sentence())

		