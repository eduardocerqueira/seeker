#date: 2022-12-23T17:09:03Z
#url: https://api.github.com/gists/c0906cc0f8bc5187ec126d926e5d1870
#owner: https://api.github.com/users/hosteren

import re
class Lixer:
	def __init__(self) -> None:
		# Your one stop shop in changable variables! Come on in, come on down. We got regex and some other shit. Please help me.
		self.punctuation_regex = r"[.:!?]"
		self.clean_regex = r"[\.\,\:\!\?\-\\\/\(\)\[\]\;\*\'\"\#\$\@\+0-9\_\n\t]"
		self.capitalised_word_regex = r"([A-ZÆØÅ][a-zæøå]+)"
		self.complex_threshold = 6

	def _count_capitalised_words(self, text: str) -> int:
		# The solution to "How many words starts with a capital letter in this paticular string" problem. 
		capitalised_words = re.findall(self.capitalised_word_regex, text)
		return len(capitalised_words)

	def _count_complex_words(self, text: str) -> int:
		# You would think that counting words longer than a certain threshold would be easy..
		# You would think that punctuation and special characters didn't matter..
		# You would be living in ignorance and be a happier person.. LOOK at how long the self.clean_regex is! I'm crying right now.
		text = re.sub(self.clean_regex, "", text)
		complex_words = [word for word in text.split() if len(word) > self.complex_threshold]
		return len(complex_words)

	def _count_periods(self, text: str) -> int:
		# Counting actual periods was ez mode though.
		periods = re.findall(self.punctuation_regex, text)
		return len(periods)

	def calc_lix(self, text: str) -> float:
		# Magic! Stripping whitespace and doin' the deed. A result of 20 is considered light reading, while 60 would be Homers Illiad
		text = text.strip()
		num_words = len(text.split())
		num_capitalised_words = self._count_capitalised_words(text)
		num_periods = self._count_periods(text)
		num_complex_words = self._count_complex_words(text)
		#print(f"The calculation looked like this:\n{num_words} / ({num_periods} + {num_capitalised_words}) + {num_complex_words} * 100 / {num_words}")
		lix_score = num_words/(num_periods+num_capitalised_words) + num_complex_words * 100 / num_words
		return f"{lix_score:.2f}"