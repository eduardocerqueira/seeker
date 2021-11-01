#date: 2021-11-01T17:05:12Z
#url: https://api.github.com/gists/48481c6c5bf78ce39856d8518542ee7b
#owner: https://api.github.com/users/reidtc82

import os
import sys
import csv
#import numpy as np
import random

class DataHandler:
	def __init__(self):
		pass
	
	def get_list(self):
		return []

class MarkovChain:
	def __init__(self, pad_length):
		pass
	
	def fit(self, data):
		pass
	
	def transform(self):
		return 'Hello World!'
	
if __name__ == "__main__":
	dh = DataHandler()
	mc = MarkovChain(5)
	the_list = dh.get_list()

	mc.fit(the_list)
	print(mc.transform())