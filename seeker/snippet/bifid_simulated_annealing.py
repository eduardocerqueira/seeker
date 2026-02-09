#date: 2026-02-09T17:35:00Z
#url: https://api.github.com/gists/491145fafd95e31b08d98f7931398447
#owner: https://api.github.com/users/wxrdnx

#!/usr/bin/env python3

# Inspired from: http://practicalcryptography.com/cryptanalysis/stochastic-searching/cryptanalysis-bifid-cipher/

import random
import math
import ngram_score as ns

TEMP = 20
STEP = 0.1
COUNT = 10000

class ngram_score(object):
    def __init__(self,ngramfile,sep=' '):
        """load a file containing ngrams and counts, calculate log probabilities """
        self.ngrams = {}
        with open(ngramfile, 'r') as file:
            for line in file.readlines():
                key, count = line.split(sep) 
                self.ngrams[key] = int(count)
        self.L = len(key)
        self.N = sum(self.ngrams.values())
        for key in self.ngrams.keys():
            self.ngrams[key] = math.log10(float(self.ngrams[key]) / self.N)
        self.floor = math.log10(0.01 / self.N)

    def score(self,text):
        """compute the score of text"""
        score = 0
        for i in range(len(text) - self.L + 1):
            if text[i:i+self.L] in self.ngrams:
                score += self.ngrams[text[i:i + self.L]]
            else:
                score += self.floor          
        return score

quadgrams = ns.ngram_score('english_quadgrams.txt')

def bifid_decipher(key, period, text):
    """Decipher text using the Bifid cipher with given key and period."""
    result = []
    text_len = len(text)
    i = 0
    
    while i < text_len:
        current_period = min(period, text_len - i)
        
        for j in range(current_period):
            a = text[i + (j // 2)]
            b = text[i + ((current_period + j) // 2)]
            
            a_ind = key.index(a)
            b_ind = key.index(b)
            
            a_row = a_ind // 5
            b_row = b_ind // 5
            a_col = a_ind % 5
            b_col = b_ind % 5
            
            if j % 2 == 0:
                result.append(key[5 * a_row + b_col])
            else:
                result.append(key[5 * a_col + b_row])
        
        i += current_period
    
    return ''.join(result)

def exchange_2_letters(key):
    """Swap two random letters in the key."""
    key_list = list(key)
    i = random.randint(0, 24)
    j = random.randint(0, 24)
    key_list[i], key_list[j] = key_list[j], key_list[i]
    return ''.join(key_list)

def bifid_crack(text, best_key, period):
    """Crack the Bifid cipher using simulated annealing."""
    text_len = len(text)
    max_key = best_key
    
    deciphered = bifid_decipher(max_key, period, text)
    max_score = quadgrams.score(deciphered)
    best_score = max_score
    
    T = TEMP
    while T >= 0:
        for count in range(COUNT):
            test_key = exchange_2_letters(max_key)
            deciphered = bifid_decipher(test_key, period, text)
            score = quadgrams.score(deciphered)
            dF = score - max_score
            
            if dF >= 0:
                max_score = score
                max_key = test_key
            elif T > 0:
                prob = math.exp(dF / T)
                if prob > random.random():
                    max_score = score
                    max_key = test_key
            
            # Keep track of best score we have seen so far
            if max_score > best_score:
                best_score = max_score
                best_key = max_key
        
        T -= STEP
    
    return best_score, best_key

def main():
    cipher = "KWTAZQLAWWZCPONIVBTTBVQUZUGRNHAYIYGIAAYURCUQLDFTYVHTNQEENUPAIFCUNQTNGITEFUSHFDWHRIFSVTBISYDHHASQSROMUEVPQHHCCRBYTQBHWYRRHTEPEKHOBFSZUQBTSYRSQUDCSAOVUUGXOAUYWHPGAYHDNKEZPFKKWRIEHDWPEIOTBKESYETPBPOGTHQSPUMDOVUEQAUPCPFCQHRPHSOPQRSSLPEVWNIQDIOTSQESDHURIEREN"
    period = 7
    
    print("Running bifid crack, this could take a few minutes...")
    
    key = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    iteration = 0
    max_score = -99e99
    
    # Run until user kills it (Ctrl+C)
    try:
        while True:
            iteration += 1
            score, key = bifid_crack(cipher, key, period)
            
            if score > max_score:
                max_score = score
                print(f"Best score so far: {score}, on iteration {iteration}")
                print(f"    Key: '{key}'")
                plaintext = bifid_decipher(key, period, cipher)
                print(f"    Plaintext: '{plaintext}'")
    except KeyboardInterrupt:
        print("\nStopped by user")

if __name__ == "__main__":
    main()