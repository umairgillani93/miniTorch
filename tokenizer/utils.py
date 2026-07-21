import os 
import sys 
import re 
import nltk

class Utility:
    def __init__(self):
        self.vocab_data = './input.txt'

    def __repr__(self):
        return f'Utility class for helper functoins'

    def create_vocab(self):
        with open(self.vocab_data, 'r') as f:
            data = f.read().split()
            all_words = set(data)

        return all_words

    def pad_sequence(self, sequences):
        '''post pad the list of sequences to zeros'''
        S = []
        max_len = 0
        for s in sequences:
            max_len = max(max_len, len(s.split()))

        for s in sequences:
            words = s.split()
            if (len(words) < max_len):
                N = max_len - len(words)
                words.extend(["0"] * N)
                S.append(words)
            else:
                S.append(words)

        return S


if __name__ == '__main__':
    u = Utility()
    S = ['this is a first len',
        'this is second',
        'abc'
        ]
    print(u.pad_sequence(S))
