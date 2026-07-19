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







