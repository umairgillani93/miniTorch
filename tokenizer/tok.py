import os 
import sys
import re

class Tokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab # vocab already haivng structure like: {'camel': 19}
        self.int_to_str = {v:k for k, v in vocab.items()} # reverts the keys and values of vocab

    def encoder(self, text):
        '''
        Encodes the text to corresponding 
        token ids.
        '''
        text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        text = [x.strip() for x in text.split() if x.strip()]
        text = [x if x in self.str_to_int else "<|unk|>"
                for x in text]
        
        ids = [self.str_to_int[x] for x in text]

        return ids

    def decode(self, ids):
        '''
        we need to pass token ids to `decode`
        and this should return us corresponding 
        text tokens.
        '''
        text = ' '.join(self.int_to_str[x] for x in ids)
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text


