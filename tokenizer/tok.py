import os 
import sys
import re
from utils import Utility

class Tokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab # vocab already haivng structure like: {'camel': 19}
        self.int_to_str = {v:k for k, v in vocab.items()} # reverts the keys and values of vocab

    def encode(self, text):
        '''
        Encodes the text to corresponding 
        token ids.
        '''
        text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        text = [x.strip() for x in text if x.strip()]
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



if __name__ == "__main__":
    text = """
    this is  a ~ tilda and this is a sign of exclamation !
    """
    u = Utility()
    vocab = u.create_vocab()
    unk_tokens = ["<|unk|>", "<|endoftext|>"]
    all_words = sorted(vocab)
    all_words.extend(unk_tokens)
    vocab = {c:i for i,c in enumerate(all_words)}
    tokenizer = Tokenizer(vocab)
    S = ['this is a first len',
        'this is second',
        'abc'
        ]
    
    # pad the sequences
    padded = u.pad_sequences(S)
    encoded = [tokenizer.encode(p) for p in padded]
    print(encoded)
