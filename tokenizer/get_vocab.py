import nltk
nltk.download('words')
from nltk.corpus import words

# Load the complete list
english_vocab = " ".join(words.words())

with open('./input.txt', 'w', encoding = 'utf-8') as f:
    f.write(english_vocab)
    print('done')

