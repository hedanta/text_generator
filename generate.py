
import argparse
import numpy as np
import pickle

from train import Model

class Generator(object):
    def __init__(self, filename):
        self.model = self.load(filename)
        
    def load(self, filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model    
    
    def generate(self, length, prefix=None):
        first_word = None
        if prefix is not None:
            first_word = Model.preprocess_text(prefix).split()[-1]
        text_arr = [prefix]
        text_arr += self.model.generate(length, first_word)[1:]
        return ' '.join(text_arr)
    
    def load_model(self, filename):
        self.model = load(filename)


# python3 generate.py --model model.pkl --prefix "я" --length 10
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text')
    parser.add_argument('--model', type=str, help='Path to saved model')
    parser.add_argument('--prefix', type=str, default=None, help='Beginning of generated text')
    parser.add_argument('--length', type=int, help='Length of generated text')
    args = parser.parse_args()
    
    gena = Generator(args.model)
    text = gena.generate(prefix=args.prefix, length=args.length)
    print(text)
