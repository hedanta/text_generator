import argparse
import pickle
import numpy as np
import os
import string

class Model(object):
    def __init__(self):
        self.words = []
        self.probs = {}

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def preprocess_text(text):
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        return text

    def fit(self, text):
        text = Model.preprocess_text(text)
        arr = text.split()
        
        for word in arr:
            self.words.append(word)
        
        # префикс - одно слово
        for i in range(0, len(arr) - 1):
            prefix = arr[i]
            next_word = arr[i + 1]
            if prefix in self.probs.keys():
                if next_word in self.probs[prefix]:
                    self.probs[prefix][next_word] += 1 
                else:
                    self.probs[prefix][next_word] = 1
            else:
                self.probs[prefix] = {next_word: 1}
        
        # префикс - два слова
        for i in range(0, len(arr) - 2):
            word1 = arr[i]
            word2 = arr[i + 1]
            next_word = arr[i + 2]
            prefix = (word1, word2)
            if prefix in self.probs.keys():
                if next_word in self.probs[prefix]:
                    self.probs[prefix][next_word] += 1 
                else:
                    self.probs[prefix][next_word] = 1
            else:
                self.probs[prefix] = {next_word: 1}
        
        # считаем вероятности
        for prefix in self.probs:
            amount = sum(self.probs[prefix].values())
            for next_word in self.probs[prefix]:
                self.probs[prefix][next_word] /= amount

    def generate_next_word(self, prefix):
        next_words = list(self.probs[prefix].keys())
        next_probs = list(self.probs[prefix].values())
        return np.random.choice(a=next_words, p=next_probs)
    
    def generate(self, length, prefix=None):
        if prefix is None:
            prefix = np.random.choice(self.words)
        result = [prefix]
        for i in range(length):
            result.append(self.generate_next_word(prefix))
            prefix = result[-1]
        return result


class Trainer(object):
    def __init__(self):
        self.model = Model()
    
    def load_text(self, filename):
        text = ''
        with open(filename, 'r') as f:
            for line in f:
                text += line
        return text
    
    def train_model(self, dirname, filename):
        text = self.load_text(dirname + '/' + filename)
        self.model.fit(text)
    
    def save_model(self, filename):
        self.model.save(filename)        
        

# python3 train.py --input-dir data --model model.pkl
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--input-dir', type=str, help='Path to dir')
    parser.add_argument('--model', type=str, help='Path to saved model')
    args = parser.parse_args()

    trainer = Trainer()

    for textfile in os.listdir(args.input_dir):
        print('Training model on file: ' + textfile)
        trainer.train_model(dirname=args.input_dir, filename=textfile)

    trainer.save_model(filename=args.model)
    print('Model trained successfully')
