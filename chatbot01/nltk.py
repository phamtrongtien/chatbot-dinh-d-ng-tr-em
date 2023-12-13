# nltk.py
import string
import numpy as np

class NltkUtils:
    @staticmethod
    def clean_up_sentence(sentence):
        sentence = ''.join([char.lower() for char in sentence if char not in string.punctuation])
        return sentence.split()

    @staticmethod
    def bag_of_words(sentence, words):
        sentence_words = NltkUtils.clean_up_sentence(sentence)
        bag = [1 if word in sentence_words else 0 for word in words]
        return np.array(bag)
