# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 16:39:28 2019


"""
# Preprocessing labels of raw IMDB data

import os

imdb_dir = r'C:\Users\jbrowning\Dropbox\ML\Deep_Learning_With_Python\aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
                
# Tokenizing the text of the raw IMBD data
                
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

maxlen = 100  # cut off review after 100 words
training_samples = 200 # train on 20 samples
validation_samples = 10000   #validate on 10000 samples
max_words = 10000 #look at top 10000 words in dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts+to_sequences(texts)

