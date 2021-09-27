# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 10:49:01 2021

@author: Gillies
"""

import pandas as pd
import numpy as np
import re
import os

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Input, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import SparseCategoricalCrossentropy

URL_web = "C://Users//Gillies//Desktop//rvm_googleCloud//rvcsv.csv"
lines = pd.read_csv(URL_web,index_col=0, engine='python')

def clean_text(sentence):
  sentence = sentence.lower()
  sentence = re.sub(r'\[.*?\]', "", sentence) 
  sentence = re.sub(r"\u2005", "", sentence)

  sentence = re.sub(r"’", "\'", sentence) 
  sentence = re.sub(r"‘", "\'", sentence)
  sentence = re.sub(r"i'm", "i am", sentence)
  sentence = re.sub(r"its", "it is", sentence)
  sentence = re.sub(r"he's", "he is", sentence)
  sentence = re.sub(r"she's", "she is", sentence)
  sentence = re.sub(r"it's", "it is", sentence)
  sentence = re.sub(r"that's", "that is", sentence)
  sentence = re.sub(r"what's", "what is", sentence)
  sentence = re.sub(r"where's", "where is", sentence)
  sentence = re.sub(r"there's", "there is", sentence)
  sentence = re.sub(r"who's", "who is", sentence)
  sentence = re.sub(r"how's", "how is", sentence)
  sentence = re.sub(r"\'ll", " will", sentence)
  sentence = re.sub(r"\'ve", " have", sentence)
  sentence = re.sub(r"\'re", " are", sentence)
  sentence = re.sub(r"\'d", " would", sentence)
  sentence = re.sub(r"won't", "will not", sentence)
  sentence = re.sub(r"can't", "cannot", sentence)
  sentence = re.sub(r"n't", " not", sentence)
  sentence = re.sub(r"n'", "ng", sentence)
  sentence = re.sub(r"\'bout", "about", sentence)
  sentence = re.sub(r"'til", "until", sentence)
  sentence = re.sub(r"c'mon", "come on", sentence)
  sentence = re.sub("\n", " ", sentence)

  sentence = re.sub(r"\u2005", "", sentence)
  sentence = re.sub("[-*/()\"’‘'#/@;:<>{}`+=~|.!?,]", "", sentence) 
  sentence = re.sub(r"'", "", sentence)
  sentence = re.sub(r"\t", "", sentence)
  sentence = re.sub(r"  ", " ", sentence)
  sentence = re.sub(r"  ", " ", sentence)
  return sentence

def generate(word, length):
    model = load_model("model.h5")
    word = clean_text(word)
    inputs = np.zeros((1, 1))
    inputs[0, 0] = word2idx[word]
    count = 1
    returnString = word
    while count <= length:
        pred = model.predict(inputs)
        word = np.argmax(pred)
        if word >= vocab_size:
            word = vocab_size - 1

        inputs[0, 0] = word
        
        returnString = returnString  + " " + idx2word[word]
        count += 1
    return returnString

lines.lines = lines.lines.apply(lambda line: clean_text(line))

x_train = [line[:-1] for line in lines.lines]
y_train = [line[1:] for line in lines.lines]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(lines.lines)

x_train = tokenizer.texts_to_sequences(x_train)
y_train = tokenizer.texts_to_sequences(y_train)


word2idx = tokenizer.word_index
idx2word = {value: key for key, value in word2idx.items()}

word2idx["<pad>"] = 0
idx2word[0] = "<pad>"


lengths = []

for sequence in x_train:
    lengths.append(len(sequence))
    
lengths = pd.Series(lengths)

maxlen = max(lengths)
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = int(vocab_size**0.25)

x_train = pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
y_train = pad_sequences(y_train, maxlen=maxlen, padding='post', truncating='post')

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True))
model.add(GRU(units=maxlen, return_sequences=True))
model.add(Dense(vocab_size))

model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True))


history = model.fit(x_train, y_train, epochs=2, verbose=1)

model.save("model.h5")

print(generate("I", 5))


