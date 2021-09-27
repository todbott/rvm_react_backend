from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import numpy as np
import pandas as pd

from google.cloud import storage



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

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))


download_blob('rvm_models', 'rvcsv.csv', '/tmp/rvcsv.csv')
lines = pd.read_csv('/tmp/rvcsv.csv',index_col=0)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines.lines)
word2idx = tokenizer.word_index
idx2word = {value: key for key, value in word2idx.items()}

vocab_size = len(tokenizer.word_index) + 1

word2idx["<pad>"] = 0
idx2word[0] = "<pad>"

def generate(request):

    request_json = request.get_json()
    if request.args and 'word' in request.args:
        word = request.args.get('word')
        length = request.args.get('length')
    elif request_json and 'word' in request_json:
        word = request_json['word']
        length = request_json['length']

    download_blob('rvm_models', 'model.h5', '/tmp/model.h5')
    model = load_model("/tmp/model.h5")
    word = clean_text(word)
    inputs = np.zeros((1, 1))
    inputs[0, 0] = word2idx[word]
    count = 1
    returnString = word
    while count <= int(length):
        pred = model.predict(inputs)
        word = np.argmax(pred)
        if word >= vocab_size:
            word = vocab_size - 1

        inputs[0, 0] = word
        
        returnString = returnString  + " " + idx2word[word]
        count += 1
    return returnString

