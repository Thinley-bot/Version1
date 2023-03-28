from pickle import load
from numpy import argmax
from pickle import dump
from keras.utils import pad_sequences
from keras.models import load_model
from numpy import array
import re
from keras.preprocessing.text import Tokenizer
import os


# Function to Save data to pickle form
def save_clean_data(data,filename):
    dump(data,open(filename,'wb'))
    print('Saved: %s' % filename)

# Function to load pickle data from disk
def load_files(filename):
    return load(open(filename,'rb'))

# Function to clean the input data
def cleanInput(lines):
    cleanSent = []
    cleanDocs = list()
    for docs in lines.split():
        line=str(docs)
        line = re.sub(r'་(?=\s|$)', '',line)
        line=re.sub(r"\s*།\s*$", '', line)
        line=re.sub(r'\s#\s',' ',line)
        cleanDocs.append(line)
    cleanSent.append(' '.join(cleanDocs))
    return array(cleanSent)

# Function for encoding and padding sequences

def encode_sequences(tokenizer,length, lines):
    # Sequences as integers
    X = tokenizer.texts_to_sequences(lines)
    # Padding the sentences with 0
    X = pad_sequences(X,maxlen=length,padding='post')
    return X

# Generate target sentence given source sequence
def Convertsequence(tokenizer, source):
    target = []
    # Creating the reverse dictionary
    reverse_eng = tokenizer.index_word
    for i in range(len(source)):
        if source[i] == 0:
            continue
        if int(source[i]) in reverse_eng:
            target.append(reverse_eng[int(source[i])])
    return ' '.join(target)

'''def Convertsequence(tokenizer,source):

    target = list()
    # Creating the reverse dictionary
    reverse_eng = tokenizer.index_word  
    for i in source:
        if i == 0:
            continue
        target.append(reverse_eng[int(i)])
    return ' '.join(target)'''


# load model
model = load_model('model1.h5')

# Function to generate predictions from source data
def generatePredictions(model,tokenizer,data):
    prediction = model.predict(data,verbose=0)

    AllPreds = []
    for i in range(len(prediction)):
        predIndex = [argmax(prediction[i, :, :], axis=-1)][0]
        target = Convertsequence(tokenizer,predIndex)
        AllPreds.append(target)
    return AllPreds