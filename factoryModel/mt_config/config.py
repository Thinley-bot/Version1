import os
from os import path


BASE_PATH = 'D:/Nerd/Desktop/V1/factoryModel/output'
# Define the path where the model is saved
MODEL_PATH = os.path.join(BASE_PATH, 'model1.h5')
# Define the path to the tokenizer
ENG_TOK_PATH = os.path.join(BASE_PATH, 'eng_tokenizer.pkl')
DZO_TOK_PATH = os.path.join(BASE_PATH, 'dzo_tokenizer.pkl')
# Path to Standard lengths of German and English sentences
DZO_STDLEN = os.path.join(BASE_PATH, 'len_Dzongkha.pkl')
ENG_STDLEN = os.path.join(BASE_PATH, 'len_english.pkl')
