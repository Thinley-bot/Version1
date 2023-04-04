from keras.models import load_model
from factoryModel.mt_config.config import *
from factoryModel.utils.nmt import *
from flask import Flask, request, jsonify, Response,render_template,json

#initializing the flask application
app = Flask(__name__)


## Define the file path to the model
modelPath = MODEL_PATH

# Load the model from the file path
model = load_model(modelPath)
print(model.summary())
# Get the paths for all the files and variables stored as pickle files
Eng_tokPath = ENG_TOK_PATH
Dzo_tokPath = DZO_TOK_PATH
Dzo_length = DZO_STDLEN

# Load the tokenizer from the pickle file
Eng_tokenizer = load_files(Eng_tokPath)
Dzo_tokenizer = load_files(Dzo_tokPath)
# Load the standard lengths
Dzo_stdlen = load_files(Dzo_length)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

        
@app.route('/api',methods=['POST'])
def entry():
    # Get the Dzongkha sentence from the Input site
    data = request.get_json()
    text=data['text']

    #clean the input Dzongkha text
    cleanText = cleanInput(text)
    
    # Converting to sequences and padding them
    # Encode the inputsentence as sequence of integers
    seq1 = encode_sequences(Dzo_tokenizer, int(Dzo_stdlen), cleanText)

    #Get the translation
    translation = generatePredictions(model,Eng_tokenizer,seq1)
    
    #response in the form of the json object
    js = json.dumps({'text':translation})    
    resp = Response(js, status=200)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
        

if __name__ == '__main__':
      app.run(debug=True)
      
'''try: ཁོ་ གིས་ མོ་ ལུ་ དྲི་བ་ དག་པ་ཅིག་ འདྲི་ ནུག །
        ང་བཅས་ ཀྱིས་ ཁྱོད་ སླབ་ ས་ མ་ གོ །
        ཁྱོད་ ཚུ་ ག་ཏེ་ ལས་ འོངམ་ སྨོ ། Where do you all come from?
        ཁྱོད་ ག་ གི་ སྐོར་ལས་ སླབ་ སྨོ ། Who are you talking about?	
'''