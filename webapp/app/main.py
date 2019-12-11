from flask import Flask, request, render_template
#import keras
#import tensorflow as tf
import numpy as np
#from sentiment_models import SimpleModel
#from synonyms import generate_alternative_sentences
#from alternatives import sort_sentences_by_sentiment
from transformer import *

# lsof -i:5000
# kill PID

app = Flask(__name__)
#graph = None
model = None
#sentiment = None
#text_input = None

def load_model():
    global model
    print('Loading model...')
    model = TransformerModel()
    print('Finished loading model!')

# Routing for main
@app.route('/')
def index_page():
    return render_template('form.html')

    # Routing for main
@app.route('/form')
def home_page():
    return render_template('form.html')

# Routing for main
@app.route('/home')
def form_page():
    return render_template('home.html')

# Routing for post
@app.route('/', methods=['POST'])
def form_post_page():
    text_input = request.form['text']
    paraphrase = model.paraphrase_sentence(text_input)
    
    print(text_input, paraphrase)

    variables = {
        'paraphrase': paraphrase,
        'text_input' : text_input
    }
    return render_template('form.html', **variables)

# Runs app
if __name__ == "__main__": 
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)