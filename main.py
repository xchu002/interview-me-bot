from flask import Flask, render_template, request
import random
import json
import pickle 
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')
nltk.download('punkt')

ignore_letters = ['?', '!', '.', ',']

app = Flask(__name__)

def clean_up(sentence):
    sentence = nltk.word_tokenize(sentence)
    sentence = [lemmatizer.lemmatize(word.lower()) for word in sentence if word not in ignore_letters]
    return sentence

def bag_of_words(sentence):
    bag = [0] * len(words)
    for word in sentence:
        if word in words:
            bag[words.index(word)] = 1
    return bag

def predict(bag):
    output = model.predict(np.array([bag]))
    class_index = np.argmax(output)
    actual_class = classes[class_index]
    for intent in intents["intents"]:
        if actual_class == intent["tag"]:
            return intent["responses"]

@app.route("/", methods=['GET', 'POST'])
def interface():
    sentence = "hey"
    prediction = ""
    if request.method == "POST":
        sentence = request.form["sentence"]
        cleaned_up_sentence = clean_up(sentence)
        bag = bag_of_words(cleaned_up_sentence)
        prediction= predict(bag)[0]
    return render_template("index.html", sentence=prediction)

if __name__ == "__main__":
    app.run(debug=True)