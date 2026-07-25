import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pandas as pd
import os
import pickle
import numpy as np
import json
import random

from keras.models import load_model
from flask import Flask, render_template, request
from datetime import datetime

# Load trained model
model = load_model('model.h5')

# Load chatbot data
with open('data.json', encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Flask app
app = Flask(
    __name__,
    template_folder='templates',
    static_folder='static'
)

# Clean sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)

    sentence_words = [
        lemmatizer.lemmatize(word.lower())
        for word in sentence_words
    ]

    return sentence_words

# Bag of words
def bow(sentence, words):

    sentence_words = clean_up_sentence(sentence)

    bag = [0] * len(words)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)

# Predict class
def predict_class(sentence, model):

    p = bow(sentence, words)

    res = model.predict(np.array([p]), verbose=0)[0]

    ERROR_THRESHOLD = 0.50

    results = [
        [i, r]
        for i, r in enumerate(res)
        if r > ERROR_THRESHOLD
    ]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({
            "intent": classes[r[0]],
            "probability": str(r[1])
        })

    return return_list

# Get response
def getResponse(ints, intents_json):

    if len(ints) == 0:
        return "Sorry, this information is not available in the system."

    tag = ints[0]['intent']

    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "Sorry, this information is not available in the system."

# Chatbot response
def chatbot_response(msg):

    ints = predict_class(msg, model)

    print("Prediction:", ints)

    if len(ints) == 0:
        return "Sorry, this information is not available in the system."

    confidence = float(ints[0]['probability'])

    if confidence < 0.50:
        return "Sorry, this information is not available in the system."

    return getResponse(ints, intents)

# Save reward with timestamp
def save_reward(question, response, reward):

    data = {
        "question": [str(question)],
        "response": [str(response)],
        "reward": [str(reward)],
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    }

    df = pd.DataFrame(data)

    file_exists = os.path.isfile("rewards.csv")

    df.to_csv(
        "rewards.csv",
        mode='a',
        header=not file_exists,
        index=False
    )

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Chat response route
@app.route("/get")
def get_bot_response():

    userText = request.args.get('msg')

    if not userText:
        return "Please enter a question."

    return chatbot_response(userText)

# Reward route
@app.route("/reward")
def reward():

    question = request.args.get("question")
    response = request.args.get("response")
    reward_value = request.args.get("value")

    save_reward(question, response, reward_value)

    return "Reward Saved"

# Run app
if __name__ == "__main__":
    app.run(debug=True, port=8000)