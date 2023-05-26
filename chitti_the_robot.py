import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import numpy as np
from keras.models import load_model

import random
import streamlit as st

# Load the saved model
model = load_model('chatbot_model.h5')

# Load the words and classes from the pickle files
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Create the lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_sentence(sentence):
    # Tokenize the sentence
    words = nltk.word_tokenize(sentence)
    # Lemmatize and convert to lowercase
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words

def predict_intent(sentence):
    # Preprocess the sentence
    sentence_words = preprocess_sentence(sentence)
    # Create the bag of words
    bag = [1 if word in sentence_words else 0 for word in words]
    # Make the prediction
    result = model.predict(np.array([bag]))[0]
    # Get the predicted intent
    intent_index = np.argmax(result)
    intent = classes[intent_index]
    # Get the probability
    probability = result[intent_index]
    return intent, probability

def chatbot_response(msg):
    intent, probability = predict_intent(msg)
    if probability > 0.7:
        # Retrieve a random response from the intent's patterns
        for intent_data in intents['intents']:
            if intent_data['tag'] == intent:
                responses = intent_data['responses']
                response = random.choice(responses)
                return response
    else:
        return "I'm sorry, but I didn't understand that."

# Load the intents from the JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Main chat interface
def chat_interface():
    st.title("Chatbot")
    chat_log = st.text_area("Chat Log", value="", height=300, disabled=True)
    user_input = st.text_input("User Input")
    if st.button("Send"):
        if user_input.strip() != "":
            chat_log += f"You: {user_input}\n"
            bot_response = chatbot_response(user_input)
            chat_log += f"Bot: {bot_response}\n"
            st.text_area("Chat Log", value=chat_log, height=300, disabled=True)

# Run the chat interface
chat_interface()
