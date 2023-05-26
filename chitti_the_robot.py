

import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import numpy as np
from keras.models import load_model

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

# Create the main app interface
def app():
    st.title("Chatbot")
    st.write("Welcome! Start a conversation by typing in the message box below.")

    # Initialize the messages list
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # User input box
    user_input = st.text_input("User Input", key="user_input")

    if st.button("Send"):
        if user_input:
            # Add user message to session state
            st.session_state['messages'].append({"role": "user", "content": user_input})

            # Get chatbot response
            bot_response = chatbot_response(user_input)

            # Add chatbot response to session state
            st.session_state['messages'].append({"role": "bot", "content": bot_response})

    # Display messages
    for message in st.session_state['messages']:
        if message['role'] == 'user':
            st.text_area("User", value=message['content'], key=message['content'])
        elif message['role'] == 'bot':
            st.text_area("Bot", value=message['content'], key=message['content'])

if __name__ == '__main__':
    app()
