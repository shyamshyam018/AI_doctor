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
    # Initialize the chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Function to display a chat message
    def display_chat_message(role, message):
        if role == 'user':
            st.write(':smiley:', message)
        elif role == 'bot':
            st.write(':robot_face:', message)

    # Display the chat interface
    st.title('Chat Application')

    # Create a column layout
    col1, col2 = st.beta_columns([1, 4])

    # Sidebar for user input and send button
    with col1:
        st.text('')  # Add space for alignment
        user_input = st.text_input('Type your message here')

        # Move the send button next to the input box
        send_button = st.button('Send')

    # Main content area for chat history
    with col2:
        st.header('Chat History')
        st.markdown('---')  # Add a horizontal line for separation

        # Display chat history
        for role, message in st.session_state['chat_history']:
            display_chat_message(role, message)

    # Process user input and generate response
    if send_button:
        # Get user input
        user_message = user_input.strip()

        # Add user message to chat history
        st.session_state['chat_history'].append(('user', user_message))

        # Generate bot response
        bot_response = chatbot_response(user_message)

        # Add bot response to chat history
        st.session_state['chat_history'].append(('bot', bot_response))

        # Clear user input
        user_input = ''

if __name__ == '__main__':
    app()
