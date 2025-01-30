import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Import pad_sequences

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    # Preprocess the text
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the tokenizer and model
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))  # Load tokenizer
model = pickle.load(open('model2.pkl', 'rb'))  # Load prediction model

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input SMS
    transformed_sms = transform_text(input_sms)

    # 2. Convert text to sequences using the tokenizer
    sequence_input = tokenizer.texts_to_sequences([transformed_sms])  # Convert text to sequences

    # 3. Pad the sequences to ensure they have the correct shape
    max_length = 100  # Adjust this if your model requires a different max length
    padded_input = pad_sequences(sequence_input, maxlen=max_length)  # Pad sequences

    # 4. Predict
    result = model.predict(padded_input)[0]  # Use padded input for prediction

    # 5. Display the result
    if result > 0.5:
        #st.header(result)
        st.header("Spam")
    else:
        #st.header(result)
        st.header("Not Spam")
