import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load word index for Sentiment Classification
word_to_index = imdb.get_word_index()

# Function to perform sentiment classification
def sentiment_classification(new_review_text, model):
    max_review_length = 500
    new_review_tokens = [word_to_index.get(word, 0) for word in new_review_text.split()]
    new_review_tokens = pad_sequences([new_review_tokens], maxlen=max_review_length)
    prediction = model.predict(new_review_tokens)
    if type(prediction) == list:
        prediction = prediction[0]
    return "Positive" if prediction > 0.5 else "Negative"

# Function to perform tumor detection
def tumor_detection(img, model):
    img = Image.open(img)
    img=img.resize((128,128))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img)
    return "Tumor Detected" if res else "No Tumor"

# Streamlit App
st.title("CLASSIFICATION WITH VARIOUS DEEP LEARNING MODELS")

# Choose between tasks
task = st.selectbox("SELECT TASK", ("TEXT_CLASSIFICATION","IMAGE CLASSIFICATION"))

if task == "TEXT_CLASSIFICATION":
    # Input box for new review
    new_review_text = st.text_area("Enter a New Review:", value="")
    if st.button("Submit") and not new_review_text.strip():
        st.warning("Please enter a review.")

    if new_review_text.strip():
        st.subheader("Choose a Model for Classification")
        model_option = st.radio("Select Model", ("Perceptron", "Backpropagation", "DNN", "RNN", "LSTM"))

        # Load models dynamically based on the selected option
        if model_option == "Perceptron":
            with open('MODELS\PERCEP_MODEL.pkl', 'rb') as file:
                model = pickle.load(file)
        elif model_option == "Backpropagation":
            with open('MODELS\Back_Prop.pkl', 'rb') as file:
                model = pickle.load(file)
        elif model_option == "DNN":
            model = load_model('MODELS\DNN_MODEL.keras')
        elif model_option == "RNN":
            model = load_model('MODELS\RNN_MODEL.keras')
        elif model_option == "LSTM":
            model = load_model('MODELS\LSTM_MODEL.keras')

        if st.button("Classify Sentiment"):
            result = sentiment_classification(new_review_text, model)
            
            st.write(f"The text is {result} ")

elif task == "IMAGE CLASSIFICATION":
    st.subheader("Tumor Detection")
    uploaded_file = st.file_uploader("Choose a tumor image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the tumor detection model
        model = load_model('MODELS\CNN_MODEL.keras')
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=False, width=200)
        st.write("")

        if st.button("Check for Tumor"):
            result = tumor_detection(uploaded_file, model)
            st.write(f"There is {result}**")
