import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding, Dropout, add
from keras.models import load_model

# Load the .h5 model
model = load_model('image_caption.h5')
tokenizer = Tokenizer()
max_length=35
# Load pre-trained model
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Set Streamlit configurations
st.set_page_config(page_title="Image Classifier App", layout="wide")


# Function to preprocess the input image
def preprocess_image(image):
    image = load_img(image, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    return image

# Function to make predictions on the input image
def predict(image):
    image = preprocess_image(image)
    feature = vgg_model.predict(image, verbose=0)
    preds = predict_caption(model, feature, tokenizer, max_length)
    preds=preds[8:-7]
    return preds

def idx_word(integer,tok):
    for word,index in tok.word_index.items():
        if index== integer:
            return word
    return None

def predict_caption(model,image,tok,max_len):
    in_text="startseq"
    for i in range(max_len):
        seq=tok.texts_to_sequences([in_text])[0]
        seq=pad_sequences([seq],max_len)
        yhat = model.predict([image, seq], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_word(yhat, tok)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

# Streamlit app
def main():
    st.title("Image Classifier App")
    st.write("Upload an image and the app will predict its class.")

    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                predictions = predict(image)
            
            st.write("Top predictions:")
            for _, label, confidence in predictions:
                st.write(f"{label}: {round(confidence * 100, 2)}%")

# Run the app
if __name__ == "__main__":
    main()
