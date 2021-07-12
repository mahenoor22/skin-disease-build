import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)
print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/Final_model.h5'

# Load your own trained model
model = load_model(MODEL_PATH)

print('Model loaded. Start serving...')


def model_predict(img, model):
    img = img.resize((28, 28))

    # Preprocessing the image
    img = img.resize((28,28))
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1,28,28,3)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #img = preprocess_input(x, mode='tf')

    preds = model.predict(img)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

decode={
    0:'Actinic keratosis',
    1:'Basal cell carcinoma',
    2:'Benign keratosis-like lesions',
    3:'Dermatofibroma',
    4:'Melanocytic nevi',
    6:'Melanoma',
    5:'Vascular lesion'  
}
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)
        
        # Process your result for human
        max_val=np.where(preds==max(preds[0]))[1] # Max probability

        result = str(decode[max_val[0]])               #index of max_probability
        result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=str(round(max(preds[0]),3)))

    return None


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
    # Serve the app with gevent
