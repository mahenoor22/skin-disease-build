import base64
import io
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from PIL import Image
from flask import Flask

app = Flask(__name__)

def get_model():
    global model
    model=load_model('CNN_model.h5')
    print("Model loaded!")

def preprocess_image(image,target_size):
    if image.mode!="RGB":
        image=image.convert("RGB")
    image=image.resize(target_size)
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)
    return image
print("*loading Model.....")
get_model()

@app.route("/predict",methods=["POST"])
def predict():
    message=request.get_json(force=True)
    encoded=message['image']
    decoded=base64.b64decode(encoded)
    image=Image.open(io.BytesIo(decoded))
    processed_image=preprocess_image(image,target_size=(28,28))
    prediction=model.predict(processed_image).tolist()
    response={
        'prediction':{
            'mn': prediction[0][0],
            'ml':prediction[0][1],
            'bk':prediction[0][2],
            'bcc':prediction[0][3],
            'ak':prediction[0][4],
            'vl':prediction[0][5],
            'df':prediction[0][6]
        }
    }
    return jsonify(response)
