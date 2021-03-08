from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
from PIL import Image

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

def get_model():
    global model
    model=load_model('Final_model.h5')
    model.load_weights("Final_model.h5")
    print("Model loaded!")

print("*loading Model.....")
get_model()

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))     
    img = Image.open('static/{}.jpg'.format(COUNT)) 
    img = img.resize((28,28))
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1,28,28,3)

    prediction = model.predict(img)
    print(prediction)
    
    ak = round(prediction[0,0],2)
    bcc = prediction[0,1]
    bk = prediction[0,2]
    df = prediction[0,3]
    mn = prediction[0,4]
    ml = prediction[0,6]
    vl = prediction[0,5]
    
    max_val=np.where(prediction==max(prediction[0]))[1]
    prediction=np.around(prediction,decimals=4)
    print(max_val)
    preds = np.array([ak,bcc,bk,df,mn,ml,vl,max_val[0]])
    COUNT += 1
    return render_template('prediction.html', data=preds)


@app.route('/load_img')
def load_img()
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)



