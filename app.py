from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model.h5'

#Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')




def model_predict(img_path, model):
    image = Image.open(img_path)
    img = image.resize((64,64))
    img = np.asarray(img).reshape((1, 64, 64, 1))/255
    prediction = np.squeeze(model.predict(img))
    if prediction < 0.5:
        prediction = 0
    else:
        prediction = 1
    
    classes = {0:"Normal",1:"Pneumonia"}
    
    return classes[(prediction)]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
	
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        os.remove(file_path)#removes file from the server after prediction has been returned
        return preds
    return None

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':
        app.run()
