
# IMPORT LIBRARIES

import tensorflow as tf 
from tensorflow import keras 

import pandas as pd
import numpy as np

import sklearn
from sklearn.preprocessing import StandardScaler

import pickle 
from pickle import load

import flask
from flask import Flask, request 
from flask_cors import CORS  


# LOAD MODEL AND SCALER

saved_forecast_model = tf.keras.models.load_model('my_model')

scaler_pkl = pickle.load(open('scaler.pkl', 'rb'))

# DEFINE APP AND APP ROUTE

app = Flask(__name__)
CORS(app)  

@app.route('/api_predict', methods=['POST', 'GET'])

def api_forecast():

    if request.method == 'GET':
        return "Please Send POST Request (api_predict)"

    elif request.method == 'POST':

        data = request.get_json()

        input1 = data['input1']
        input2 = data['input2']
        input3 = data['input3']
        input4 = data['input4']
        input5 = data['input5']

        # create array and reshape it
        data = np.array([[input1, input2, input3, input4, input5]]).reshape(5,1)

        # scale the data
        input_sc = scaler_pkl.transform(data)
        #return str(input_sc) ## good

        # reshape to 3D tensor
        input_sc = input_sc.reshape(1,5,1)

        # forecast with model
        forecast = saved_forecast_model.predict(input_sc)

        # inverse transform the forecast, round it up, convert to integer, flatten and extract number
        forecast = scaler_pkl.inverse_transform(forecast).round().astype(int).flatten()[0]

        # return forecast 
        return str(forecast)
        

if __name__ == "__main__":
    app.run() 