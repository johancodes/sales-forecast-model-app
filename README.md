# sales-forecast-model-app

Imagine being able to remove the guesswork from sales forecasting. We used Deep Learning (LSTM, Keras TensorFlow) to train a model to do just that. Our model was trained on 7 years of monthly Norwegian car sales data. Please provide an input between 10000 and 14000 for months 1 through 5. Your monthly numbers can rise or fall over the 5 months. 

Notes:
- App serves an endpoint.
- When data (car sales for five consecutive months) is sent to the model, a sales forecast for the sixth month is returned.
- Download app.py, my_model folder and .pkl to test on your machine. Use 'test data' to receive a forecast.
