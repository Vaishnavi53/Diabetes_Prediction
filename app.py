import os
import numpy as np
import flask
import joblib
from flask import Flask, render_template, request

# Creating instance of the class
app = Flask(__name__)

# Load the trained KNN model
knn_model = joblib.load('knn_model.pkl')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 8)
    result = knn_model.predict(to_predict)
    return result[0]

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        
        if int(result) == 1:
            prediction = 'Warning: High risk of diabetes. Please consult your physician.'
        else:
            prediction = 'Great news: Low risk of diabetes. Keep up the healthy lifestyle!'
            
        return render_template("result.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
