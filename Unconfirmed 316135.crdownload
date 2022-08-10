# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 22:30:33 2022

@author: 91931
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

import pickle

app = Flask(__name__)
model = pickle.load(open('iphonetext.pkl','rb'))  

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
print(cv)
corpus=pd.read_csv('corpus_dataset1.csv')

corpus1=corpus['corpus'].tolist()
X = cv.fit_transform(corpus1).toarray()


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
        
    '''
    For rendering results on HTML GUI
    '''
    text = request.args.get('text')
    text=[text]
    input_data = cv.transform(text).toarray()
    
    prediction = model.predict(input_data)
    if prediction==1:
      result="Positive"
    else:
      result="Negative"
            
    return render_template('index.html', prediction_text='Iphone Review is : {}'.format(result))

if __name__ == "__main__":
    app.run(debug =True)