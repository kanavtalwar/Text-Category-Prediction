# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 00:08:26 2019

@author: USER
"""

from flask import Flask,render_template,url_for,request
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import flask
import pickle
with open('model/KTcategory-predictor.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('model/KTtfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)
    
app = Flask(__name__, template_folder='templates')
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    category_map = {'talk.politics.misc': 'Politics', 'rec.autos': 'Autos', 
    'rec.sport.hockey': 'Hockey', 'sci.electronics': 'Electronics', 
    'sci.med': 'Medicine'}
    count_vectorizer = CountVectorizer()
    training_data = fetch_20newsgroups(subset='train',categories=category_map.keys(), shuffle=True, random_state=5)
    train_tc = count_vectorizer.fit_transform(training_data.data)
    
    tfidf = TfidfTransformer()
    train_tfidf = tfidf.fit_transform(train_tc)

      
    if request.method == 'POST':
        message = request.form['message']
        data = message.split(" ")
        input_tc = count_vectorizer.transform(data)
        input_tfidf = tfidf.transform(input_tc)
        predictions = model.predict(input_tfidf)
        for category in predictions:
            my_prediction = category_map[training_data.target_names[category]]
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)