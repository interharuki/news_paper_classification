# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:53:33 2019

@author: inter
"""

from flask import Flask, render_template, request

from wtforms import Form, TextAreaField, validators


import sqlite3

import os

import numpy as np



# import HashingVectorizer from local dir

#from vectorizer import vect



app = Flask(__name__)



######## Preparing the Classifier

cur_dir = os.path.dirname(__file__)
"""
clf = pickle.load(open(os.path.join(cur_dir,

                 'pkl_objects',

                 'classifier.pkl'), 'rb'))
"""
db = os.path.join(cur_dir, 'reviews.sqlite')

#####################
from sklearn.externals import joblib

def get_model_path(type='model'):
    model_name = "mlp"

    return 'models/'+model_name+"_"+type+'.pkl'

model = joblib.load(get_model_path())
classes =  joblib.load(get_model_path('class')).tolist()
vectorizer = joblib.load(get_model_path('vect'))
le = joblib.load(get_model_path('le'))
#########################

def classify(document):


    X = vectorizer.transform([document])

    key = model.predict(X)

    #proba = np.max(clf.predict_proba(X))
    probability = model.predict_proba(X)[0]
    proba= np.max(probability)
    company_list =["朝日新聞","毎日新聞","日経新聞","産経新聞","読売新聞"]
      
    return company_list[key[0]],proba,probability 



def train(document, y):

    X = vectorizer.transform([document])

    model.partial_fit(X, [y])



def sqlite_entry(path, document, y):

    conn = sqlite3.connect(path)

    c = conn.cursor()

    c.execute("INSERT INTO review_db (review, sentiment, date)"\

    " VALUES (?, ?, DATETIME('now'))", (document, y))

    conn.commit()

    conn.close()



######## Flask

class ReviewForm(Form):

    moviereview = TextAreaField('',

                                [validators.DataRequired(),

                                validators.length(min=15)])



@app.route('/')

def index():

    form = ReviewForm(request.form)

    return render_template('reviewform.html', form=form)



@app.route('/results', methods=['POST'])

def results():

    form = ReviewForm(request.form)

    if request.method == 'POST' and form.validate():

        review = request.form['moviereview']

        y, proba,probability = classify(review)

        return render_template('results.html',

                                content=review,

                                prediction=y,
                                
                                asahi_prob = round(probability[0]*100,2),
                               
                                mainichi_prob = round(probability[1]*100,2),
                                
        
                                nikkei_prob = round(probability[2]*100,2),
                                 
                                sankei_prob = round(probability[3]*100,2),
                        
                                yomiuri_prob = round(probability[4]*100,2),
        
                                

                                probability=round(proba*100, 2))

    return render_template('reviewform.html', form=form)



@app.route('/thanks', methods=['POST'])

def feedback():

    feedback = request.form['feedback_button']

    review = request.form['review']

    prediction = request.form['prediction']



    inv_label = {'negative': 0, 'positive': 1}

    y = inv_label[prediction]

    if feedback == 'Incorrect':

        y = int(not(y))

    train(review, y)

    sqlite_entry(db, review, y)

    return render_template('thanks.html')



if __name__ == '__main__':

    app.run(debug=True)