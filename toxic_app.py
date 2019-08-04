from flask import Flask, request, make_response, jsonify, Response
from flask_cors import CORS
import urllib.request, json
import datetime as dt
import pandas as pd
import numpy as np
import pickle as p
import re
import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

app = Flask(__name__)
CORS(app)
# app.listen(process.env.PORT || 3000)

@app.route("/")
def hello():
    return ('00110001 01000111 01111000 01001101')

def tf_idf_vectorizer():
    return TfidfVectorizer(ngram_range=(1,2),tokenizer=tokenize, min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1, smooth_idf=1, sublinear_tf=1 )

def tokenize(s):
    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
    return re_tok.sub(r' \1 ', s).split()

@app.route("/predict_toxicity", methods=['POST'])
def predict_toxicity():
    comment_instance = request.get_json(silent=True, force=True)
    # comment_instance = {'comment_text': 'You don\'t have brains'}
    vec_loc = 'models/vectorizer.pickle'

    vec = p.load(open(vec_loc, 'rb'))

    insult_model_loc = 'models/insultmodel.pickle'
    toxic_model_loc = 'models/toxicmodel.pickle'
    severe_toxic_model_loc = 'models/severe_toxicmodel.pickle'
    threat_model_loc = 'models/threatmodel.pickle'
    obscene_model_loc = 'models/obscenemodel.pickle'

    insult_model = p.load(open(insult_model_loc, 'rb'))
    toxic_model = p.load(open(toxic_model_loc, 'rb'))
    severe_toxic_model = p.load(open(severe_toxic_model_loc, 'rb'))
    threat_model = p.load(open(threat_model_loc, 'rb'))
    obscene_model = p.load(open(obscene_model_loc, 'rb'))

    vec = p.load(open(vec_loc, 'rb'))
    # vec = tf_idf_vectorizer()
    print("iterable comment: ", comment_instance)

    unseen_tf_doc = vec.transform([comment_instance['comment_text']])

    insult_prediction = insult_model.predict_proba(unseen_tf_doc)
    toxic_prediction = toxic_model.predict_proba(unseen_tf_doc)
    severe_toxic_prediction = severe_toxic_model.predict_proba(unseen_tf_doc)
    threat_prediction = threat_model.predict_proba(unseen_tf_doc)
    obscene_prediction = obscene_model.predict_proba(unseen_tf_doc)

    insult_conf = insult_prediction[0][1]
    toxic_conf = toxic_prediction[0][1]
    severe_toxic_conf = severe_toxic_prediction[0][1]
    threat_conf = threat_prediction[0][1]
    obscene_conf = obscene_prediction[0][1]

    print(insult_conf)
    print(toxic_conf)
    # print(severe_toxic_conf)
    print(threat_conf)
    print(obscene_conf)

    results = [insult_conf, toxic_conf, threat_conf, obscene_conf]

    if (insult_conf > 0.5 or toxic_conf > 0.5 or threat_conf > 0.5 or obscene_conf > 0.5):
        pred_val = 1
    else:
        pred_val = 0

    # prediction = y_pred

    prediction = {'prediction': str(pred_val), 'insult_score': str(insult_conf), 'toxic_score': str(toxic_conf), 'threat_score': str(threat_conf), 'obscene_score': str(obscene_conf)}
    prediction = json.dumps(prediction)
    print (prediction)

    resp = Response(prediction, status=200, mimetype='application/json')
    return resp

if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 5000))
    # app.run(host='0.0.0.0', port=port)
    # get_specific_invoice('1211')
    # vec_loc = 'models/vectorizer.pickle'
    # vec = p.load(open(vec_loc, 'rb'))
    app.debug = True
    app.run()
