import sys
sys.path.insert(0, '../src')
import uvicorn
from fastapi import FastAPI 
#from sklearn.externals 
import joblib
import pickle
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns 
from lightgbm import LGBMClassifier
import shap 
import plotly.graph_objects as go
from enum import Enum

# Création de l'instance API et chargement du dataset
app = FastAPI()


# Prérequis API (data + model + fonction de prédiction)
X = pd.read_csv("/home/mogzs/openclassroom/p7/data_test.csv", index_col='SK_ID_CURR', encoding ='utf-8')

def load_model():
        '''loading the trained model'''
        clf = joblib.load(open('/home/mogzs/openclassroom/p7/clf_0.pkl','rb'))
        return clf
    
clf = load_model()

# Récupération de la prédiction du crédit pour les clients 
def load_prediction(X, id, clf):
    score = clf.predict(X[X.index == id])
    return float(score)
    #predict_proba

@app.get('/')
def home():
    return {'text': 'Welcome to Home Credit Prediction'}

# Endpoint qui retourne le score en fonction de l'id client
@app.get('/predict')
def read_items(id: int):
 #   if id < 100002 or id > 123325:
 #       return 'id client inconnu'
    score = load_prediction(X, id, clf)
    
    return float(score)
    #return {'score': score}

# Exécuter l'API avec uvicorn
#    Exécution sur http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    #uvicorn model:app --reload;