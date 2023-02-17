#Importation des librairies
import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns 
import pickle
from lightgbm import LGBMClassifier
import shapely 
import plotly.graph_objects as go
#from streamlit_echarts import st_echarts
from enum import Enum
import requests
import joblib
import shap

#Chargement du dataframe et du modèle
#model = joblib.load(open('clf_0.pkl','rb'))
data = pd.read_csv("./X_test.csv", index_col='SK_ID_CURR', encoding ='utf-8')


class Gender(Enum):
    MALE = 0.0
    FEMALE = 1.0

class FamilyStatus(Enum):
    NOT_MARRIED = 0.0
    MARRIED = 1.0 
    

def load_model():
        '''loading the trained model'''
        clf = joblib.load(open('./clf_0.pkl','rb'))
        return clf
    
clf = load_model()
    
#Récupération des informations générales clients 
@st.cache
def load_infos_gen(data):
    nb_credits = data.shape[0]
    rev_moy = round(data["AMT_INCOME_TOTAL"].mean(),2)
    credits_moy = round(data["AMT_CREDIT"].mean(), 2)
    #targets = data.TARGET.value_counts()

    return nb_credits, rev_moy, credits_moy#, targets
    #

#Récupération de l'identifiant client 
def identite_client(data, id):
    data_client = data[data.index == int(id)]
    return data_client


# Récupération de la prédiction du crédit pour les clients 
def load_prediction(X, id, clf):
    score = clf.predict(X[X.index == id])
    return float(score)

#Récupération de l'âge de la population de l'échantillon 
@st.cache
def load_age_population(data):
    data_age = round((data["DAYS_BIRTH"]/-365), 2)
    return data_age

#Récupération du revenu de la population de l'échantillon 
@st.cache
def load_income_population(data):
    df_income = pd.DataFrame(data["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
    return df_income

#Récupération du crédit de la population de l'échantillon 
@st.cache
def load_amt_credit_population(data):
    amt_credit_pop = pd.DataFrame(data["AMT_CREDIT"])
    return amt_credit_pop

#Récupération des défauts de paiement de la population de l'échantillon 
#@st.cache
#def load_obs_defaultpayment_population(data):
#    obs_def_payment_pop = pd.DataFrame(data["OBS_30_CNT_SOCIAL_CIRCLE"])
#    return obs_def_payment_pop

#Récupération des taux de paiement de la population de l'échantillon 
#@st.cache
#def load_payment_rate_population(data):
#    payment_rate_pop = pd.DataFrame(data["PAYMENT_RATE"])
#    return payment_rate_pop

#Récupération de la prédiction du crédit pour les clients 
@st.cache
def load_prediction(data, id, clf):
    score = clf.predict(data[data.index == id])
    return float(score)


#Chargement de l'identifiant client 
id_client = data.index.values

#Affichage du titre
html_temp = """
<div style="background-color: LightSeaGreen; padding:10px; border-radius:10px">
<h1 style="color: white; font-size: 30px; text-align:center">Dashboard for consumer credit prediction</h1>
</div>
<p style="font-size: 15px; font-weight: bold; text-align:center">Credit decision support…</p>
"""
st.markdown(html_temp, unsafe_allow_html=True)


### Création du menu sur le côté gauche ### 
#Sélection de l'id du client 
st.sidebar.header("**General Information**")

#Chargement de la boîte de sélection du client 
chk_id = st.sidebar.selectbox("Client ID", id_client)

#Chargement des informations générales 
nb_credits, rev_moy, credits_moy = load_infos_gen(data)
#rev_moy, , targets

#Nombre total de crédits de l'échantillon 
st.sidebar.markdown("<u>Number of loans in the sample :</u>", unsafe_allow_html=True)
st.sidebar.text(nb_credits)

#Revenu moyen des clients dans l'échantillon 
st.sidebar.markdown("<u>Average income (USD) :</u>", unsafe_allow_html=True)
st.sidebar.text(rev_moy)

#Crédit moyen des clients dans l'échantillon 
st.sidebar.markdown("<u>Average loan amount (USD) :</u>", unsafe_allow_html=True)
st.sidebar.text(credits_moy)

#PieChart concernant le nombre de crédits acceptés et refusés dans l'échantillon 
#st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
#fig, ax = plt.subplots(figsize=(3,3))
#plt.pie(targets, explode=[0, 0.1], labels=['No default', 'Default'], autopct='%1.1f%%', startangle=90)
#plt.title("Customer creditworthiness chart", fontweight = 'bold')
#st.sidebar.pyplot(fig)



### Création de la page principale ###
st.markdown("<h1 style='text-align: center; color: white; font-size: 20px;'>Features importance global for consumer credit prediction</h1>", unsafe_allow_html=True)

#Features Importance global 
X = data#.iloc[:, :-1]
fig, ax = plt.subplots(figsize=(10, 10))
explainer = shap.TreeExplainer(load_model())
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values[0], X, color_bar=False, plot_size=(5, 5))
st.pyplot(fig)



#Affichage des informations clients : Genre, Age, Statut familial, …
with st.expander("Customer information display"):
    #Affichage de l'identifiant du client sélectionné à partir du menu 
    st.write("Customer ID selection :", chk_id)
    
    if st.checkbox("Show customer information ?"):
        
        infos_client = identite_client(data, chk_id)
        client_gender = Gender(infos_client.iloc[0]['CODE_GENDER']).name
        client_age = ((infos_client.iloc[0]['DAYS_BIRTH'] /-365))
        client_status = FamilyStatus(infos_client.iloc[0]['NAME_FAMILY_STATUS_Married']).name
        st.write("**Infos clients**", infos_client)
        st.write("**Gender :**", (client_gender))
        st.write("**Age :**", round(client_age))
        st.write("**Family status :**", client_status)

        #Distribution par âge
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(((data['DAYS_BIRTH'])/-365), edgecolor = 'k', color="teal", bins=30)
        ax.axvline(int(client_age), color="green", linestyle='--')
        ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
        st.pyplot(fig)


        st.subheader("*Income - Credit -Default payment - Payment rate*")
        client_income = infos_client.iloc[0]['AMT_INCOME_TOTAL']
        client_credit = infos_client.iloc[0]['AMT_CREDIT']
        client_annuity = infos_client.iloc[0]['AMT_ANNUITY']
        client_property_credit = infos_client.iloc[0]['AMT_GOODS_PRICE']
        st.write("**Income total :**", round(client_income))
        st.write("**Credit amount :**", round(client_credit))
        st.write("**Credit annuities :**", round(client_annuity))
        st.write("**Amount of property for credit :**", round(client_property_credit))

        #Distribution par revenus
        data_income = load_income_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="teal", bins=20)
        ax.axvline(int(client_income), color="green", linestyle='--')
        ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
        st.pyplot(fig)

        #Distribution des crédits demandés dans l'échantillon 
        data_amt_credit = load_amt_credit_population(data)
        client_credit = infos_client.iloc[0]['AMT_CREDIT']
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_amt_credit["AMT_CREDIT"], edgecolor = 'k', color="teal", bins=20)
        ax.axvline(int(client_credit), color="green", linestyle='--')
        ax.set(title='Customer credit', xlabel='Credit (USD)', ylabel='')
        st.pyplot(fig)

        #Distribution des défauts de paiements sur 30 jours dans l'échantillon 
        #data_defpayment = load_obs_defaultpayment_population(data)
        #client_defpayment = infos_client.iloc[0]['OBS_30_CNT_SOCIAL_CIRCLE']
        #fig, ax = plt.subplots(figsize=(10, 5))
        #sns.histplot(data_defpayment["OBS_30_CNT_SOCIAL_CIRCLE"], edgecolor = 'k', color="teal", bins=20)
        #ax.axvline(int(client_defpayment), color="green", linestyle='--')
        #ax.set(title='Customer default payment', xlabel='Number of default payment (on 30 days)', ylabel='')
        #st.pyplot(fig)
        
        #Distribution des taux de paiement ("payment_rate") dans l'échantillon 
        #data_payment_rate = load_payment_rate_population(data)
        #client_payment_rate = infos_client.iloc[0]['PAYMENT_RATE']
        #fig, ax = plt.subplots(figsize=(10, 5))
        #sns.histplot(data_payment_rate["PAYMENT_RATE"], edgecolor = 'k', color="teal", bins=20)
        #ax.axvline((client_payment_rate), color="green", linestyle='--')
        #ax.set(title='Customer payment rate', xlabel='Payment rate', ylabel='')
        #st.pyplot(fig)

        
#Customer probability repayment display
with st.expander('Credit default probability'):

    infos_client = identite_client(data, chk_id)
    #client_target = infos_client.iloc[0]['TARGET']
    #prediction = load_prediction(data, chk_id, clf)
    prediction = load_prediction(data, chk_id, clf)
    
   # equests.get('http://localhost:8000/predict?id=' + str(int(chk_id)))
    st.write(prediction)
    score = prediction
    #score = prediction.json()['score']
    
    #formatted_score = round(float(score)*100, 2)
    #formatted_prediction = round(float(prediction)*100, 2)
    #st.write("**Your credit default probability is :** {:.0f} %".format(formatted_prediction))
    st.write("**Your credit default probability is :**",score)
    
    #if client_target == 1.0 :
    #if score >= 0.275:
    if score == 1:
        st.error('Your credit application has been rejected! Please contact customer support for more information.', icon="❌")
    else : 
        st.success('Congratulations! Your credit application has been accepted!', icon="✅")
        
    option = {
        "tooltip": {
            "formatter": '{a} <br/>{b} : {c}%'
        },
        "series": [{
            "name": 'Credit default probability',
            "type": 'gauge',
            "startAngle": 180,
            "endAngle": 0,
            "progress": {
                "show": "true"
            },
            "radius":'100%', 

            "itemStyle": {
                "color": '#5499C7',
                "shadowColor": 'rgba(0,138,255,0.45)',
                "shadowBlur": 10,
                "shadowOffsetX": 2,
                "shadowOffsetY": 2,
                "radius": '55%',
            },
            "progress": {
                "show": "true",
                "roundCap": "true",
                "width": 15
            },
            "pointer": {
                "length": '60%',
                "width": 8,
                "offsetCenter": [0, '5%']
            },
            "detail": {
                "valueAnimation": "true",
                "formatter": '{value}%',
                "backgroundColor": '#5499C7',
                "borderColor": '#999',
                "borderWidth": 4,
                "width": '60%',
                "lineHeight": 20,
                "height": 20,
                "borderRadius": 188,
                "offsetCenter": [0, '40%'],
                "valueAnimation": "true",
            },
            "data": [{
                "value": score,
                "name": 'Credit default probability'
            }]
        }]
    };


    #st_echarts(options=option, key="1")

    
#Customer solvability display
with st.expander("Customer file analysis"):
    
    #Customer solvability display
    st.markdown("<u>Customer Data :</u>", unsafe_allow_html=True)
    st.write(identite_client(data, chk_id))

    
    #Feature importance / description

    if st.checkbox("Customer ID {:.0f} feature importance ?".format(chk_id)):
        shap.initjs()
        X = data 
        X = X[X.index == chk_id]
        number = st.slider("Pick a number of features…", 0, 20, 5)

        fig, ax = plt.subplots(figsize=(10, 10))
        explainer = shap.TreeExplainer(load_model())
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
        st.pyplot(fig)