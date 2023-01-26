
from lib2to3.pgen2.pgen import DFAState
import streamlit as st
#from st_aggrid import AgGrid
#from st_aggrid.shared import JsCode
#from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
import plotly as plt
import plotly.express as px
#from joblib import load
import pickle
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split 
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from sklearn import preprocessing
from xgboost import XGBClassifier
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
import hashlib
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import operator 
from heapq import nlargest
import time
import json
import csv
from SPARQLWrapper import SPARQLWrapper, JSON
import ssl
import streamlit_authenticator as stauth
import yaml
from streamlit_authenticator import hasher 
import pymysql
import functools
import datetime
import pandas as pd 
import numpy as np 
import requests
import json
import mysql.connector
from mysql.connector import errorcode
import nltk.data
from tables import index
from sklearn import preprocessing
import jprops
from jproperties import Properties
import operator 
from heapq import nlargest
from sqlalchemy import create_engine
import sqlalchemy 
from sklearn.metrics import classification_report as creport





def get_model_info_mysql():
   

    host="linked.aub.edu.lb"
    port=3306
    database ="reviews_actions_ml"


    
    configs = Properties()
    
    with open('dbconfig.properties', 'rb') as config_file:
         configs.load(config_file)
           
    dbConnection = mysql.connector.connect(user=configs.get("db.username").data, password=configs.get("db.password").data, host="linked.aub.edu.lb", database="reviews_actions_ml")



    model_data = pd.read_sql_query("SELECT * FROM ML_models", dbConnection)

      
    
    dbConnection.commit()
   
    

    return  model_data





def preprocess_text(x):
    
    #x = df_train[["Review Body"]]

    #x=x.iloc[:,0]
    #y=y.iloc[:,:]
    #X=x.to_dict()
    X=list(x) 

    count_vect=CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    X_train_counts=count_vect.fit_transform(X)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf= X_train_tfidf.toarray()

    return X_train_tfidf, count_vect, tfidf_transformer


def count_vectorizer(annotation, count_vect, tfidf_transformer) -> float:
    
    annotation_counts=count_vect.transform(annotation)
    annotation_tfidf = tfidf_transformer.fit_transform(annotation_counts)
    annotation_tfidf= annotation_tfidf.toarray()
    return annotation_tfidf



# Train the action flag model. The dataset is manually done and not from sparql. 
def train_model_action_flag(df):

    #df = pd.read_csv("Action classified dataset.csv")
    x = df[["annotation"]]
    y= df[["ActionFlag"]]

    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state =1, shuffle = True)

    x=x.iloc[:,0]
    y=y.iloc[:,:]
    #X=x.to_dict()
    X=list(x)  
    #x = x.transpose()
   
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state =1, shuffle = True)
    
    #X_train_tfidf, count_vect, tfidf_transformer = preprocess_text(x_train)

    count_vect=CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    X_train_counts=count_vect.fit_transform(X)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf= X_train_tfidf.toarray()

    clf= SVC(random_state = 0,  probability=True)
    clf.fit(X_train_tfidf, y.values)
    acc = clf.score(X_train_tfidf, y.values)
      
    #x_test_tfidf = count_vectorizer(x_test, count_vect, tfidf_transformer)
    #y_pred_flag = clf.predict(x_test_tfidf)
   

    return acc, clf
   
   
   
def save_action_noaction_model(clf):
   
    filename_svm = 'SVM_action_noaction_model_version2.sav'
    pickle.dump(clf, open(filename_svm, 'wb'))



def train_environment_detection_model(df_train):
    #df_train["Environment"] = df_train["Environment"].str.replace("http://linked.aub.edu.lb/actionrec/Environment/", "")

    x = df_train[["annotation"]]
    y = df_train[["Environment"]]
    
    x=x.iloc[:,0]
    y=y.iloc[:,:]


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state =1, shuffle = True)
    
    X_train_tfidf, count_vect, tfidf_transformer = preprocess_text(x_train)
    clf= SVC(random_state = 0)
    clf.fit(X_train_tfidf, y_train.values)
    acc = clf.score(X_train_tfidf, y_train.values)
    print("Accuracy for Environment detection model on training set (80% of total dataset) is :", acc)
    

    x_test_tfidf = count_vectorizer(x_test, count_vect, tfidf_transformer)
    y_pred_env = clf.predict(x_test_tfidf)
    
   
    
    return creport(y_test, y_pred_env), clf


def save_env_model(clf):
   filename_clf = 'SVM_environment_model_version2.sav'
    pickle.dump(clf, open(filename_clf, 'wb'))
      
      
      
      


def train_valence_detection_model(df_train):
    x = df_train[["annotation"]]
    y = df_train[["Valence"]]
    
    x=x.iloc[:,0]
    y=y.iloc[:,:]


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state =1, shuffle = True)

    X_train_tfidf, count_vect, tfidf_transformer = preprocess_text(x_train)
    LR= linear_model.LogisticRegression()
    LR.fit(X_train_tfidf, y_train.values)
    LR.score(X_train_tfidf, y_train.values)
 


    x_test_tfidf = count_vectorizer(x_test, count_vect, tfidf_transformer)
    y_pred_val = LR.predict(x_test_tfidf)

    

    return creport(y_test, y_pred_val), LR
   
   
   
def save_valence_model(LR):
    filename_LR = 'LR_valence_model_version2.sav'
    pickle.dump(LR, open(filename_LR, 'wb'))



def train_object_detection_model(df_train):
    x = df_train[["annotation"]]
    y = df_train[["Object"]]
    
    x=x.iloc[:,0]
    y=y.iloc[:,:]


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state =1, shuffle = True)

    X_train_tfidf, count_vect, tfidf_transformer = preprocess_text(x_train)
    clf= SVC(random_state = 0)
    clf.fit(X_train_tfidf, y_train.values)
    acc = clf.score(X_train_tfidf, y_train.values)


    x_test_tfidf = count_vectorizer(x_test, count_vect, tfidf_transformer)
    y_pred_obj = clf.predict(x_test_tfidf)
 
    

    return creport(y_test, y_pred_obj), clf
   
   
def save_obj_model(clf):
    filename_clf = 'SVM_object_model_version2.sav'
    pickle.dump(clf, open(filename_clf, 'wb'))
   
   
   





def train_agent_detection_model(df_train):
    # Train the agent detection model with a split 80%, 20% and observe. 
    
    #df_train["Agent"] = df_train["Agent"].str.replace("http://linked.aub.edu.lb/actionrec/Agent/", "")

    x = df_train[["annotation"]]
    y = df_train[["Agent"]]
    
    x=x.iloc[:,0]
    y=y.iloc[:,:]


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state =1, shuffle = True)

    X_train_tfidf, count_vect, tfidf_transformer = preprocess_text(x_train)
    clf= SVC(random_state = 0)
    clf.fit(X_train_tfidf, y_train.values)
    acc = clf.score(X_train_tfidf, y_train.values)
    #print("Accuracy for agent detection model on training set (80% of total dataset) is :", acc)

    x_test_tfidf = count_vectorizer(x_test, count_vect, tfidf_transformer)
    y_pred_agent = clf.predict(x_test_tfidf)

    #filename_clf_agent = 'SVM_agent_model_version2.sav'
    #pickle.dump(clf, open(filename_clf_agent, 'wb'))

    return creport(y_test, y_pred_agent), clf
   
   
   
def save_agent_model(clf):
    filename_clf = 'SVM_agent_model_version2.sav'
    pickle.dump(clf, open(filename_clf, 'wb'))
   
   
   
def get_train_data_mysql():
   

    host="linked.aub.edu.lb"
    port=3306
    database ="reviews_actions_ml"


    
    configs = Properties()
    
    with open('dbconfig.properties', 'rb') as config_file:
         configs.load(config_file)
           
    dbConnection = mysql.connector.connect(user=configs.get("db.username").data, password=configs.get("db.password").data, host="linked.aub.edu.lb", database="reviews_actions_ml")



    checked_data = pd.read_sql_query("SELECT * FROM CheckedAnnotation", dbConnection)
    checked_data["ActionProbability"] = checked_data["ActionProbability"].astype(float)
      
    dbConnection.commit()
   


    return  checked_data



  
  
  
def main():
    st.markdown('<div class="header"> <H1 align="center"><font style="style=color:lightblue; "> The Machine Learning Models Page</font></H1></div>', unsafe_allow_html=True)
    model_data = get_model_info_mysql()
    df_train = get_train_data_mysql()
    with st.expander("View ML Models Information"):
        st.write(model_data)
      
    
    df_flag_report, model = train_model_action_flag(df_train)
    st.write("Action Flag Model Retrained")
    with st.expander("View report"):
         st.write(df_flag_report)
         save1= st.button("Save new action/noaction model")
         if save1:
            save_action_noaction_model(model)
            
         
    df_env_report, model = train_environment_detection_model(df_train)
    st.write("Environement Model Retrained")  
    with st.expander("View Report"):
         st.write(df_env_report)
         save2= st.button("Save new Environment model")
         if save2:
            save_env_model(model)
             
             
    df_agent_report, model = train_agent_detection_model(df_train)
    st.write("Agent Model Retrained")  
    with st.expander("View Report"):
         st.write(df_agent_report)
         save3= st.button("Save new Agent model")
         if save3:
            save_agent_model(model)
             
             
    df_valence_report, model = train_valence_detection_model(df_train)
    st.write("Valence Model Retrained")  
    with st.expander("View Report"):
         st.write(df_valence_report)
         save4= st.button("Save new Valence model")
         if save4:
            save_valence_model(model)
              
              
    df_object_report, model = train_object_detection_model(df_train)
    st.write("Object Model Retrained")  
    with st.expander("View Report"):
         st.write(df_object_report)
         save5= st.button("Save new Object model")
         if save5:
            save_valence_model(model)
  
     
  
if __name__ == "__main__":
    
    main()
  
  
  
  
  
