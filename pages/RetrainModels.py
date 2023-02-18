
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from github3 import login



def get_training_det():
   df = pd.read_csv("TrainingSet.csv")
   return df

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


def store_new_model_data(model_version, label, accuracy):
   
    host="linked.aub.edu.lb"
    port=3306
    database ="reviews_actions_ml"

        
    
    configs = Properties()
    
    with open('dbconfig.properties', 'rb') as config_file:
           configs.load(config_file)
            
 
    dbConnection = mysql.connector.connect(user=configs.get("db.username").data, password=configs.get("db.password").data, host="linked.aub.edu.lb", database="reviews_actions_ml")
    model_version = str(model_version)
    label = str(label)
    accuracy = float(accuracy)
    
    cursor = dbConnection.cursor()
    sql = "INSERT INTO `ML_models` (model_version, label, accuracy) VALUES ("+ model_version + "," + label + "," + str(accuracy) + ")"
   
    cursor.execute(sql)
    st.write("New Model Information is now stored in MYSQL")
             
    # the connection is not autocommitted by default, so we must commit to save our changes
    dbConnection.commit()
   
   
   
   
   
def update_to_git(model, filename_str):
    #file_info = [path]
    username = "Bma52"
    password = "HB#Fa*232711"
    account = "bma52@mail.aub.edu"
    repo = "Reviews_Actions"
    
    gh = github3.login(username=username, password=password)
    repository = gh.repository(account, repo)
    filename = filename_str
    content = pickle.dump(model, open(filename, 'wb'))
    
    #with open(file_info, 'rb') as fd:
            #contents = fd.read()
    contents_object = repository.contents(filename)
    contents_object.update("Model Updated",content)
   
   
   


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
   
   
   
   
def train_action_model(df_train):
   #df = pd.read_csv("Gummy_data_for_multi_label_action_model.csv")
   df = df_train[["annotation", "Actions"]]
   df_final = get_dummy_actions(df_train)
   
   
   x=df_final["annotation"]
   y=df_final.iloc[:,3:]
   y.astype(int)


   x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42, shuffle=True)
   pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None)))
            ])


   parameters = {
                'tfidf__max_df': (0.25, 0.5, 0.75),
                'tfidf__ngram_range': [(1, 2)],
                'tfidf__min_df': [1, 3, 5],
                'clf__estimator__alpha': (1e-2, 1e-3)
            }

   grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
   y_train = np.argmax(y_train.values, axis=1)
   grid_search_cv.fit(x_train, y_train)

   #pipeline.fit(x_train, y_train)
   predictions = grid_search_cv.predict(x_test)

   #predictions = np.argmax(predictions, axis = 1)
   #y_test = np.argmax(y_test, axis = 1)
   y_test = np.argmax(y_test.values, axis = 1)
   accuracy = accuracy_score(y_test ,predictions)
   #f1_score = f1_score(y_test, predictions, average="micro")
   #hamming_loss = hamming_loss(y_test, predictions)

   return accuracy, grid_search_cv


def save_action_model(grid_search_cv, accuracy, model_data):
   
   filename_multi_action = 'multi_label_action_model.sav'
   #pickle.dump(grid_search_cv, open(filename_multi_action, 'wb'))
   update_to_git(grid_search_cv, filename_multi_action)
   model_data_action = model_data[model_data["label"] == "Action"]
   count = len(list(model_data_action["label"]))
   model_version = "Version_" + str(count+1)
   label = "Action"
   accuracy = accuracy*100
   #store_new_model_data(model_version, label, accuracy)





   
def save_action_noaction_model(clf):
   
    filename_svm = 'SVM_action_noaction_model.sav'
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
    
   
    
    return acc, clf


def save_env_model(clf, accuracy, model_data):
    filename_clf = 'SVM_environment_model_2.sav'
    pickle.dump(clf, open(filename_clf, 'wb'))
      
    model_data_env = model_data[model_data["label"] == "Environment"]
    count = len(list(model_data_env["label"]))
    model_version = "version_" + str(count+1)
    label = "Environment"
    accuracy = accuracy*100
    store_new_model_data(model_version, label, accuracy)
      
      


def train_valence_detection_model(df_train):
    x = df_train[["annotation"]]
    y = df_train[["Valence"]]
    
    x=x.iloc[:,0]
    y=y.iloc[:,:]


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state =1, shuffle = True)

    X_train_tfidf, count_vect, tfidf_transformer = preprocess_text(x_train)
    LR= linear_model.LogisticRegression()
    LR.fit(X_train_tfidf, y_train.values)
    acc = LR.score(X_train_tfidf, y_train.values)
 


    x_test_tfidf = count_vectorizer(x_test, count_vect, tfidf_transformer)
    y_pred_val = LR.predict(x_test_tfidf)

    

    return acc, LR
   
   
   
def save_valence_model(LR, accuracy, model_data):
    filename_LR = 'LR_valence_model_2.sav'
    pickle.dump(LR, open(filename_LR, 'wb'))
   
    model_data_valence = model_data[model_data["label"] == "Valence"]
    count = len(list(model_data_valence["label"]))
    model_version = "version_" + str(count+1)
    label = "Valence"
    accuracy = accuracy*100
    store_new_model_data(model_version, label, accuracy)



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
 
    

    return acc, clf
   
   
def save_obj_model(clf, accuracy, model_data):
    filename_clf = 'SVM_object_model_2.sav'
    pickle.dump(clf, open(filename_clf, 'wb'))
   
   
    model_data_obj = model_data[model_data["label"] == "Object"]
    count = len(list(model_data_obj["label"]))
    model_version = "version_" + str(count+1)
    label = "Object"
    accuracy = accuracy*100
    store_new_model_data(model_version, label, accuracy)





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

    return acc, clf
   
   
   
def save_agent_model(clf, accuracy, model_data):
    filename_clf = 'SVM_agent_model_2.sav'
    pickle.dump(clf, open(filename_clf, 'wb'))
   
    model_data_agent = model_data[model_data["label"] == "Agent"]
    count = len(list(model_data_agent["label"]))
    model_version = "version_" + str(count+1)
    label = "Agent"
    accuracy = accuracy*100
    store_new_model_data(model_version, label, accuracy)
   
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

 

def get_dummy_actions(df_train_checked):
   
   df_actions = df_train_checked[["annotation", "Actions"]]
   encoder = OneHotEncoder(handle_unknown='ignore')
   encoder_df = pd.DataFrame(encoder.fit_transform(df_actions[['Actions']]).toarray())
   final_df = df_actions.join(encoder_df)
   final_df.drop('Actions', axis=1, inplace=True)
   
   return final_df
  
  
  
def main():
    st.markdown('<div class="header"> <H1 align="center"><font style="style=color:lightblue; "> The Machine Learning Models Page</font></H1></div>', unsafe_allow_html=True)
    model_data = get_model_info_mysql()
    df_train_checked = get_train_data_mysql()
    df_train_checked = df_train_checked[["reviewBody", "annotation", "Actions", "Agent", "Environment", "Features", "Valence", "Object", "ActionFlag"]]
    df_train_initial = get_training_det()
    df_train_initial = df_train_initial[["reviewBody", "annotation", "Actions", "Agent", "Environment", "Features", "Valence", "Object", "ActionFlag"]]
    #df_actions_new = get_dummy_actions(df_train_checked)
    df_train = df_train_initial.append(df_train_checked, ignore_index = True)
    with st.expander("View ML Models Information"):
        st.write(model_data)
      
    
    df_flag_report, model_1 = train_model_action_flag(df_train)
    st.write("Action Flag Model Retrained")
    with st.expander("View report"):
         st.write(df_flag_report)
         save1= st.button("Save new action/noaction model")
         if save1:
            save_action_noaction_model(model_1)
            
    df_action_report, model_2 = train_action_model(df_train)
    st.write("Action Model Retrained")
    with st.expander("View report"):
         st.write(df_action_report)
         save6= st.button("Save new action model")
         if save6:
            save_action_model(model_2, df_action_report, model_data)
            
         
    df_env_report, model_3 = train_environment_detection_model(df_train)
    st.write("Environement Model Retrained")  
    with st.expander("View Report"):
         st.write(df_env_report)
         save2= st.button("Save new Environment model")
         if save2:
            save_env_model(model_3, df_env_report, model_data)
             
             
    df_agent_report, model_4 = train_agent_detection_model(df_train)
    st.write("Agent Model Retrained")  
    with st.expander("View Report"):
         st.write(df_agent_report)
         save3= st.button("Save new Agent model")
         if save3:
            save_agent_model(model_4, df_agent_report, model_data)
             
             
    df_valence_report, model_5 = train_valence_detection_model(df_train)
    st.write("Valence Model Retrained")  
    with st.expander("View Report"):
         st.write(df_valence_report)
         save4= st.button("Save new Valence model")
         if save4:
            save_valence_model(model_5, df_valence_report, model_data)
              
              
    df_object_report, model_6 = train_object_detection_model(df_train)
    st.write("Object Model Retrained")  
    with st.expander("View Report"):
         st.write(df_object_report)
         save5= st.button("Save new Object model")
         if save5:
            save_obj_model(model_6, df_object_report, model_data)
  
     
  
if __name__ == "__main__":
    
    main()
  
  
  
  
  
