  
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



st.markdown('<div class="header"> <H1 align="center"><font style="style=color:lightblue; "> The Admin Page</font></H1></div>', unsafe_allow_html=True)

chart = functools.partial(st.plotly_chart, use_container_width=True)


def fetch_reviews(product_url):
    
    payload="{\"url\":\"https://www.bestbuy.com/site/reviews/microsoft-surface-laptop-studio-14-4-touch-screen-intel-core-i7-16gb-memory-nvidia-geforce-rtx-3050-ti-512gb-ssd-platinum/6478302?variant=A\"}"

  
    server = "http://linked.aub.edu.lb:8585/ml"
    payload="{{\"url\":\"{0}\"}}".format(product_url)


    headers = {

      'Authorization': 'Basic dXNyOnBhc3M=',

      'Content-Type': 'application/json',

      'Cookie': 'session_id=8460d79fca0ceebde7b83960570e1db3ca181c18'

     }


    response = requests.request("POST", server, headers=headers, data=payload)
    data = response.json()

    df = pd.DataFrame.from_dict(data, orient="index")

    data_product = df.iloc[1,:]


    products = []
    reviews = []

    for i in data_product:
        if i != None:
            for key, value in i.items():
               if i["@type"] == "Product":
                   products.append(i)
                   break;
               else:
                   if i["@type"] == "Review":
                       reviews.append(i)
                       break;
    
    df_product = pd.DataFrame(products)
    df_reviews = pd.DataFrame(reviews)


    #st.write("Number of products in this page is {}".format(len(products)))

    #st.write("Number of reviews in this page is {}".format(len(reviews)))

    df_product = df_product.rename(columns = {'name': 'product_name', '@type': 'type', '@context': 'context'})
    df_product['brand'].apply(pd.Series)
    df_product["aggregateRating"].apply(pd.Series)
    df_product["offers"].apply(pd.Series)


    df_product = pd.concat([df_product, df_product["brand"].apply(pd.Series)], axis=1)
    df_product = df_product.rename(columns = {'name': 'brand_name'})
    df_product = pd.concat([df_product, df_product["aggregateRating"].apply(pd.Series)], axis=1)
    df_product = pd.concat([df_product, df_product["offers"].apply(pd.Series)], axis=1)
    df_product = pd.concat([df_product, df_product["seller"].apply(pd.Series)], axis=1)
    df_product = df_product.rename(columns = {'name': 'seller_name'})



    del df_product['@type']
    del df_product['brand']
    del df_product['aggregateRating']
    del df_product['offers']
    del df_product['seller']


    df_reviews = df_reviews.rename(columns = {'name': 'review_name', '@type': 'type', '@context': 'context'})

    df_reviews = pd.concat([df_reviews, df_reviews["itemReviewed"].apply(pd.Series)], axis=1)
    df_reviews = df_reviews.rename(columns = {'name': 'product_name'})

    df_reviews = pd.concat([df_reviews, df_reviews["author"].apply(pd.Series)], axis=1)
    df_reviews = df_reviews.rename(columns = {'name': 'author_name'})

    df_reviews = pd.concat([df_reviews, df_reviews["reviewRating"].apply(pd.Series)], axis=1)

    df_reviews = pd.concat([df_reviews, df_reviews["publisher"].apply(pd.Series)], axis=1)
    df_reviews = df_reviews.rename(columns = {'name': 'publisher_name'})

    del df_reviews['@type']
    del df_reviews['itemReviewed']
    del df_reviews['author']
    del df_reviews['reviewRating']
    del df_reviews['publisher']

    return df_product, df_reviews



# Getting the training dataset from Sparql DBMS 

def sparql_query() -> pd.DataFrame:
    ssl._create_default_https_context = ssl._create_unverified_context
    sparql = SPARQLWrapper(
    "https://linked.aub.edu.lb:8080/fuseki/actionrec_ml/"
    )

    sparql.setQuery("""
    PREFIX dct: <http://purl.org/dc/terms/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX oa: <http://www.w3.org/ns/oa#>  
    PREFIX dcterms: <http://purl.org/dc/terms/>  
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>  
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#type/>  
    PREFIX schema: <http://schema.org/>  
    PREFIX arec: <http://linked.aub.edu.lb/actionrec/> 
    
 
    SELECT ?productName ?product ?action ?agent ?environment ?feature ?object ?reviewBody ?rating ?valence  ?annotationDescription ?annotationLabel ?annotation ?AnnotationCreationDate
    WHERE {   
    ?product schema:potentialAction ?action .  
    ?product dcterms:isPartOf ?annotation .  
    ?product schema:name ?productName .  
    ?action dcterms:isPartOf ?annotation .  
    ?agent dcterms:isPartOf ?annotation . 
    ?environment dcterms:isPartOf ?annotation .  
    ?feature dcterms:isPartOf ?annotation .  
    ?object  dcterms:isPartOf ?annotation .
    ?action schema:agent ?agent . 
    ?action schema:object ?object . 
    ?action schema:location ?environment . 
    ?product arec:hasFeature ?feature . 
    ?annotation oa:hasTarget ?review . 
    ?review schema:reviewBody ?reviewBody . 
    ?review schema:reviewRating ?review_rating . 
    ?review_rating schema:ratingValue ?rating .
    ?annotation rdf:subClassOf oa:Annotation . 
    ?annotation arec:hasValence ?valence .
    ?annotation schema:description ?annotationDescription .
    ?annotation rdfs:label ?annotationLabel .
    ?annotation dct:created ?AnnotationCreationDate
   }  
   ORDER BY ?productName
       """
   )

    
    sparql.setOnlyConneg(True)
    sparql.setReturnFormat(JSON)


    ret = sparql.query().convert()
    list_name = []
    list_product=[]
    list_action = []
    list_agent = []
    list_object = []
    list_environment = []
    list_feature = []
    list_reviewBody = []
    list_rating = []
    list_valence = []
    list_annotation_desc = []
    list_annotation_label  = []
    list_annotation = []
    list_annotation_date = []

    for item in ret["results"]["bindings"]:
         list_name.append(item['productName']["value"])
         list_product.append(item['product']["value"])
         list_action.append(item['action']["value"])
         list_object.append(item['object']["value"])
         list_agent.append(item['agent']["value"])
         list_environment.append(item["environment"]["value"])
         list_feature.append(item["feature"]["value"])
         list_reviewBody.append(item["reviewBody"]["value"])
         list_rating.append(item["rating"]["value"])
         list_valence.append(item["valence"]["value"])
         list_annotation_desc.append(item["annotationDescription"]["value"])
         list_annotation_label.append(item["annotationLabel"]["value"])
         list_annotation.append(item["annotation"]["value"])
         list_annotation_date.append(item["AnnotationCreationDate"]["value"])

    df_train = pd.DataFrame(list(zip(list_name, list_product,
        list_action, list_agent, list_environment, list_feature, list_object, list_reviewBody, 
        list_rating, list_valence, list_annotation_desc, list_annotation_label, list_annotation, list_annotation_date)), columns = ["Product Name", "Product id", "Action", "Agent", 
        "Environment", "Feature", "Object", "Review Body", "Rating", "Valence", "Annotation Description", "Annotation Label", "Annotation", "Annotation Date"]) 
    return df_train



    
    
def get_new_reviews_mysql():
   

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
   


    
    

def insert_to_sparql(df_tuples, annotation_md5):

    tripletsString_concat = " "
    for index in df_tuples.index:
          tripletString = "<{0}> <{1}> <{2}> .".format(df_tuples["Subject"][index], 
                                      df_tuples["Predicate"][index], df_tuples["Object"][index])
        
          tripletsString_concat = tripletsString_concat + tripletString
    queryString = "INSERT DATA { GRAPH <{0}> {".format(annotation_md5) + tripletsString_concat + " }"
    
   
    

        
    st.write(queryString)
    ssl._create_default_https_context = ssl._create_unverified_context
    sparql = SPARQLWrapper(
          "https://linked.aub.edu.lb:8080/fuseki/actionrec_ml/update"
        )

    sparql.setQuery(queryString) 
    sparql.method = 'POST'
    sparql.query()
    st.write("Successfully inserted into triple store.")


    





def predict_informative_annotations(df_reviews):
    df_reviews = df_reviews.reset_index()
    df_reviews = df_reviews.rename(columns={"index":"Review id"})
    df_reviews['Review id'] = df_reviews.index + 1000

    
    nltk.download('punkt')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    review_sentences = []
    for review in df_reviews["reviewBody"]:
        review_sent = tokenizer.tokenize(str(review))
        review_sentences.append(review_sent)

    df_reviews["review sentences"] = review_sentences

    df_splitted_review = df_reviews["review sentences"].apply(pd.Series).stack().rename("review sentences").reset_index()

    df_final = pd.merge(df_splitted_review,df_reviews,left_on='level_0',right_index=True, suffixes=(['','_old']))[df_reviews.columns]
    x = df_final["review sentences"]
    count_vect, tfidf_transformer = train_model_action_flag()

    result_action_flag, proba_action = action_no_action_model(x,count_vect, tfidf_transformer)
    df_final["ActionFlag"] = result_action_flag

    probas = []
    for i in proba_action[:,0]:
        proba = round(i, 2) * 100
        probas.append(proba)
    
    df_final["ActionProbability"] = probas  
    
    #st.write("Number of informative annotations that might refers to an actions is {0}, out of {1}".format(df_final['Action Flag'].value_counts()['Action Exist'], df_final.shape[0]))

    return df_final





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
def train_model_action_flag() -> object:

    df = pd.read_csv("Action classified dataset.csv")
    x = df[["review sentences"]]
    y=df[["Action Flag"]]

    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state =1, shuffle = True)

    x=x.iloc[:,0]
    y=y.iloc[:,:]
    #X=x.to_dict()
    X=list(x)  

    count_vect=CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    X_train_counts=count_vect.fit_transform(X)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf= X_train_tfidf.toarray()

    clf= SVC(random_state = 0,  probability=True)
    clf.fit(X_train_tfidf, y.values)
    clf.score(X_train_tfidf, y.values)
    

    filename_svm = 'SVM_action_noaction_model.sav'
    pickle.dump(clf, open(filename_svm, 'wb'))

    return count_vect, tfidf_transformer



def train_environment_detection_model(df_train):
    df_train["Environment"] = df_train["Environment"].str.replace("http://linked.aub.edu.lb/actionrec/Environment/", "")

    x = df_train[["Review Body"]]
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
    
    
    
    filename_clf = 'SVM_environment_model_2.sav'
    pickle.dump(clf, open(filename_clf, 'wb'))
    return creport(y_test, y_pred_env)


def train_valence_detection_model(df_train):
    x = df_train[["Review Body"]]
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

    
    filename_LR = 'LR_valence_model_2.sav'
    pickle.dump(LR, open(filename_LR, 'wb'))
    return creport(y_test, y_pred_val)



def train_object_detection_model(df_train):
    x = df_train[["Review Body"]]
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
 
    
    filename_clf = 'SVM_object_model_2.sav'
    pickle.dump(clf, open(filename_clf, 'wb'))
    return creport(y_test, y_pred_obj)




def train_agent_detection_model(df_train):
    # Train the agent detection model with a split 80%, 20% and observe. 
    
    df_train["Agent"] = df_train["Agent"].str.replace("http://linked.aub.edu.lb/actionrec/Agent/", "")

    x = df_train[["Review Body"]]
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
    y_pred_env = clf.predict(x_test_tfidf)

    filename_clf = 'SVM_agent_model_2.sav'
    pickle.dump(clf, open(filename_clf, 'wb'))

    return count_vect, tfidf_transformer




def action_no_action_model(x, count_vect, tfidf_transformer ):
     filename_svm = 'SVM_action_noaction_model.sav'
     annotations_tfidf = count_vectorizer(x, count_vect, tfidf_transformer)
     loaded_model_svm = pickle.load(open(filename_svm, 'rb'))
     result_svm = loaded_model_svm.predict(annotations_tfidf)
     proba_svm = loaded_model_svm.predict_proba(annotations_tfidf)

     return result_svm, proba_svm




def predict_action(df):

    df = df[df["ActionFlag"] == "Action Exist"]
    x = df["review sentences"]
    filename_multi_action = 'multi_label_action_model.sav'
    loaded_model_multi_action = pickle.load(open(filename_multi_action, 'rb'))
    result_multi_action = loaded_model_multi_action.predict(x)
    
    clf = preprocessing.LabelBinarizer()
    clf.fit(result_multi_action)
    pred_array = clf.transform(result_multi_action)

    actions = ['Action_AnimateAction', 'Action_BackupAction', 'Action_BendAction',
       'Action_BrowseAction', 'Action_CallAction', 'Action_CarryAction',
       'Action_ChargeAction', 'Action_ClickAction', 'Action_ConnectAction',
       'Action_CrackAction', 'Action_CreateAction', 'Action_DesignAction',
       'Action_DevelopAction', 'Action_DisableAction', 'Action_DownloadAction',
       'Action_DrawAction', 'Action_EditAction', 'Action_ExerciseAction',
       'Action_FilmAction', 'Action_FoldAction', 'Action_HearAction',
       'Action_HoldAction', 'Action_InstallAction', 'Action_IntegrateAction',
       'Action_InteractAction', 'Action_ListenAction', 'Action_LoadAction',
       'Action_MeetAction', 'Action_MultitaskAction', 'Action_NavigateAction',
       'Action_PlayAction', 'Action_ProcessAction', 'Action_ReadAction',
       'Action_RecordAction', 'Action_ResizeAction', 'Action_RunAction',
       'Action_ScanAction', 'Action_ScrollAction', 'Action_SelectAction',
       'Action_SellAction', 'Action_SetupAction', 'Action_ShutDownAction',
       'Action_SignAction', 'Action_StartAction', 'Action_StoreAction',
       'Action_StreamAction', 'Action_StudyAction', 'Action_TeachAction',
       'Action_TextAction', 'Action_TouchAction', 'Action_TransferAction',
       'Action_TypeAction', 'Action_UnlockAction', 'Action_UpgradeAction',
       'Action_ViewAction', 'Action_WatchAction', 'Action_WorkAction',
       'Action_WriteAction']
    id_action = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
       53, 54, 55, 56, 57]


    dict_actions = dict(zip(id_action, actions))
    pred_actions = []
    for i in result_multi_action:
        for j in dict_actions:
           if i == j:
             pred_actions.append(dict_actions[j])

    df_pred_actions = pd.DataFrame(pred_actions, columns = {"Actions"})
    
    df_review_with_action = pd.concat([df.reset_index(drop=True), df_pred_actions.reset_index(drop = True)], axis=1)

    st.write("Action Predicted")
    return df_review_with_action


def predict_agent(df_reviews, count_vect, tfidf_transformer):
    #Agent Detection 
    reviews = df_reviews[["reviewBody"]]
    reviews=reviews.iloc[:,0]
    #count_vect, tfidf_transformer = train_agent_detection_model()
    reviews_tfidf = count_vectorizer(reviews, count_vect, tfidf_transformer)
    filename_clf =  'SVM_agent_model_2.sav'
    loaded_agent_detection_model = pickle.load(open(filename_clf, 'rb'))
    agent = loaded_agent_detection_model.predict(reviews_tfidf)

    df_reviews["Agent"] = agent
    st.write("Agent Predicted ")
    return df_reviews




def predict_environment(df_reviews, count_vect, tfidf_transformer):
    reviews = df_reviews[["reviewBody"]]
    reviews=reviews.iloc[:,0]

    reviews_tfidf = count_vectorizer(reviews, count_vect, tfidf_transformer)
    filename_env = 'SVM_environment_model_2.sav'
    loaded_env_detection_model = pickle.load(open(filename_env, 'rb'))
    environment = loaded_env_detection_model.predict(reviews_tfidf)

    df_reviews["Environment"] = environment
    st.write("Environment Predicted ")
    return df_reviews




def predict_valence(df_reviews, count_vect, tfidf_transformer):
    reviews = df_reviews[["reviewBody"]]
    reviews=reviews.iloc[:,0]
    reviews_tfidf = count_vectorizer(reviews, count_vect, tfidf_transformer)
    filename_LR = 'LR_valence_model_2.sav'
    loaded_valence_detection_model = pickle.load(open(filename_LR, 'rb'))
    valence = loaded_valence_detection_model.predict(reviews_tfidf)

    df_reviews["Valence"] = valence
    st.write("Valence Predicted")
    return df_reviews



def predict_object(df_reviews, count_vect, tfidf_transformer):
    reviews = df_reviews[["reviewBody"]]
    reviews=reviews.iloc[:,0]
    reviews_tfidf = count_vectorizer(reviews, count_vect, tfidf_transformer)
    filename_obj = 'SVM_object_model_2.sav'
    loaded_obj_detection_model = pickle.load(open(filename_obj, 'rb'))
    obj = loaded_obj_detection_model.predict(reviews_tfidf)
    
    df_reviews["Object"] = obj
    st.write("Object Predicted")
    return df_reviews





def convert_review(string):
    return (string.split())




def feature_extraction(df):
    df = df[df["ActionFlag"] == "Action Exist"]
    review_lists = []
    for i in df["review sentences"]:
        review_splitted = convert_review(i)
        review_lists.append(review_splitted)
  
    df["review_into_words"] = review_lists

    ps = PorterStemmer()


    review_stemmed = []
    for review in df["review_into_words"]:
        stemmed_words = []
        for word in review:
            w = ps.stem(word)
            stemmed_words.append(w)
        review_stemmed.append(stemmed_words)

    df["review_words_stemmed"] = review_stemmed

    feature_list = ['screenresolution', 'graphicscard', 'performance', 'fps', 'screenquality', 'processingpower', 'lightweight', 'screenrefreshrate',
     'cpu', 'speed', 'batterylife', 'fans', 'discdrive', 'pairxboxcontroller', 'camera', 'screensize', 'applepencil', 'logitechpencil', 
     'splitscreen', 'wirelesskeyboard', 'size', 'programs', 'attachablekeyboard', 'multitask', 'bigsur', 'memory', 'speakers', 'operatingsystem', 
     'ssd', 'wordprocessing', 'bluetooth', 'keyboard', 'tabletfunction', 'touchscreen', 'microphone', 'processor', 'powersettings', 'hinges',
      'wirelessconnection', 'touchpad', 'wifi', 'ports', 'battery', 'googleplaystore', 'stylus', 'harddrive', 'sdcardslot', 'trackpad', 'screen', 
      'screenbrightness', 'cddrive', 'cables', 'powerbutton', 'privacymode', 'igpu', 'fingerprintscanner', 'supportassistant', 'camerabutton', 
      'microphonebutton', 'gpu', 'charger', 'weight', 'tr', 'internaldrive', 'microsoft', 'fan', 'numerickeypad', 'cortana', 'sleepmode', 
      'openbox', 'surfaceslimpen', 'windowshello', 'spen']

    feature_list_lower = []
    for i in feature_list:
        i = i.lower()
        feature_list_lower.append(i)
    

    # Features Extraction 


    features_extracted =[]
    scores_extracted = []

    for review in df["review_words_stemmed"]:
        features = []
        scores = []
        for word in review:
            for key in feature_list_lower:
               try:
                  res = cosdis(word2vec(word), word2vec(key))
                  if res != 0:
                     features.append(key)
                     scores.append(res)
               except IndexError:
                     pass

        features_extracted.append(features)
        scores_extracted.append(scores)




  
    list_dict =[]
    for i in range(len(features_extracted)):
       dictionary = dict(zip(features_extracted[i], scores_extracted[i]))
       top_5_features = nlargest(1, dictionary, key = dictionary.get)
       list_dict.append(top_5_features)

    df["Features"] = list_dict
    st.write("Feature Extracted")

    df = df.drop(["review_into_words", "review_words_stemmed"], axis=1)
    return df








# Cosine similarity method with a threshold 80% to inquire a good batch of reasonable matching features to the review sentences. 

def word2vec(word):
    from collections import Counter
    from math import sqrt

    # count the characters in word
    cw = Counter(word)
    # precomputes a set of the different characters
    sw = set(cw)
    # precomputes the "length" of the word vector
    lw = sqrt(sum(c*c for c in cw.values()))

    # return a tuple
    return cw, sw, lw

def cosdis(v1, v2):
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance we have
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]




def computeMD5hash(my_string):
    m = hashlib.md5()
    m.update(my_string.encode('utf-8'))
    return m.hexdigest()


def insert_to_mysql(df_product, df_reviews, df_annotation):
    
    #Connect to mysql daabase 

    #host="localhost"
    #port=3306
    #user="bma52"
    #password="HB#FaZa*23271130**"
    #database="ActionRec_DB"
    #sqlEngine = create_engine("mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(user, password, host, port, database))
    host="linked.aub.edu.lb"
    port=3306
    database ="reviews_actions_ml"

    #reader = ResourceBundle.getBundle("dbconfig.properties")
    
    configs = Properties()
    
    with open('dbconfig.properties', 'rb') as config_file:
         configs.load(config_file)
            
 
    #dbConnection =  mysql.connector.connect("mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(configs.get("db.username").data,configs.get("db.password").data, host, port, database))
    dbConnection = mysql.connector.connect(user=configs.get("db.username").data, password=configs.get("db.password").data, host="linked.aub.edu.lb", database="reviews_actions_ml")
    cursor=dbConnection.cursor()

    
    
    # Product Table 
    df_product["context"] = df_product["context"].astype(str)
    df_product["type"] = df_product["type"].astype(str)
    df_product["product_name"] = df_product["product_name"].astype(str)
    df_product["url"] = df_product["image"].astype(str)
    df_product["description"] = df_product["description"].astype(str)
    df_product["image"] = df_product["image"].astype(str)
    df_product["sku"] = df_product["sku"].astype(str)
    df_product["model"] = df_product["model"].astype(str)
    df_product["brand_name"] = df_product["brand_name"].astype(str)
    df_product["ratingValue"] = df_product["ratingValue"].astype(int)
    df_product["reviewCount"] = df_product["reviewCount"].astype(int)
    df_product["priceCurrency"] = df_product["priceCurrency"].astype(str)
    df_product["price"] = df_product["price"].astype(float)
    df_product["availability"] = df_product["availability"].astype(str)
    df_product["seller_name"] = df_product["seller_name"].astype(str)

    product_md5 = []
    for i in df_product["product_name"]:
        hashed_name = computeMD5hash(i)
        product_md5.append(hashed_name)
    
    df_product['product_name_md5'] = product_md5
    
    product_cols = "`,`".join([str(i) for i in df_product.columns.tolist()])

    
    for i,row in df_product.iterrows():
        sql = "INSERT INTO `Product` (`" + product_cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cursor.execute(sql, tuple(row))
        # the connection is not autocommitted by default, so we must commit to save our changes
        dbConnection.commit()

    #frame_product = df_product.to_sql("Products", dbConnection, index = False, if_exists='append')




    #df_reviews["Review id"] = df_reviews["Review id"].astype(str)
    df_reviews["context"] = df_reviews["context"].astype(str)
    df_reviews["type"] = df_reviews["type"].astype(str)
    df_reviews["review_name"] = df_reviews["review_name"].astype(str)
    df_reviews["product_name"] = df_reviews["product_name"].astype(str)
    df_reviews["author_name"] = df_reviews["author_name"].astype(str)
    df_reviews["reviewBody"] = df_reviews["reviewBody"].astype(str)
    df_reviews["ratingValue"] = df_reviews["ratingValue"].astype(int)
    df_reviews["bestRating"] = df_reviews["bestRating"].astype(int)
    df_reviews["publisher_name"] = df_reviews["publisher_name"].astype(str)


    md5_review = []
    for i in df_reviews["reviewBody"]:
        hashed_review = computeMD5hash(i)
        md5_review.append(hashed_review)
    
    df_reviews['reviewBody_md5'] = md5_review

    #frame_review = df_reviews.to_sql("Reviews", dbConnection, index = False, if_exists='append')
    
    review_cols = "`,`".join([str(i) for i in df_reviews.columns.tolist()])
    
    for i,row in df_reviews.iterrows():
        sql = "INSERT INTO `Review` (`" +review_cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cursor.execute(sql, tuple(row))
        # the connection is not autocommitted by default, so we must commit to save our changes
        dbConnection.commit()

        
        
        
        
        
    # Annotation Table 
    df_annotation.rename(columns = {'review sentences':'annotation'}, inplace = True)
    df_annotation = df_annotation[['reviewBody','annotation', 'ActionFlag',
       'ActionProbability', 'Actions', 'Features', 'Agent', 'Environment',
       'Valence', 'Object', 'Ability']]

    md5_annotation = []
    for i in df_annotation["annotation"]:
        hashed_annotation = computeMD5hash(i)
        md5_annotation.append(hashed_annotation)
    
    df_annotation["annotation_md5"] = md5_annotation

    n_seconds = len(df_annotation["annotation"])
    # today's date in timestamp
    base = pd.Timestamp.today()
    # calculating timestamps for the next 10 days
    timestamp_list = [base + datetime.timedelta(seconds=x) for x in range(n_seconds)]
    # iterating through timestamp_list
    df_annotation["createdTimestamp"] = timestamp_list

    #df_annotation["Review id"] = df_annotation["Review id"].astype(str)
    df_annotation["reviewBody"] = df_annotation["reviewBody"].astype(str)
    df_annotation["annotation"] = df_annotation["annotation"].astype(str)
    df_annotation["ActionFlag"] = df_annotation["ActionFlag"].astype(str)
    df_annotation["ActionProbability"] = df_annotation["ActionProbability"].astype(float)
    df_annotation["Actions"] = df_annotation["Actions"].astype(str)
    df_annotation["Features"] = df_annotation["Features"].astype(str)
    df_annotation["Agent"] = df_annotation["Agent"].astype(str)
    df_annotation["Environment"] = df_annotation["Environment"].astype(str)
    df_annotation["Valence"] = df_annotation["Valence"].astype(str)
    df_annotation["Object"] = df_annotation["Object"].astype(str)
    df_annotation["Ability"] = df_annotation["Ability"].astype(str)
    df_annotation["annotation_md5"] = df_annotation["annotation_md5"].astype(str)

    #frame_annotation = df_annotation.to_sql("Annotation", dbConnection, index = False, if_exists='append')
    
    annotation_cols = "`,`".join([str(i) for i in df_annotation.columns.tolist()])
    
    for i,row in df_annotation.iterrows():
        sql = "INSERT INTO `Annotation` (`" +annotation_cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
        cursor.execute(sql, tuple(row))
        # the connection is not autocommitted by default, so we must commit to save our changes
        dbConnection.commit()
    
    
    st.write("Data is now stored in MySQL Data base management system.")




def create_triplets(df, i):




    dct= "http://purl.org/dc/terms/"
    rdfs= "http://www.w3.org/2000/01/rdf-schema#"
    oa= "http://www.w3.org/ns/oa#" 
    dcterms= "http://purl.org/dc/terms/" 
    xsd= "http://www.w3.org/2001/XMLSchema#"
    rdf= "http://www.w3.org/1999/02/22-rdf-syntax-ns#type/" 
    schema= "http://schema.org/"
    arec= "http://linked.aub.edu.lb/actionrec/"
    os = "http://www.w3.org/ns/os#"

    # We will randomly choose the index 10, and set the knowleedge graph of the 10th annotation based on the rules below. 

    # Rule 1: <schema:Action><dct:isPartOf><oa:Annotation>
    # Rule 2: <arec:Feature><dct:isPartOf><oa:Annotation>
    # Rule 3: <arec:Ability><dct:isPartOf><oa:Annotation>
    # Rule 4: <schema:Location><dct:isPartOf><oa:Annotation>
    # Rule 5: <schema:Object><dct:isPartOf><oa:Annotation>
    # Rule 6: <arec:Agent><arec:hasAbility><arec:Ability>
    # Rule 7: <arec:Ability><arec:supports><schema:Action>
    # Rule 8: <schema:Action><schema:agent><arec:Agent>
    # Rule 9: <schema:Action><schema:location><schema:Location>
    # Rule 10: <schema:Action><schema:object><schema:Object>
    # Rule 11: <oa:Annotation><os:hasTarget><schema:Review>
    # Rule 12: <arec:Feature><arec:supports><schema:Action>
    # Rule 13: <oa:Annotation><arec:hasValence><valence>
    # Rule 14: <schema:Review><schema:reviewBody><review text>
    # Rule 15: <schema:Product><dct:isPartOf><os:Annotation>
    # Rule 16: <schema:Action><schema:object><schema:Object>
    # Rule 17: <schema:Product><schema:potentialAction><schema:Action>
    # Rule 18: <schema:Product><arec:hasFeature><arec:Feature>
    # Rule 19: <schema:Review><schema:itemReviewed><schema:Product>
    # Rule 20: <schema:Review><schema:reviewRating><review_rating>
    # Rule 21: <schema:Offer><schema:offeredBy><schema:Organization>
    # Rule 22: <schema:Offer><schema:itemOffered><schema:Product>
    # Rule 23: <oa:Annotation><dct:cretaed><date/time>
    # Rule 24: <oa:Annotation><rdfs:label><Label>
    # Rule 25: <schema:Review><schema:name><name>
    # Rule 26: <schema:Review><rdfs:label><name>
    # Rule 27: <schema:Review><schema:publisher><schema:Organization>
    # Rule 28: <schema:Offer><schema:itemCondition><condition>
    # Rule 29: <schema:Offer><schema:price><price>
    # Rule 30: <schema:Offer><schema:currency><currency>
    # Rule 31: <schema:Product><schema:model><model>
    # Rule 32: <schema:Product><schema:name><name> 
    # Rule 33: <schema:Product><rdfs:label><name>
    # Rule 34: <schema:Product><schema:description><description>
    # Rule 35: <schema:Product><schema:brand><brand>
    # Rule 36: <schema:Product><schema:URL><product URL>
    # Rule 37: <schema:Product><schema:image><image URL>
    # Rule 38: <review_rating><schema:ratingValue><value>
    # Rule 39: <review_rating><schema:bestRating><value>
    # Rule 40: <review_rating><schema:worstRating><value>
    # Rule 41: <review_rating><rdfs:label><value>


    if df["Actions"][i] == "No_ActionAction":
        st.write(df["Actions"][i])
        st.error("No triplets created, this annotation has no action")
        #return None
    else:


       Subject = []
       Predicate= []
       Object = []

       list_subjects = [schema + str(df['Actions'][i]), 
                 arec + str(df['Features'][i]), 
                 arec + str(df['Ability'][i]), 
                 schema + str(df['Environment'][i]),
                 schema + str(df['Object'][i]),
                 arec + str(df['Agent'][i]),
                 arec + str(df['Ability'][i]),
                 schema + str(df['Actions'][i]),
                 schema + str(df['Actions'][i]),
                 schema + str(df['Actions'][i]),
                 oa + str(df['annotation'][i]),
                 arec + str(df['Features'][i]),
                 oa + str(df['annotation_md5'][i]),
                 schema + str(df['reviewBody_md5'][i]),
                 schema + str(df["product_name_md5"][i]),
                 schema + str(df["Actions"][i]),
                 schema + str(df["product_name_md5"][i]),
                 schema + str(df["product_name_md5"][i]),
                 schema + str(df['reviewBody_md5'][i]),
                 schema + str(df['reviewBody_md5'][i]),
                 schema + str(df["availability"][i]),
                 schema + str(df["availability"][i]),
                 oa + str(df['annotation_md5'][i]),
                 oa + str(df['annotation_md5'][i]),
                 schema + str(df['reviewBody_md5'][i]),
                 schema + str(df['reviewBody_md5'][i]),
                 schema + str(df['reviewBody_md5'][i]),
                 schema + str(df["availability"][i]),
                 schema + str(df["price"][i]),
                 schema + str(df["priceCurrency"][i]),
                 schema + str(df["product_name_md5"][i]),
                 schema + str(df["product_name_md5"][i]),
                 schema + str(df["product_name_md5"][i]),
                 schema + str(df["product_name_md5"][i]),
                 schema + str(df["product_name_md5"][i]),
                 schema + str(df["product_name_md5"][i]),
                 schema + str(df["product_name_md5"][i])
                 ]

 

       list_predicates = [dct + "isPartOf",
                   dct + "isPartOf",
                   dct + "isPartOf",
                   dct + "isPartOf",
                   dct + "isPartOf",
                   arec + "hasAbility",
                   arec + "supports",
                   schema + "agent",
                   schema + "location",
                   schema + "object",
                   os + "hasTarget",
                   arec + "supports",
                   arec + "hasValence",
                   schema + "reviewBody",
                   dct + "isPartOf",
                   schema + "object",
                   schema + "potentialAction",
                   arec + "hasFeature",
                   schema + "itemReviewed",
                   schema + "reviewRating",
                   schema + "offeredBy",
                   schema + "itemOffered",
                   dct + "created",
                   rdfs + "label",
                   schema + "name",
                   rdfs + "label",
                   schema + "publisher",
                   schema + "itemCondition",
                   schema + "price",
                   schema + "currency",
                   schema + "model",
                   schema + "name",
                   rdfs + "label",
                   schema + "description",
                   schema + "brand",
                   schema + "URL",
                   schema + "image"
                
                   ]             


       list_objects = [oa + str(df['annotation_md5'][i]),
                oa + str(df['annotation_md5'][i]),
                oa + str(df['annotation_md5'][i]),
                oa + str(df['annotation_md5'][i]),
                oa + str(df['annotation_md5'][i]),
                arec + str(df['Ability'][i]),
                schema + str(df['Actions'][i]),
                arec + str(df['Agent'][i]),
                schema + str(df['Environment'][i]),
                schema + str(df['Object'][i]),
                schema + str(df['reviewBody_md5'][i]),
                schema + str(df['Actions'][i]),
                str(df['Valence'][i]) + '^^'+'<{0}string>'.format(xsd),
                str(df['reviewBody'][i]) + '^^'+'<{0}string>'.format(xsd),
                oa + str(df['annotation_md5'][i]),
                schema + str(df["Object"][i]),
                schema + str(df['Actions'][i]),
                arec + str(df['Features'][i]),
                schema + str(df["product_name_md5"][i]),
                str(df["ratingValue"][i]) + '^^'+'<{0}decimal>'.format(xsd),
                schema + str(df["seller_name"][i]),
                schema + str(df["product_name_md5"][i]),
                str(df["created_timestamp"][i]) + '^^'+'<{0}string>'.format(xsd),
                str(df['annotation'][i]) + '^^'+'<{0}string>'.format(xsd),
                str(df["product_name"][i]) + '^^'+'<{0}string>'.format(xsd),
                str(df["product_name"][i]) + '^^'+'<{0}string>'.format(xsd),
                schema + str(df["seller_name"][i]),
                str(df["availability"][i]) + '^^'+'<{0}string>'.format(xsd),
                str(df["price"][i]) + '^^'+'<{0}decimal>'.format(xsd),
                str(df["priceCurrency"][i]) + '^^'+'<{0}string>'.format(xsd),
                str(df["model"][i]) + '^^'+'<{0}string>'.format(xsd),
                str(df["product_name"][i]) + '^^'+'<{0}string>'.format(xsd),
                str(df["product_name"][i]) + '^^'+'<{0}string>'.format(xsd),
                str(df["description"][i]) + '^^'+'<{0}string>'.format(xsd),
                str(df["brand_name"][i]) + '^^'+'<{0}string>'.format(xsd),
                str(df["url"][i]) + '^^'+'<{0}string>'.format(xsd),
                str(df["image"][i]) + '^^'+'<{0}string>'.format(xsd),
               ]    


       df_tuples = pd.DataFrame(columns={"Subject", "Predicate", "Object"})

       df_tuples["Subject"] = list_subjects
       df_tuples["Predicate"] = list_predicates
       df_tuples["Object"] = list_objects

       insert_to_sparql(df_tuples, df["annotation_md5"][i])
    
    
    




# Main function of the app

def main():
    
    product_url = st.text_area("Product URL")
    get_product_data = st.button("Get new Reviews.")
    if get_product_data: 
        df_product, df_reviews = fetch_reviews(product_url)
        df_final = predict_informative_annotations(df_reviews)

        tab1, tab2, tab3 = st.tabs(["View Product Raw Data", "View Reviews Raw Data", "View Annotations Raw Data"])
        with tab1: 
            st.write(df_product)
        with tab2:
            st.write(df_reviews)
        with tab3:
            st.write(df_final)
        

        st.markdown("""
           <style>
            div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.1);
            padding: 5% 5% 5% 10%;
            border-radius: 5px;
            color: rgb(30, 103, 119);
            overflow-wrap: break-word;
           }
         /* breakline for metric text         */
         div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
        overflow-wrap: break-word;
         white-space: break-spaces;
         color: red;
        }
        </style>
          """
          , unsafe_allow_html=True)

    
        container2 = st.container()
        col1, col2, col3 = container2.columns(3)
        col1.metric(label="Number of Products", value=df_product.shape[0])
        col2.metric(label="Number of Reviews", value=df_reviews.shape[0])     
        col3.metric(label="Number of Annotations", value = df_final.shape[0])

        st.write("Out of {0} annotations, {1} are informtaive referencing to an action.".format(df_final.shape[0], df_final[df_final["ActionFlag"] == "Action Exist"].shape[0]))
        with st.spinner('Predicting Semantics'):
             #time.sleep(50)
            df_train = sparql_query()
            count_vect, tfidf_transformer = train_agent_detection_model(df_train)
            container3 = st.container()
            col1, col2, col3 = container3.columns(3)
            with col1:
               df_final = predict_action(df_final)
               df_final = feature_extraction(df_final)
            with col2:
               df_final = predict_agent(df_final, count_vect, tfidf_transformer)
               df_final = predict_valence(df_final, count_vect, tfidf_transformer)
            with col3:
               df_final = predict_environment(df_final, count_vect, tfidf_transformer)
               df_final = predict_object(df_final, count_vect, tfidf_transformer)
            
        list_ability =[]
        for i in df_final["Actions"]:
            ability = i.replace("Action", "Ability")
            list_ability.append(ability)

        df_final["Ability"] = list_ability
        df_final["Ability"]  = df_final["Ability"] .str.replace("Ability_", "")
        df_final["Actions"] = df_final["Actions"].str.replace("Action_", "")
	
        
        with st.expander("View Final Data Set"):
            st.write(df_final)
             
        insert_to_mysql(df_product, df_reviews, df_final)
        
        st.markdown("""---""")
        st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 20px;">The Checked Annotation Section</p>', unsafe_allow_html=True)
        checked_data = get_new_reviews_mysql()
	

        checked_by = st.selectbox("Checked By at least", ["Checked by at least 1 annotator", "Checked by at least 2 annotators", "Checked by at least 3 annotators"])
        
	
        create_triplets(checked_data, 0)

            
        
        
        
        retrain_btn = st.button("Retrain ML models")
        if retrain_btn:
            report_agent = train_agent_detection_model(df_train)
            report_env = train_environment_detection_model(df_train)
            report_valence = train_valence_detection_model(df_train)
            report_obj = train_object_detection_model(df_train)
                                
        
    


if __name__ == "__main__":
    
    main()

    
