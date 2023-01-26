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
import re






def get_new_reviews_mysql():
   

    host="linked.aub.edu.lb"
    port=3306
    database ="reviews_actions_ml"


    
    configs = Properties()
    
    with open('dbconfig.properties', 'rb') as config_file:
         configs.load(config_file)
           
    dbConnection = mysql.connector.connect(user=configs.get("db.username").data, password=configs.get("db.password").data, host="linked.aub.edu.lb", database="reviews_actions_ml")



    checked_data = pd.read_sql_query("SELECT * FROM CheckedAnnotation", dbConnection)
    review_data = pd.read_sql_query("SELECT * FROM Review", dbConnection)
    product_data = pd.read_sql_query("SELECT * FROM Product", dbConnection)
    checked_data["ActionProbability"] = checked_data["ActionProbability"].astype(float)
      
    
    dbConnection.commit()
   
    

    return  checked_data, review_data, product_data




def insert_to_sparql(df_tuples, annotation_md5):

    #tripletsString_concat = " "
    for index in df_tuples.index:
          tripletString = "INSERT DATA {GRAPH <{0}> { <<{1}>> <<{2}>> {3} .".format(str(annotation_md5), df_tuples["Subject"][index], df_tuples["Predicate"][index], df_tuples["Object"][index])
          queryString = "INSERT DATA {{{0}}}".format(tripletString)
          #tripletsString_concat += tripletString
          st.write(queryString)
            
          ssl._create_default_https_context = ssl._create_unverified_context
          sparql = SPARQLWrapper(
          "https://linked.aub.edu.lb:8080/fuseki/actionrec_ml/update"
            )

          sparql.setQuery(queryString) 
          sparql.method = 'POST'
          sparql.query()
    st.write("Successfully inserted into triple store.")
    
    
   
    

        
    

   
   
def computeMD5hash(my_string):
    m = hashlib.md5()
    m.update(my_string.encode('utf-8'))
    return m.hexdigest()
    
    
    
def create_triplets(df, df_review, df_product, i):
   
    df_review = df_review[df_review['reviewBody'] == df['reviewBody'][i]]
    
    #df_review['product_name_md5'] = computeMD5hash(df_review['product_name'][i])
      
    df_product = df_product[df_product['product_name'] == str(df_review.iloc[0]['product_name'])]
   
   

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
    
    #cleanString = re.sub(r"[^A-Za-z]+",'', string)
    df["annotation"] = df["annotation"].str.replace(r'[^a-zA-Z0-9\s]+', '', regex=True)
    df["reviewBody"] = df["reviewBody"].str.replace(r'[^a-zA-Z0-9\s]+', '', regex=True)
    df_review["reviewBody"] = df_review["reviewBody"].str.replace(r'[^a-zA-Z0-9\s]+', '', regex=True)
    df_product["product_name"] = df_product["product_name"].str.replace(r'[^a-zA-Z0-9\s]+', '', regex=True)
    df_product["description"] = df_product["description"].str.replace(r'[^a-zA-Z0-9\s]+', '', regex=True)
    df_product["model"] = df_product["model"].str.replace(r'[^a-zA-Z0-9\s]+', '', regex=True)
    df_product["seller_name"] = df_product["seller_name"].str.replace(' ', '', regex=True)
    df["checkedTimestamp"] = pd.to_datetime(df["checkedTimestamp"])
      

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
                 oa + str(df['annotation_md5'][i]),
                 arec + str(df['Features'][i]),
                 oa + str(df['annotation_md5'][i]),
                 schema + str(df_review.iloc[0]['reviewBody_md5']),
                 schema + str(df_product.iloc[0]["product_name_md5"]),
                 schema + str(df["Actions"][i]),
                 schema + str(df_product.iloc[0]["product_name_md5"]),
                 schema + str(df_product.iloc[0]["product_name_md5"]),
                 schema + str(df_review.iloc[0]['reviewBody_md5']),
                 schema + str(df_review.iloc[0]['reviewBody_md5']),
                 schema + str(df_product.iloc[0]["availability"]),
                 schema + str(df_product.iloc[0]["availability"]),
                 oa + str(df['annotation_md5'][i]),
                 oa + str(df['annotation_md5'][i]),
                 schema + str(df_review.iloc[0]['reviewBody_md5']),
                 schema + str(df_review.iloc[0]['reviewBody_md5']),
                 schema + str(df_review.iloc[0]['reviewBody_md5']),
                 schema + str(df_product.iloc[0]["availability"]),
                 schema + str(df_product.iloc[0]["price"]),
                 schema + str(df_product.iloc[0]["priceCurrency"]),
                 schema + str(df_product.iloc[0]["product_name_md5"]),
                 schema + str(df_product.iloc[0]["product_name_md5"]),
                 schema + str(df_product.iloc[0]["product_name_md5"]),
                 schema + str(df_product.iloc[0]["product_name_md5"]),
                 schema + str(df_product.iloc[0]["product_name_md5"]),
                 #schema + str(df_product.iloc[0]["product_name_md5"]),
                 #schema + str(df_product.iloc[0]["product_name_md5"])
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
                   #schema + "URL",
                   #schema + "image"
                
                   ]             


       list_objects = ['<<' + oa + str(df['annotation_md5'][i]) + '>>',
                '<<' + oa + str(df['annotation_md5'][i])+ '>>',
                '<<' + oa + str(df['annotation_md5'][i])+ '>>',
                '<<' + oa + str(df['annotation_md5'][i])+ '>>',
                '<<' + oa + str(df['annotation_md5'][i])+ '>>',
                '<<' + arec + str(df['Ability'][i])+ '>>',
                '<<' + schema + str(df['Actions'][i])+ '>>',
                '<<' + arec + str(df['Agent'][i])+ '>>',
                '<<' + schema + str(df['Environment'][i])+ '>>',
                '<<' + schema + str(df['Object'][i])+ '>>',
                '<<' + schema + str(df_review.iloc[0]['reviewBody_md5'])+ '>>',
                '<<' + schema + str(df['Actions'][i])+ '>>',
                "'" + str(df['Valence'][i])+ "'" + '^^'+'<<{0}string>>'.format(xsd),
                "'" +str(df['reviewBody'][i])+"'"  + '^^'+'<<{0}string>>'.format(xsd),
                '<<' + oa + str(df['annotation_md5'][i])+ '>>',
                '<<' + schema + str(df["Object"][i])+ '>>',
                '<<' + schema + str(df['Actions'][i])+ '>>',
                '<<' + arec + str(df['Features'][i])+ '>>',
                '<<' + schema + str(df_product.iloc[0]["product_name_md5"])+ '>>',
                "'" +str(df_product.iloc[0]["ratingValue"])+ "'" + '^^'+'<<{0}decimal>>'.format(xsd),
                '<<' + schema + str(df_product.iloc[0]["seller_name"])+ '>>',
                '<<' + schema + str(df_product.iloc[0]["product_name_md5"])+ '>>',
                "'" +str(df["checkedTimestamp"][i])+ "'" + '^^'+'<<{0}dateTime>>'.format(xsd),
                "'" +str(df['annotation'][i])+ "'" + '^^'+'<<{0}string>>'.format(xsd),
                "'" +str(df_product.iloc[0]["product_name"])+ "'" + '^^'+'<<{0}string>>'.format(xsd),
                "'" +str(df_product.iloc[0]["product_name"])+ "'" + '^^'+'<<{0}string>>'.format(xsd),
                '<<' + schema + str(df_product.iloc[0]["seller_name"])+ '>>',
                '<<' + str(df_product.iloc[0]["availability"])+ '>>',
                "'" +str(df_product.iloc[0]["price"])+ "'" + '^^'+'<<{0}decimal>>'.format(xsd),
                "'" +str(df_product.iloc[0]["priceCurrency"])+ "'" + '^^'+'<<{0}string>>'.format(xsd),
                "'" +str(df_product.iloc[0]["model"])+ "'" + '^^'+'<<{0}string>>'.format(xsd),
                "'" +str(df_product.iloc[0]["product_name"])+ "'" + '^^'+'<<{0}string>>'.format(xsd),
                "'" +str(df_product.iloc[0]["product_name"])+ "'" + '^^'+'<<{0}string>>'.format(xsd),
                "'" +str(df_product.iloc[0]["description"])+ "'" + '^^'+'<<{0}string>>'.format(xsd),
                "'" +str(df_product.iloc[0]["brand_name"])+ "'" + '^^'+'<<{0}string>>'.format(xsd),
                #str(df_product.iloc[0]["url"]) + '^^'+'<<{0}string>>'.format(xsd),
                #str(df_product.iloc[0]["image"]) + '^^'+'<<{0}string>>'.format(xsd),
               ]    


       df_tuples = pd.DataFrame(columns={"Subject", "Predicate", "Object"})

       df_tuples["Subject"] = list_subjects
       df_tuples["Predicate"] = list_predicates
       df_tuples["Object"] = list_objects

       insert_to_sparql(df_tuples, df["annotation_md5"][i])
    
    
    
    
def main():
  
     st.markdown('<div class="header"> <H1 align="center"><font style="style=color:lightblue; ">Checked Annotations Page</font></H1></div>', unsafe_allow_html=True)
     checked_data, review_data, product_data = get_new_reviews_mysql()
     
     checked_by = st.selectbox("Checked By at least", ["Checked by at least 1 annotator", "Checked by at least 2 annotators", "Checked by at least 3 annotators"])
     
      
     with st.expander("View Checked Annotation"):
          #st.write(checked_data)
       for j in range(0,checked_data.shape[0]):
          col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = st.columns(13)
          with col1:
              st.write(checked_data["annotation"][j])
          with col2:
              st.write(checked_data["ActionFlag"][j])
          with col3:
              st.write(checked_data["ActionProbability"][j])
          with col4:
              st.write(checked_data["Actions"][j])
          with col5: 
              st.write(checked_data["Features"][j])
          with col6:
              st.write(checked_data["Agent"][j])
          with col7:
              st.write(checked_data["Environment"][j])
          with col8: 
              st.write(checked_data["Valence"][j])
          with col9:
              st.write(checked_data["Object"][j])
          with col10:
              st.write(checked_data["Ability"][j])
          with col11:
              st.write(checked_data["Ability"][j])
          with col12:
              st.write(checked_data["checkedBy"][j])
          with col13:
              (st.button("Construct KG" ,key= checked_data["checked_annotation_id"][j]))

            
          

     #on_click=create_triplets(checked_data, review_data, product_data, j)
     for i in checked_data.index:
      
          create_triplets(checked_data, review_data, product_data, i)


         
if __name__ == "__main__":
    
    main()

    
