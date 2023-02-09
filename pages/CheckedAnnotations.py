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
from SPARQLWrapper import SPARQLWrapper, BASIC





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


def construct_graph(df_tuples, index, annotation_md5):
      
       tripletString = " <<{0}>> <<{1}>> {2}".format( df_tuples["Subject"][index], df_tuples["Predicate"][index], df_tuples["Object"][index])
       queryString =  "INSERT DATA {{{0}}}".format(tripletString) 
       #insert_to_sparql(queryString)
         
       return str(queryString)
       

   
   
def computeMD5hash(my_string):
    m = hashlib.md5()
    m.update(my_string.encode('utf-8'))
    return m.hexdigest()
    
    
    
def create_triplets(df, df_review, df_product, i):
    
    reviewBody = str(df.iloc[i]["reviewBody"])
    df_review = df_review[df_review['reviewBody'] == reviewBody]
    #st.write(df_review)
    product_name = df_review["product_name"].unique()
    str_product = product_name[0]
    #st.write(str_product)
    #df_review['product_name_md5'] = computeMD5hash(df_review['product_name'][i])
      
    df_product = df_product[df_product['product_name'] == str(str_product)]
    #st.write(df_product)
   

    dct= "http://purl.org/dc/terms/"
    rdfs= "http://www.w3.org/2000/01/rdf-schema#"
    #oa= "http://www.w3.org/ns/oa#"
    oa = "http://linked.aub.edu.lb/actionrec/Annotation/"
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
      

    if df.iloc[i]["Actions"] == "No_ActionAction":
        st.write(df.iloc[i]["Actions"])
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
                 schema + str(df_product["product_name_md5"][0]),
                 schema + str(df["Actions"][i]),
                 schema + str(df_product["product_name_md5"][0]),
                 schema + str(df_product["product_name_md5"][0]),
                 schema + str(df_review.iloc[0]['reviewBody_md5']),
                 schema + str(df_review.iloc[0]['reviewBody_md5']),
                 schema + str(df_product["availability"][0]),
                 schema + str(df_product["availability"][0]),
                 oa + str(df['annotation_md5'][i]),
                 oa + str(df['annotation_md5'][i]),
                 schema + str(df_review.iloc[0]['reviewBody_md5']),
                 schema + str(df_review.iloc[0]['reviewBody_md5']),
                 schema + str(df_review.iloc[0]['reviewBody_md5']),
                 schema + str(df_product["availability"][0]),
                 schema + str(df_product["price"][0]),
                 schema + str(df_product["priceCurrency"][0]),
                 schema + str(df_product["product_name_md5"][0]),
                 schema + str(df_product["product_name_md5"][0]),
                 schema + str(df_product["product_name_md5"][0]),
                 schema + str(df_product["product_name_md5"][0]),
                 schema + str(df_product["product_name_md5"][0]),
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


       list_objects = ['<' + oa + str(df['annotation_md5'][i]) + '>',
                '<' + oa + str(df['annotation_md5'][i])+ '>',
                '<' + oa + str(df['annotation_md5'][i])+ '>',
                '<' + oa + str(df['annotation_md5'][i])+ '>',
                '<' + oa + str(df['annotation_md5'][i])+ '>',
                '<' + arec + str(df['Ability'][i])+ '>',
                '<' + schema + str(df['Actions'][i])+ '>',
                '<' + arec + str(df['Agent'][i])+ '>',
                '<' + schema + str(df['Environment'][i])+ '>',
                '<' + schema + str(df['Object'][i])+ '>',
                '<' + schema + str(df_review.iloc[0]['reviewBody_md5'])+ '>',
                '<' + schema + str(df['Actions'][i])+ '>',
                "'" + str(df['Valence'][i])+ "'" + '^^'+'<{0}string>'.format(xsd),
                "'" +str(df['reviewBody'][i])+"'"  + '^^'+'<{0}string>'.format(xsd),
                '<' + oa + str(df['annotation_md5'][i])+ '>',
                '<' + schema + str(df["Object"][i])+ '>',
                '<' + schema + str(df['Actions'][i])+ '>',
                '<' + arec + str(df['Features'][i])+ '>',
                '<' + schema + str(df_product["product_name_md5"][0])+ '>',
                "'" +str(df_product["ratingValue"][0])+ "'" + '^^'+'<{0}decimal>'.format(xsd),
                '<' + schema + str(df_product["seller_name"][0])+ '>',
                '<' + schema + str(df_product["product_name_md5"][0])+ '>',
                "'" +str(df["checkedTimestamp"][i])+ "'" + '^^'+'<{0}dateTime>'.format(xsd),
                "'" +str(df['annotation'][i])+ "'" + '^^'+'<{0}string>'.format(xsd),
                "'" +str(df_product["product_name"][0])+ "'" + '^^'+'<{0}string>'.format(xsd),
                "'" +str(df_product["product_name"][0])+ "'" + '^^'+'<{0}string>'.format(xsd),
                '<' + schema + str(df_product["seller_name"][0])+ '>',
                '<' + str(df_product["availability"][0])+ '>',
                "'" +str(df_product["price"][0])+ "'" + '^^'+'<{0}decimal>'.format(xsd),
                "'" +str(df_product["priceCurrency"][0])+ "'" + '^^'+'<{0}string>'.format(xsd),
                "'" +str(df_product["model"][0])+ "'" + '^^'+'<{0}string>'.format(xsd),
                "'" +str(df_product["product_name"][0])+ "'" + '^^'+'<{0}string>'.format(xsd),
                "'" +str(df_product["product_name"][0])+ "'" + '^^'+'<{0}string>'.format(xsd),
                "'" +str(df_product["description"][0])+ "'" + '^^'+'<{0}string>'.format(xsd),
                "'" +str(df_product["brand_name"][0])+ "'" + '^^'+'<{0}string>'.format(xsd),
                #str(df_product.iloc[0]["url"]) + '^^'+'<<{0}string>>'.format(xsd),
                #str(df_product.iloc[0]["image"]) + '^^'+'<<{0}string>>'.format(xsd),
               ]    
       
       
           

       df_tuples = pd.DataFrame(columns={"Subject", "Predicate", "Object"})

       df_tuples["Subject"] = list_subjects
       df_tuples["Predicate"] = list_predicates
       df_tuples["Object"] = list_objects
       #st.write(df_tuples)
       #for i in df_tuples.index:
           #queryString = construct_graph(df_tuples, i, df['annotation_md5'][i])
       #st.write(df_tuples)
       
       ssl._create_default_https_context = ssl._create_unverified_context
           #for index in df_tuples.index:
       #queryString1 = "INSERT DATA { <http://schema.org/{{0}}> <http://purl.org/dc/terms/isPartOf> <http://linked.aub.edu.lb/actionrec/Annotation/{{1}}> }".format(str(df['Actions'][i]),str(df['annotation_md5'][i])) 
       #queryString = "INSERT DATA {{ <<{0}>> <<{1}>> {2} }}".format( str(df_tuples.iloc[0]["Subject"]), str(df_tuples.iloc[0]["Predicate"]), str(df_tuples.iloc[0]["Object"]))
       #queryString = "INSERT DATA {" + str(" <<") + str(df_tuples.iloc[0]["Subject"]) + str(">>") +  str(" <<") + str(df_tuples.iloc[0]["Predicate"]) + str(">> ") + str(df_tuples.iloc[0]["Object"]) + "}"
       #queryString = "INSERT DATA { <http://schema.org/LearnAction> <http://purl.org/dc/terms/isPartOf> <http://linked.aub.edu.lb/actionrec/Annotation/b6a5da3c79c2f579c35f52ad663ef049>}"
       for row in df_tuples.index:
      
           queryString = "INSERT DATA {" + str(" <") + str(df_tuples.iloc[row]["Subject"]) + str(">") +  str(" <") + str(df_tuples.iloc[row]["Predicate"]) + str("> ") + str(df_tuples.iloc[row]["Object"]) + "}"
    
           #st.write(queryString)
       
           sparql = SPARQLWrapper(
             "https://linked.aub.edu.lb:8080/fuseki/actionrec_ml/update"
              )

           sparql.setQuery(queryString)
                             
           sparql.setMethod('POST')
           sparql.query()
            
       st.write("Successfully inserted into triple store.")
       
           #insert_to_sparql(df_tuples, df['annotation_md5'][i])
       
    
   
def add_txtForm():
    st.session_state.col1 += (st.session_state.input_col1 + '  \n')
    st.session_state.col2 += (st.session_state.input_col2 + '  \n')
    st.session_state.col3 += (st.session_state.input_col3 + '  \n')
    st.session_state.col4 += (st.session_state.input_col4 + '  \n')
   
   
   
   
def display_reviews(checked_data):

         for j in checked_data.index:
                 col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12 = st.columns(12)
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
                 #with col13:
                  #KG = (st.button("Construct KG" ,key= checked_data["checked_annotation_id"][j]))
                  #if KG:
                     #create_triplets(checked_data, review_data, product_data, j)
                     
         return checked_data
   
    
    
def main():
  
     st.markdown('<div class="header"> <H1 align="center"><font style="style=color:lightblue; ">Checked Annotations Page</font></H1></div>', unsafe_allow_html=True)
     checked_annotation_data, review_data, product_data = get_new_reviews_mysql()
     col1, col2, col3 = st.columns(3)

     checked_by = st.multiselect("Filter Checked data by annotators:", ["","Bma52", "Fz13", "Wk14"])
     if checked_by:
        checked_data = checked_annotation_data[checked_annotation_data["checkedBy"].isin(checked_by)]
    
        reviews = checked_data["annotation"].unique()
        checked_annotation_data, review_data, product_data = get_new_reviews_mysql()
        for review in reviews:
           st.write(review)
           
           checked_data = checked_annotation_data[checked_annotation_data["checkedBy"].isin(checked_by)]
           checked_data = checked_data[checked_data["annotation"] == str(review)]
           
           with st.expander("View Checked Annotation"):
              checked_data = display_reviews(checked_data)

              #txtForm = st.form(key=review)
              #with txtForm:
              txtColumns = st.columns(6)
              with txtColumns[0]:
                   action = st.text_input('Action', key="{0} 1".format(review))
              with txtColumns[1]:
                   agent = st.text_input('Agent', key="{0} 2".format(review))
              with txtColumns[2]:
                   env = st.text_input('Environment', key="{0} 3".format(review))
              with txtColumns[3]:
                   valence = st.text_input('Valence', key="{0} 4".format(review))
              with txtColumns[4]:
                   feature = st.text_input('Feature', key="{0} 5".format(review))
              with txtColumns[5]:
                   obj = st.text_input('Object', key="{0} 6".format(review))
              checked_data["Actions"][0] = action
              checked_data["Agent"][0] = agent
              checked_data["Environment"][0] = env
              checked_data["Valence"][0] = valence
              checked_data["Features"][0] = feature
              checked_data["Object"][0] = obj
              
              if st.button("Construct KG", key= review):
                 create_triplets(checked_data, review_data, product_data, 0)
              
              
          

       #on_click=create_triplets(checked_data, review_data, product_data, j)
       # for i in checked_data.index:
      
            #create_triplets(checked_data, review_data, product_data, i)


         
if __name__ == "__main__":
    
    main()

    
