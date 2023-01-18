

from cProfile import label
from email.policy import default
import functools
from locale import D_FMT
from os import uname

from pickletools import float8
from pyexpat import features
from sqlite3 import DatabaseError
from statistics import multimode
#from turtle import color
#from xmlrpc.client import Boolean
#from msilib import datasizemask
#from pathlib import Path

import streamlit as st
#from st_aggrid import AgGrid
#from st_aggrid.shared import JsCode
#from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd

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
import datetime
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import operator 
from heapq import nlargest
import mysql.connector
import json
import csv
from SPARQLWrapper import SPARQLWrapper, JSON
import ssl
import streamlit_authenticator as stauth
import yaml
from streamlit_authenticator import hasher 
import pymysql
import jprops
from jproperties import Properties
import sqlalchemy 

from flask import Flask



#st.markdown('<html><style>.header { width: 1000px; padding: 60px;text-align: center;background: #1abc9c;color: white;font-size: 30px;}</style></html>', unsafe_allow_html=True)




st.markdown('<div class="header"> <H1 align="center"><font style="style=color:lightblue; "> The Annotator Page</font></H1></div>', unsafe_allow_html=True)

chart = functools.partial(st.plotly_chart, use_container_width=True)




def get_new_reviews_mysql():
   

    host="linked.aub.edu.lb"
    port=3306
    database ="reviews_actions_ml"

    #reader = ResourceBundle.getBundle("dbconfig.properties")
    
    configs = Properties()
    
    with open('dbconfig.properties', 'rb') as config_file:
         configs.load(config_file)
            
 
    #dbConnection =  mysql.connector.connect("mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(configs.get("db.username").data,configs.get("db.password").data, host, port, database))
    dbConnection = mysql.connector.connect(user=configs.get("db.username").data, password=configs.get("db.password").data, host="linked.aub.edu.lb", database="reviews_actions_ml")
    #cursor=dbConnection.cursor()

    product_data = pd.read_sql_query("SELECT * FROM Product", dbConnection)
    #product_data = product_data.rename(columns = {'name': 'product_name'}, inplace=True)
    review_data = pd.read_sql_query("SELECT * FROM Review", dbConnection)
    annotation_data = pd.read_sql_query("SELECT * FROM Annotation", dbConnection)
    checked_data = pd.read_sql_query("SELECT * FROM CheckedAnnotation", dbConnection)
    annotation_data["ActionProbability"] = annotation_data["ActionProbability"].astype(float)
      
    dbConnection.commit()

    annotation_ids = list(annotation_data["annotation_id"])
    checked_annotation_ids = list(checked_data["checked_annotation_id"])
      
      
    common_ids = set(annotation_ids).intersection(checked_annotation_ids)
    if len(common_ids) != 0:
        for i in common_ids:
             annotation_ids.remove(i)
    
    
        annotation_data= pd.read_sql_query("SELECT * FROM Annotation WHERE annotation_id IN annotation_ids", dbConnection)
    
        checked_data = pd.read_sql_query("SELECT * FROM Annotation WHERE checked_annotation_id IN common_ids", dbConnection)
      
        final_annotation_data = annotation_data.append(checked_data, ignore_index=True)
         
         
    else:
      final_annotation_data = annotation_data
   
   
   
   
   
    #df_1 = pd.merge(product_data, review_data, how = 'right', on='product_name', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    #df_product = pd.DataFrame(df_1)
    #df_full = pd.merge(df_product, final_annotation_data, how = 'right', on='reviewBody', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    df_product = pd.DataFrame(product_data)
    df_review = pd.DataFrame(review_data)
     
    #final_annotation_data = final_annotation_data.drop_duplicates(subset=["annotation_md5"], keep="first")
    

    return  df_product, df_review, final_annotation_data





def create_triplets(df_annotation, i):

    # database connection
    connection = pymysql.connect(host="localhost", port=3306, user="bma52", passwd="HB#FaZa*23271130**", database="ActionRec_DB")
    cursor = connection.cursor()


    product_data = pd.read_sql("SELECT * FROM Products", connection)
    #product_data = product_data.rename(columns = {'name': 'product_name'}, inplace=True)
    review_data = pd.read_sql("SELECT * FROM Reviews", connection)

    df_1 = pd.merge(product_data, review_data, how = 'right', on = 'product_name', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    df_product = pd.DataFrame(df_1)
    df = pd.merge(df_product, df_annotation, how = 'right', on='Review id', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')



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


    if df["Actions"][i] == "No_Action":
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
                df['Valence'][i],
                df['reviewBody'][i], 
                oa + str(df['annotation_md5'][i]),
                schema + str(df["Object"][i]),
                schema + str(df['Actions'][i]),
                arec + str(df['Features'][i]),
                schema + str(df["product_name_md5"][i]),
                df["ratingValue"][i],
                schema + str(df["seller_name"][i]),
                schema + str(df["product_name_md5"][i]),
                df["created_timestamp"][i],
                df['annotation'][i],
                df["product_name"][i],
                df["product_name"][i],
                schema + str(df["seller_name"][i]),
                df["availability"][i],
                df["price"][i],
                df["priceCurrency"][i],
                df["model"][i],
                df["product_name"][i],
                df["product_name"][i],
                df["description"][i],
                df["brand_name"][i],
                df["url"][i],
                df["image"][i]
               ]    


       df_tuples = pd.DataFrame(columns={"Subject", "Predicate", "Object"})

       df_tuples["Subject"] = list_subjects
       df_tuples["Predicate"] = list_predicates
       df_tuples["Object"] = list_objects

       insert_to_sparql(df_tuples, df["annotation_md5"][i])


    


def insert_to_sparql(df_tuples, annotation_md5):

    for index in df_tuples.index:
        queryString = "INSERT DATA {{GRAPH < {0} > {{<<{1}>> <<{2}>> <<{3}>>}}}}".format(annotation_md5, df_tuples["Subject"][index], 
                                      df_tuples["Predicate"][index], df_tuples["Object"][index])
        st.write(queryString)
        ssl._create_default_https_context = ssl._create_unverified_context
        sparql = SPARQLWrapper(
          "https://linked.aub.edu.lb:8080/fuseki/actionrec_ml/update"
        )

        sparql.setQuery(queryString) 
        sparql.method = 'POST'
        sparql.query()
        print("Triplets for the annotation", annotation_md5, "were inserted to sparql sucessfully.")



def insert_checked_annotation(df, i):
    
    # database connection
    try:
        connection = pymysql.connect(host="localhost", port=3306, user="bma52", passwd="HB#FaZa*23271130**", database="ActionRec_DB")
        cursor = connection.cursor()
        mySql_insert_query = """INSERT INTO Laptop (Review_id, reviewBody, annotation, Action_Flag, Action_Probability, Actions, Features, Agent, Environment, Valence, Object, Ability, annotation_md5, created_timestamp) 
                           VALUES 
                           ({0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}) """.format(df["Review id"][i], str(df["reviewBody"][i]), str(df["annotation"][i]), 
                                                      str(df["Action Flag"][i]), float(df["Action Probability"][i]), str(df["Actions"][i]), str(df["Features"][i]), str(df["Agent"][i]),
                                                      str(df["Environment"][i]), str(df["Valence"][i]), str(df["Object"][i]), str(df["Ability"][i]), str(df["annotation_md5"][i]), str(df["created_timestamp"][i]))

        cursor = connection.cursor()
        cursor.execute(mySql_insert_query)
        connection.commit()
        print(cursor.rowcount, "Record inserted successfully into Checked Annotation table")
        cursor.close()

    except mysql.connector.Error as error:
        print("Failed to insert record into Checked Annotation table {}".format(error))






@st.cache(suppress_st_warning=True)
def main(df_product, df_review, df_annotation) -> None:


    actions = ['<select>', 'No_Action','Carry','Chat','Download','Game','Listen','Play','Stream','Teach','Watch','Work','Design','Draw','Exercise',
    'Multitask','Read','Study','Surf','Write','Attend','Browse','Call','Capture','Connect','Move','Scroll','Store','Text','Transfer','Travel',
    'Type','Unlock','Use','Edit','Meet','UsingVideo','Absorb','Access','Add','Break','Buy','Charge','Consume','Crack','Cruise','Do','Drop','Find',
    'Flicker','Flip','Fold','Hold','PlugIn','Purchase','Put','Rotate','Run','Send','Setup','Switch','Take','Touch','View','ch','Delete','Expect','Hear',
    'Install','Load','Looking','Open','Pay','Pickup','Produce','Realize','Reboot','Receive','Remove','Return','Save','Set','Support','Surprise',
    'Upgrade','Backup','Bend','Boot','Close','Communicate','Disconnect','Display','Fall','Improve','Lift','Light','Look','Navigate','Notify','Place',
    'Power','Press','Process','Project','Protect','Reduce','Reflect','Refresh','Respond','Scan','See','Select','Shake','Sign','Sketch','Start','Turn','Update',
    'Vege','Weight','Wipe','Code','Develop','Film','Note','Photograph','Compute','Create','Interact','Record']
    features = ['<select>','ScreenResolution', 'GraphicsCard', 'Performance', 'FPS',
       'ScreenQuality', 'ProcessingPower', 'Lightweight',
       'ScreenRefreshRate', 'CPU', 'Speed', 'BatteryLife', 'Fans',
       'DiscDrive', 'PairXboxController', 'Camera', 'ScreenSize',
       'ApplePencil', 'LogitechPencil', 'SplitScreen', 'WirelessKeyboard',
       'Size', 'Programs', 'AttachableKeyboard', 'Multitask', 'BigSur',
       'Memory', 'Speakers', 'OperatingSystem', 'SSD', 'WordProcessing',
       'Bluetooth', 'Keyboard', 'TabletFunction', 'TouchScreen',
       'Microphone', 'Processor', 'PowerSettings', 'Hinges',
       'WirelessConnection', 'TouchPad', 'WiFi', 'Ports', 'Battery',
       'GooglePlayStore', 'Stylus', 'HardDrive', 'SDCardSlot', 'Trackpad',
       'Screen', 'ScreenBrightness', 'CDDrive', 'Cables', 'PowerButton',
       'PrivacyMode', 'iGPU', 'FingerprintScanner', 'SupportAssistant',
       'CameraButton', 'MicrophoneButton', 'GPU', 'Charger', 'Weight',
       'tr', 'InternalDrive', 'Microsoft', 'Fan', 'NumericKeypad',
       'Cortana', 'SleepMode', 'OpenBox', 'SurfaceSlimPen',
       'WindowsHello', 'SPen']
    environments = ['<select>', 'Universal', 'Travel', 'University', 'Home', 'Work', 'Office',
       'Room']
    agents = ['<select>', 'Person', 'Gamer', 'Employee', 'Son', 'Student', 'Artist',
       'Designer', 'Musician', 'GraphicDesigner', 'Daughter', 'Teacher',
       'Kid', 'Wife', 'Father', 'Psychotherapist', 'FilmMaker',
       'Freelancer', 'Developer', 'Photographer']
    valence = ['<select>', 'Positive', 'Negative', 'Neutral']
    objects = ['<select>','Games', 'Media', 'Application', 'Movie', 'Pictures',
       'Netflix', 'Notes', 'Internet', 'StudentWork', 'Artwork',
       'SchoolWork', 'Drawing', 'Product', 'Lectures', 'VirtualMeeting',
       'Design', 'Data', 'WorkTasks', 'Music', 'FaceTime', 'iMessage',
       'iPhone', 'Laptop', 'Sims', 'Facebook', 'OnlineClasses',
       'AppleProducts', 'Programs', 'Word', 'Show', 'BluetoothDevice',
       'VirtualTeaching', 'Video', 'ZoomMeeting', 'OnlineLearning',
       'Screen', 'Book', 'Meetings', 'WirelessDevice', 'Document',
       'Sound', 'WiFi', 'YouTube', 'VideoObject', 'Spotify', 'Message',
       'MicrosoftOffice', 'Mouse', 'GooglePhone', 'Files', 'Trackpad',
       'RemoteServer', 'CD', 'Storage', 'Desktop', 'Skype', 'Windows10',
       'Solitaire', 'SmartSpeakers', 'Documents', 'Camera', 'PhotoObject',
       'SupportAssistant', 'DrawingObject', 'Microphone', 'Monitor',
       'Memory', 'MailApplication', 'FanNoise', 'USB', 'Numbers',
       'laptop', 'Keys', 'ImageObject', 'Song', 'Charging', 'Manual',
       'Mobile', 'Code', 'Film', 'Audio', 'SignedDocuments', 'Text',
       'GraphicDesigns', 'Information', 'UI/UX', 'Paper', 'eGPU',
       'Leisure', 'MovieObject']


    
    
    
    def form(df_annotation, i):
       st.session_state = i
       #st.session_state.a_list = []
       
       df_checked_annotation = df_annotation

       with st.container():
           st.subheader(df_annotation["annotation"][i])
    
           col1, col2, col3 = st.columns(3)
           
        
           with col1: 
               st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 10px;">Action</p>', unsafe_allow_html=True)
               st.write(df_annotation["Actions"][i])
               #st.caption("Please confirm machine results")
               checked_action = st.radio(
                 "Is machine prediction correct?",
                 ('Yes', 'No'), key="action"+ str(i))
               
               if checked_action == 'Yes':
                    
                    st.caption('')
               else:
                    #st.caption("Please enter the correct action")
                    new_action = st.selectbox(
                       "Please select the correct Action.", actions
                            )
                    st.write(new_action)
                    if new_action != '<select>':
                        
                        df_checked_annotation["Actions"][i] = new_action+"Action"
                    else:
                        df_checked_annotation["Actions"][i] = df_annotation["Actions"][i]
                    
               st.markdown("""---""")
 
               st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 10px;">Feature</p>', unsafe_allow_html=True)
               st.write(df_annotation["Features"][i])
               #st.caption("Please confirm machine results")
               checked_feature = st.radio(
                 "Is machine prediction correct?",
                 ('Yes', 'No'), key="feature"+ str(i))

               if checked_feature == "Yes":
                    st.caption('')
               else:
                    #st.caption("Please enter the correct action")
                    new_feature = st.selectbox(
                       "Please select the correct Feature.", features
                            )
                    if new_feature != '<select>':
                        df_checked_annotation["Features"][i] = new_feature
                    else:
                        df_checked_annotation["Features"][i] = df_annotation["Features"][i]

               #st.button(label="Edit Labels")
               
               
           with col2: 
               st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 10px;">Agent</p>', unsafe_allow_html=True)
               st.write(df_annotation["Agent"][i])
               #st.caption("Please confirm machine results")
               checked_agent = st.radio(
                 "Is machine prediction correct?",
                 ('Yes', 'No'), key="agent"+ str(i))

               if checked_agent == "Yes":
                    st.caption('')
               else:
                    #st.caption("Please enter the correct action")
                    new_agent = st.selectbox(
                       "Please select the correct Agent.", agents
                            )
                    if new_agent != '<select>':
                        df_checked_annotation["Agent"][i] = new_agent
                    else:
                        df_checked_annotation["Agent"][i] = df_annotation["Agent"][i]
                    
                  
               st.markdown("""---""")


               st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 10px;">Valence</p>', unsafe_allow_html=True)
               st.write(df_annotation["Valence"][i])
               #st.caption("Please confirm machine results")
               checked_valence = st.radio(
                 "Is machine prediction correct?",
                 ('Yes', 'No'), key="valence"+ str(i))

               if checked_valence == "Yes":
                    st.caption('')
               else:
                    #st.caption("Please enter the correct action")
                    new_valence = st.selectbox(
                       "Please select the correct Valence.", valence
                            )
                    if new_valence != '<select>':
                        df_checked_annotation["Valence"][i] = new_valence
                    else:
                        df_checked_annotation["Valence"][i] = df_annotation["Valence"][i]

                    
               

           with col3: 
               st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 10px;">Environment</p>', unsafe_allow_html=True)
               st.write(df_annotation["Environment"][i])
               #st.caption("Please confirm machine results")
               checked_env = st.radio(
                 "Is machine prediction correct?",
                 ('Yes', 'No'), key="environment"+ str(i))

               if checked_env == "Yes":
                    st.caption('')
               else:
                    #st.caption("Please enter the correct action")
                    new_env = st.selectbox(
                       "Please select the correct Environment.", environments
                            )
                    if new_env != '<select>':
                        df_checked_annotation["Environment"][i] = new_env
                    else:
                        df_checked_annotation["Environment"][i] = df_annotation["Environment"][i]
                  
               st.markdown("""---""")

               st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 10px;">Object</p>', unsafe_allow_html=True)
               st.write(df_annotation["Object"][i])
               #st.caption("Please confirm machine results")
               checked_obj = st.radio(
                 "Is machine prediction correct?",
                 ('Yes', 'No'), key="object"+ str(i))

               if checked_obj == "Yes":
                    st.caption('')
               else:
                    #st.caption("Please enter the correct action")
                    new_obj = st.selectbox(
                       "Please select the correct Object.", objects
                            )
                    if new_obj != '<select>':
                        df_checked_annotation["Object"][i] = new_obj
                    else:
                        df_checked_annotation["Object"][i] = df_annotation["Object"][i]

               
               confirmed_check = st.checkbox("Confirm annotation", key = i)
       
       return df_checked_annotation, i

               


    def no_form(df_annotation, i):
       #st.session_state.a_list = []
       df_checked_annotation = df_annotation
       st.session_state = i
       with st.container():
           st.subheader(df_annotation["annotation"][i])
           
           st.write("This Annotation has no Action 🚨")
           result1 = st.checkbox("Confirm machine result")
           result2 = st.checkbox("Incorrect machine result, the sentence contains labels.")
           #st.form_submit_button(label="Edit")

           if result2:
               st.write("Please enter all the labels for this annotation.")
               col1, col2, col3 = st.columns(3)
               with col1:
                    new_action = st.selectbox(
                       "Please select the correct Action.", actions
                            )
                    
                    if new_action != '<select>':
                        st.write(new_action)
                        df_checked_annotation["Actions"][i] = new_action+"Action"
                    else:
                        df_checked_annotation["Actions"][i] = df_annotation["Actions"][i]

                    new_feature = st.selectbox(
                       "Please select the correct Feature.", features
                            )
                    if new_feature != '<select>':
                        df_checked_annotation["Features"][i] = new_feature
                    else:
                        df_checked_annotation["Features"][i] = df_annotation["Features"][i]
                        
               with col2:
                    new_agent = st.selectbox(
                       "Please select the correct Agent.", agents
                            )
                    if new_agent != '<select>':
                        df_checked_annotation["Agent"][i] = new_agent
                    else:
                        df_checked_annotation["Agent"][i] = df_annotation["Agent"][i]


                    new_valence = st.selectbox(
                       "Please select the correct Valence.", valence
                            )
                    if new_valence != '<select>':
                        df_checked_annotation["Valence"][i] = new_valence
                    else:
                        df_checked_annotation["Valence"][i] = df_annotation["Valence"][i]


               with col3:
                    new_env = st.selectbox(
                       "Please select the correct Environment.", environments
                            )

                    if new_env != '<select>':
                        df_checked_annotation["Environment"][i] = new_env
                    else:
                        df_checked_annotation["Environment"][i] = df_annotation["Environment"][i]

                    new_obj = st.selectbox(
                       "Please select the correct Object.", objects
                            )

                    if new_obj != '<select>':
                        df_checked_annotation["Object"][i] = new_obj
                    else:
                        df_checked_annotation["Object"][i] = df_annotation["Object"][i]


                    confirmed_check = st.checkbox("Confirm annotation", key = i)
       
       return df_checked_annotation, i



    

    list_reviews = df_review["reviewBody"].unique()
      
    

    
    def review_container(i):
          #placeholder = st.empty()
            
          df_one_review = df_annotation.loc[df_annotation['reviewBody'] == i]
          st.write(df_one_review)
          st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 20px;">Product Name:</p>', unsafe_allow_html=True)
          st.subheader(df_review["product_name"][df_review["reviewBody"] == i].unique())
          st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 20px;">Review Text:</p>', unsafe_allow_html=True)
          st.write(i)
          sorting_proba = st.checkbox("Sort annotations by machine scores", key = i)
          if sorting_proba:
             df_one_review = df_one_review.sort_values(by = ["ActionProbability"] , ascending=False)
          for row in df_one_review.index:
            st.write("The probability of this part of the review having an action is ", df_one_review["ActionProbability"][row])
            if df_one_review["ActionFlag"][row] == "Action Exist":
                 df_checked_annotation, i = form(df_one_review, row)
            else:
                 df_checked_annotation, i = no_form(df_one_review, row)

                 st.markdown("""---""")

 
          


    for i in list_reviews:
        review_container(i)
        
        submit_btn = st.button("Submit Review", key = i)
        
        if submit:
            st.write("Your Review was submitted successfully")
            next_btn = st.button("Next Review", key = i)
            
            if next_btn:
               continue;
            else:
               break;



    

if __name__ == "__main__":


    df_product, df_review, df_annotation = get_new_reviews_mysql()
    main(df_product, df_review, df_annotation)
