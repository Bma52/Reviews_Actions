

from cProfile import label
from email.policy import default
import functools
from locale import D_FMT
from os import uname

from pickletools import float8
from pyexpat import features
from sqlite3 import DatabaseError
from statistics import multimode
from turtle import color
from xmlrpc.client import Boolean
#from msilib import datasizemask
#from pathlib import Path

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

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import operator 
from heapq import nlargest

import json
import csv
from SPARQLWrapper import SPARQLWrapper, JSON
import ssl
import networkx as nx
import streamlit_authenticator as stauth
import yaml
from streamlit_authenticator import hasher 
import pymysql


from flask import Flask



#st.markdown('<html><style>.header { width: 1000px; padding: 60px;text-align: center;background: #1abc9c;color: white;font-size: 30px;}</style></html>', unsafe_allow_html=True)




st.markdown('<div class="header"> <H1 align="center"><font style="style=color:lightblue; "> The ActionRec Annotator Page</font></H1></div>', unsafe_allow_html=True)

chart = functools.partial(st.plotly_chart, use_container_width=True)




def get_new_reviews_mysql() -> pd.DataFrame:
   
    # database connection
    connection = pymysql.connect(host="localhost", port=8889, user="root", passwd="root", database="ActionRec_DB")
    cursor = connection.cursor()


    product_data = pd.read_sql("SELECT * FROM Products", connection)
    product_data = product_data.rename(columns = {'name': 'product_name'}, inplace=True)
    review_data = pd.read_sql("SELECT * FROM Reviews", connection)
    annotation_data = pd.read_sql("SELECT * FROM Annotation", connection)
    annotation_data["Action Probability"] = annotation_data["Action Probability"].astype(float)
    
    #df_1 = product_data.merge(review_data, how = 'right', on = '')
    df_full = review_data.merge(annotation_data, how = 'right', on='Review id', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    

    return df_full


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
                 oa + str(df['Annotation'][i]),
                 arec + str(df['Features'][i]),
                 oa + str(df['Annotation'][i]),
                 schema + str(df['reviewBody'][i]),
                 schema + str(df["Product"][i]),
                 schema + str(df["Actions"][i]),
                 schema + str(df["name_md5"][i]),
                 schema + str(df["name_md5"][i]),
                 schema + str(df['reviewBody'][i]),
                 schema + str(df["offers"][i]),
                 schema + str(df["offers"][i]),
                 oa + str(df['Annotation'][i]),
                 oa + str(df['annotation_md5'][i]),
                 schema + str(df['reviewBody_md5'][i]),
                 schema + str(df['reviewBody_md5'][i]),
                 schema + str(df['reviewBody_md5'][i]),
                 schema + str(df["offers"][i]),
                 schema + str(df["offers"][i]),
                 schema + str(df["offers"][i]),
                 schema + str(df["name_md5"][i]),
                 schema + str(df["name_md5"][i]),
                 schema + str(df["name_md5"][i]),
                 schema + str(df["name_md5"][i]),
                 schema + str(df["name_md5"][i]),
                 schema + str(df["name_md5"][i]),
                 schema + str(df["name_md5"][i])
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


    list_objects = [oa + str(df['Annotation'][i]),
                oa + str(df['Annotation'][i]),
                oa + str(df['Annotation'][i]),
                oa + str(df['Annotation'][i]),
                oa + str(df['Annotation'][i]),
                arec + str(df['Ability'][i]),
                schema + str(df['Actions'][i]),
                arec + str(df['Agent'][i]),
                schema + str(df['Environment'][i]),
                schema + str(df['Object'][i]),
                schema + str(df['reviewBody'][i]),
                schema + str(df['Actions'][i]),
                df['Valence'][i],
                df['reviewBody'][i], 
                oa + str(df['Annotation'][i]),
                schema + str(df["Object"][i]),
                schema + str(df['Actions'][i]),
                arec + str(df['Feature'][i]),
                schema + str(df["Product"][i]),
                df["reviewRating"][i],
                schema + str(df["publisher"][i]),
                schema + str(df["Product"][i]),
                df["created_timestamp"][i],
                df['annotation'][i],
                df["name"][i],
                df["name"][i],
                schema + str(df["publisher"][i]),
                df["Offers"][i],
                df["Offers"][i],
                df["Offers"][i],
                df["model"][i],
                df["name"][i],
                df["name"][i],
                df["description"][i],
                df["brand"][i],
                df["url"][i],
                df["image"][i]
               ]    


    df_tuples = pd.DataFrame(columns={"Subject", "Predicate", "Object"})

    df_tuples["Subject"] = list_subjects
    df_tuples["Predicate"] = list_predicates
    df_tuples["Object"] = list_objects


    return df_tuples





def main(df) -> None:

    actions = ['No Action','Carry','Chat','Download','Game','Listen','Play','Stream','Teach','Watch','Work','Design','Draw','Exercise',
    'Multitask','Read','Study','Surf','Write','Attend','Browse','Call','Capture','Connect','Move','Scroll','Store','Text','Transfer','Travel',
    'Type','Unlock','Use','Edit','Meet','UsingVideo','Absorb','Access','Add','Break','Buy','Charge','Consume','Crack','Cruise','Do','Drop','Find',
    'Flicker','Flip','Fold','Hold','PlugIn','Purchase','Put','Rotate','Run','Send','Setup','Switch','Take','Touch','View','ch','Delete','Expect','Hear',
    'Install','Load','Looking','Open','Pay','Pickup','Produce','Realize','Reboot','Receive','Remove','Return','Save','Set','Support','Surprise',
    'Upgrade','Backup','Bend','Boot','Close','Communicate','Disconnect','Display','Fall','Improve','Lift','Light','Look','Navigate','Notify','Place',
    'Power','Press','Process','Project','Protect','Reduce','Reflect','Refresh','Respond','Scan','See','Select','Shake','Sign','Sketch','Start','Turn','Update',
    'Vege','Weight','Wipe','Code','Develop','Film','Note','Photograph','Compute','Create','Interact','Record']
    features = ['ScreenResolution', 'GraphicsCard', 'Performance', 'FPS',
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
    environments = ['Universal', 'Travel', 'University', 'Home', 'Work', 'Office',
       'Room']
    agents = ['Person', 'Gamer', 'Employee', 'Son', 'Student', 'Artist',
       'Designer', 'Musician', 'GraphicDesigner', 'Daughter', 'Teacher',
       'Kid', 'Wife', 'Father', 'Psychotherapist', 'FilmMaker',
       'Freelancer', 'Developer', 'Photographer']
    valence = ['Positive', 'Negative', 'Neutral']
    objects = ['Games', 'Media', 'Application', 'Movie', 'Pictures',
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


    
    
    
    def form(df, i):
       st.session_state = i
       df_checked_annotation = df
       
       with st.form(key=f"{i}"):
           st.subheader(df["annotation"][i])
    
           col1, col2, col3 = st.columns(3)
           
        
           with col1: 
               
               st.write(df["Actions"][i])
               #st.caption("Please confirm machine results")
               checked_action = st.radio(
                 "Please verify machine prediction",
                 ('Yes', 'No'), key="action")

               if checked_action == "Yes":
                    st.caption('')
               else:
                    #st.caption("Please enter the correct action")
                    new_action = st.selectbox(
                       "Please select the correct Action.", actions
                            )
                    if new_action:
                        df_checked_annotation["Actions"][i] = new_action
                    else:
                        df_checked_annotation["Actions"][i] = df["Actions"][i]
                  
               st.markdown("""---""")
 

               st.write(df["Features"][i])
               #st.caption("Please confirm machine results")
               checked_feature = st.radio(
                 "Please confirm machine prediction",
                 ('Yes', 'No'), key="feature")

               if checked_feature == "Yes":
                    st.caption('')
               else:
                    #st.caption("Please enter the correct action")
                    new_feature = st.selectbox(
                       "Please select the correct Feature.", features
                            )
                    if new_feature:
                        df_checked_annotation["Features"][i] = new_feature
                    else:
                        df_checked_annotation["Features"][i] = df["Features"][i]

               st.form_submit_button(label="Edit Labels")
               
               
           with col2: 
               st.write(df["Agent"][i])
               #st.caption("Please confirm machine results")
               checked_agent = st.radio(
                 "Please verify machine prediction",
                 ('Yes', 'No'), key="agent" )

               if checked_agent == "Yes":
                    st.caption('')
               else:
                    #st.caption("Please enter the correct action")
                    new_agent = st.selectbox(
                       "Please select the correct Agent.", agents
                            )
                    if new_agent:
                        df_checked_annotation["Agent"][i] = new_agent
                    else:
                        df_checked_annotation["Agent"][i] = df["Agent"][i]
                    
                  
               st.markdown("""---""")


               st.write(df["Valence"][i])
               #st.caption("Please confirm machine results")
               checked_valence = st.radio(
                 "Please confirm machine prediction",
                 ('Yes', 'No'), key="valence")

               if checked_valence == "Yes":
                    st.caption('')
               else:
                    #st.caption("Please enter the correct action")
                    new_valence = st.selectbox(
                       "Please select the correct Valence.", valence
                            )
                    if new_valence:
                        df_checked_annotation["Valence"][i] = new_valence
                    else:
                        df_checked_annotation["Valence"][i] = df["Valence"][i]
               

           with col3: 
               st.write(df["Environment"][i])
               #st.caption("Please confirm machine results")
               checked_env = st.radio(
                 "Please verify machine prediction",
                 ('Yes', 'No'), key="environment")

               if checked_env == "Yes":
                    st.caption('')
               else:
                    #st.caption("Please enter the correct action")
                    new_env = st.selectbox(
                       "Please select the correct Environment.", environments
                            )
                    if new_env:
                        df_checked_annotation["Environment"][i] = new_env
                    else:
                        df_checked_annotation["Environment"][i] = df["Environment"][i]
                  
               st.markdown("""---""")


               st.write(df["Object"][i])
               #st.caption("Please confirm machine results")
               checked_obj = st.radio(
                 "Please confirm machine prediction",
                 ('Yes', 'No'), key="object")

               if checked_obj == "Yes":
                    st.caption('')
               else:
                    #st.caption("Please enter the correct action")
                    new_obj = st.selectbox(
                       "Please select the correct Object.", objects
                            )
                    if new_obj:
                        df_checked_annotation["Object"][i] = new_obj
                    else:
                        df_checked_annotation["Object"][i] = df["Object"][i]
               

               

            
    def no_form(df, i):
       #st.session_state.a_list = []
       df_checked_annotation = df
       st.session_state = i
       with st.form(key=f"{i}"):
           st.subheader(df["annotation"][i])
           
           st.write("This Annotation has no Action 🚨")
           result1 = st.checkbox("Confirm machine result")
           result2 = st.checkbox("Incorrect machine result, the sentence contains labels.")
           st.form_submit_button(label="Edit")

           if result2:
               st.write("Please enter all the labels for this annotation.")
               col1, col2, col3 = st.columns(3)
               with col1:
                    new_action = st.selectbox(
                       "Please select the correct Action.", actions
                            )
                    if new_action:
                        df_checked_annotation["Actions"][i] = new_action
                    else:
                        df_checked_annotation["Actions"][i] = df["Actions"][i]

                    new_feature = st.selectbox(
                       "Please select the correct Feature.", features
                            )
                    if new_feature:
                        df_checked_annotation["Features"][i] = new_feature
                    else:
                        df_checked_annotation["Features"][i] = df["Features"][i]
                        
               with col2:
                    new_agent = st.selectbox(
                       "Please select the correct Agent.", agents
                            )
                    if new_agent:
                        df_checked_annotation["Agent"][i] = new_agent
                    else:
                        df_checked_annotation["Agent"][i] = df["Agent"][i]


                    new_valence = st.selectbox(
                       "Please select the correct Valence.", valence
                            )
                    if new_valence:
                        df_checked_annotation["Valence"][i] = new_valence
                    else:
                        df_checked_annotation["Valence"][i] = df["Valence"][i]


               with col3:
                    new_env = st.selectbox(
                       "Please select the correct Environment.", environments
                            )

                    if new_env:
                        df_checked_annotation["Environment"][i] = new_env
                    else:
                        df_checked_annotation["Environment"][i] = df["Environment"][i]

                    new_obj = st.selectbox(
                       "Please select the correct Object.", objects
                            )

                    if new_obj:
                        df_checked_annotation["Object"][i] = new_obj
                    else:
                        df_checked_annotation["Object"][i] = df["Object"][i]


                    st.form_submit_button(label="Submit")



    

    list_reviews = df["reviewBody"].unique()

    
    def review_container(i):
          #placeholder = st.empty()
          
          df_one_review = df.loc[df['reviewBody'] == i]
          st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 20px;">Product Name:</p>', unsafe_allow_html=True)
          st.subheader(df_one_review["itemReviewed"].unique())
          st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 20px;">Review Text:</p>', unsafe_allow_html=True)
          st.write(i)
          sorting_proba = st.checkbox("Sort annotations by machine scores", key = i)
          if sorting_proba:
             df_one_review = df_one_review.sort_values(by = ["Action Probability"] , ascending=False)
          for row in df_one_review.index:
            st.write("The probability of this part of the review having an action is ", df_one_review["Action Probability"][row])
            if df_one_review["Action Flag"][row] == "Action Exist":
                 form(df_one_review, row)
            else:
                 no_form(df_one_review, row)

    for i in list_reviews:
        review_container(i)
        submit = st.button("Submit Review", key = i)
        next = st.button("Next Review", key = i)
        if submit:
            st.write("Your Review was submitted successfully")
            
        if next:
            continue;
        else:
            break;



    

if __name__ == "__main__":



    df = get_new_reviews_mysql()
    main(df)