

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
   
   
    df_1 = pd.merge(product_data, review_data, how = 'right', on='product_name', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    df_product = pd.DataFrame(df_1)
    df_full = pd.merge(df_product, annotation_data, how = 'right', on='reviewBody', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    #df_product = pd.DataFrame(product_data)
    #df_review = pd.DataFrame(review_data)
     
    #final_annotation_data = final_annotation_data.drop_duplicates(subset=["annotation_md5"], keep="first")
    

    return  df_full
   
   
   
   
def insert_checked_annotation(df):
    
    # database connection
    try:
        host="linked.aub.edu.lb"
        port=3306
        database ="reviews_actions_ml"

        
    
        configs = Properties()
    
        with open('dbconfig.properties', 'rb') as config_file:
              configs.load(config_file)
            
 
         #dbConnection =  mysql.connector.connect("mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(configs.get("db.username").data,configs.get("db.password").data, host, port, database))
        dbConnection = mysql.connector.connect(user=configs.get("db.username").data, password=configs.get("db.password").data, host="linked.aub.edu.lb", database="reviews_actions_ml")


        cursor = dbConnection.cursor()
        #mySql_insert_query = """INSERT INTO CheckedAnnotation (reviewBody, annotation, ActionFlag, ActionProbability, Actions, Features, Agent, Environment, Valence, Object, Ability, annotation_md5) 
                          # VALUES 
                         #  ({0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}) """.format(str(df["reviewBody"][i]), str(df["annotation"][i]), 
                          #                            str(df["ActionFlag"][i]), float(df["ActionProbability"][i]), str(df["Actions"][i]), str(df["Features"][i]), str(df["Agent"][i]),
                          #                            str(df["Environment"][i]), str(df["Valence"][i]), str(df["Object"][i]), str(df["Ability"][i]), str(df["annotation_md5"][i]))

        cols = "`,`".join([str(i) for i in df.columns.tolist()])

    
        for i,row in df.iterrows():
              sql = "INSERT INTO `Product` (`" + cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
              cursor.execute(sql, tuple(row))
              # the connection is not autocommitted by default, so we must commit to save our changes
              
              dbConnection.commit()
        st.write("Record inserted successfully into Checked Annotation table")
      
        #cursor.execute(mySql_insert_query)
        #connection.commit()
        #print(cursor.rowcount, "Record inserted successfully into Checked Annotation table")
        #cursor.close()

    except mysql.connector.Error as error:
        print("Failed to insert record into Checked Annotation table {}".format(error))






@st.cache(ttl=60*5,max_entries=20)
def main(df_annotation) -> None:


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


    df_annotation["Actions"] = df_annotation["Actions"].str.replace("Action", "")
    
    
    def form(df_annotation, i):
       st.session_state = i
       #st.session_state.a_list = []
      
       
       #df_checked_annotation["reviewBody"] = df_annotation["reviewBody"][i]
       #df_checked_annotation["annotation"] = df_annotation["annotation"][i]
       #df_checked_annotation["ActionFlag"] = df_annotation["ActionFlag"][i]
       #df_checked_annotation["ActionProbability"] = df_annotation["ActionProbability"][i]
       #df_checked_annotation["annotation_md5"] = df_annotation["annotation_md5"][i]
       #st.write(df_checked_annotation)
       
       
       
       with st.container():
           st.subheader(df_annotation["annotation"][i])
    
           col1, col2, col3 = st.columns(3)

           
           with col1: 
               #annotation_list = [df_annotation["reviewBody"][i], df_annotation["annotation"][i], df_annotation["ActionFlag"][i], df_annotation["ActionProbability"][i]]
      
    
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
                       "Please select the correct Action.", actions, index= actions.index(df_annotation["Actions"][i])
                            )
                    st.write(new_action)
                    

                    if new_action != '<select>':  
                       df_checked_annotation["Actions"] = new_action+"Action"
                        
                    else:
                       df_checked_annotation["Actions"] = df_annotation["Actions"][i] 
                        

                   
                  
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
                       "Please select the correct Feature.", features, index= features.index(df_annotation["Features"][i])
                            )

  
                  
                    if new_feature != '<select>':
                        df_checked_annotation["Features"] = new_feature
                        
                    else:
                        df_checked_annotation["Features"] = df_annotation["Features"][i]
                       
                  
                    

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
                       "Please select the correct Agent.", agents, index= agents.index(df_annotation["Agent"][i])
                            )
               
                     
                     
                    if new_agent != '<select>':
                        df_checked_annotation["Agent"] = new_agent
                      
                    else:
                        df_checked_annotation["Agent"] = df_annotation["Agent"][i]
                      
                  
                    
                    
                  
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
                       "Please select the correct Valence.", valence, index= valence.index(df_annotation["Valence"][i])
                            )
                  
            
                     
                     
                    if new_valence != '<select>':
                         df_checked_annotation["Valence"] = new_valence
                       
                    else:
                        df_checked_annotation["Valence"] = df_annotation["Valence"][i]
                   
                  
                    

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
                       "Please select the correct Environment.", environments, index= environments.index(df_annotation["Environment"][i])
                            )
                
                     
                     
                    if new_env != '<select>':
                        df_checked_annotation["Environment"] = new_env
                      
                    else:
                       df_checked_annotation["Environment"] = df_annotation["Environment"][i]
                       
                    

                  
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
                       "Please select the correct Object.", objects, index= objects.index(df_annotation["Object"][i])
                            )
                  
                 
                     
                     
                    if new_obj != '<select>':
                        df_checked_annotation["Object"] = new_obj
                      
                    else:
                        df_checked_annotation["Object"] = df_annotation["Object"][i]
         
                    

               
               #annotation_list.append(df_annotation["annotation_md5"][i])
               df_checked_annotation = pd.DataFrame(columns = ["reviewBody","annotation", "ActionFlag", "ActionProbability", "Actions", "Features", "Agent", "Environment", "Valence", "Object", "Ability", "annotation_md5"])
               
               #df_final_checked = df_checked_annotation.append(pd.DataFrame([annotation_list],columns = ["reviewBody","annotation", "ActionFlag", "ActionProbability", "Actions", "Features", "Agent", "Environment", "Valence", "Object", "Ability", "annotation_md5"]), ignore_index=True)
               
               #df_checked_annotation = annotation_list
               
               confirmed_check = st.checkbox("Confirm annotation", key = i)
               if confirmed_check:
                  st.write(df_checked_annotation)
                  insert_checked_annotation(df_checked_annotation)

                  
       st.markdown("""---""")
       return df_checked_annotation, i

               


    def no_form(df_annotation, i):
       #st.session_state.a_list = []
       
       st.session_state = i
         
       
       with st.container():
           st.subheader(df_annotation["annotation"][i])
           df_checked_annotation = pd.DataFrame(columns = ["reviewBody","annotation", "ActionFlag", "ActionProbability", "Actions", "Features", "Agent", "Environment", "Valence", "Object", "Ability", "annotation_md5"])

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
                        df_checked_annotation["Actions"] = new_action+"Action"
                    else:
                        df_checked_annotation["Actions"] = df_annotation["Actions"][i]

                    new_feature = st.selectbox(
                       "Please select the correct Feature.", features
                            )
                    if new_feature != '<select>':
                        df_checked_annotation["Features"] = new_feature
                    else:
                        df_checked_annotation["Features"] = df_annotation["Features"][i]
                        
               with col2:
                    new_agent = st.selectbox(
                       "Please select the correct Agent.", agents
                            )
                    if new_agent != '<select>':
                        df_checked_annotation["Agent"] = new_agent
                    else:
                        df_checked_annotation["Agent"] = df_annotation["Agent"][i]


                    new_valence = st.selectbox(
                       "Please select the correct Valence.", valence
                            )
                    if new_valence != '<select>':
                        df_checked_annotation["Valence"] = new_valence
                    else:
                        df_checked_annotation["Valence"] = df_annotation["Valence"][i]


               with col3:
                    new_env = st.selectbox(
                       "Please select the correct Environment.", environments
                            )

                    if new_env != '<select>':
                        df_checked_annotation["Environment"] = new_env
                    else:
                        df_checked_annotation["Environment"] = df_annotation["Environment"][i]

                    new_obj = st.selectbox(
                       "Please select the correct Object.", objects
                            )

                    if new_obj != '<select>':
                        df_checked_annotation["Object"] = new_obj
                    else:
                        df_checked_annotation["Object"] = df_annotation["Object"][i]

                    st.write(df_checked_annotation)
                    confirmed_check = st.checkbox("Confirm annotation", key = i)
                     
                    df_checked_annotation["reviewBody"] = df_annotation["reviewBody"] 
                    df_checked_annotation["annotation"] = df_annotation["annotation"]
                    df_checked_annotation["ActionFlag"] = df_annotation["ActionFlag"] 
                    df_checked_annotation["ActionProbability"] = df_annotation["ActionProbability"]
                    df_checked_annotation["annotation_md5"] = df_annotation["annotation_md5"]
                  
                    if confirmed_check:
                       insert_checked_annotation(df_checked_annotation)
       st.markdown("""---""")
       return df_checked_annotation, i



    

    list_reviews = df_annotation["reviewBody"].unique()
      
    

    
    def review_container(i):
          #placeholder = st.empty()
          #review_text = df_review["reviewBody"].loc[df_review["reviewBody_md5"] == i]
          df_one_review = df_annotation.loc[df_annotation['reviewBody'] == i]
          st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 20px;">Product Name:</p>', unsafe_allow_html=True)
          st.subheader(df_annotation["product_name"].unique())
          st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 20px;">Review Text:</p>', unsafe_allow_html=True)
          st.write(i)
          sorting_proba = st.checkbox("Sort annotations by machine scores", key = i)
          if sorting_proba:
             df_one_review = df_one_review.sort_values(by = ["ActionProbability"] , ascending=False)
          #list_annotation_md5s = []
          for row in df_one_review.index:
            st.write("The probability of this part of the review having an action is ", df_one_review["ActionProbability"][row])
            
            if df_one_review["ActionFlag"][row] == "Action Exist":
                 df_checked_annotation, i = form(df_one_review, row)
                 #list_annotation_md5s.append(df_checked_annotation["annotation_md5"][i])
            else:
                 df_checked_annotation, i = no_form(df_one_review, row)
                 #list_annotation_md5s.append(df_checked_annotation["annotation_md5"][i])

                 st.markdown("""---""")
                  
          
               
         

    for i in list_reviews:
        review_container(i)
        
        load_next_btn = st.button("Load Next Review", key = df_annotation["review_id"][df_annotation["reviewBody"] == i])
        #next_btn = st.button("Next Review", key = df_one_review["reviewBody"].unique())
        
        if load_next_btn:
            #st.write("Your Review was submitted successfully")
            continue;
                                                                         
        else:
            break;



    

if __name__ == "__main__":


    df = get_new_reviews_mysql()
    main(df)
