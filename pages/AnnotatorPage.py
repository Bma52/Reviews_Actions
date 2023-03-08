

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
from numpy.random import rand
from flask import Flask



#st.markdown('<html><style>.header { width: 1000px; padding: 60px;text-align: center;background: #1abc9c;color: white;font-size: 30px;}</style></html>', unsafe_allow_html=True)




st.markdown('<div class="header"> <H1 align="center"><font style="style=color:lightblue; "> The Annotator Page</font></H1></div>', unsafe_allow_html=True)

chart = functools.partial(st.plotly_chart, use_container_width=True)




def get_new_reviews_mysql(annotator_name):
   

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
   
    
    #checked_data = checked_data[checked_data['checkedBy'] == annotator_name]
   
    #list_checked_annotations = list(checked_data["annotation_md5"])
    #annotation_data = annotation_data[~annotation_data['annotation_md5'].isin(list_checked_annotations)]
      
    
   
    df_1 = pd.merge(product_data, review_data, how = 'right', on='product_name', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    df_product = pd.DataFrame(df_1)
    df_full = pd.merge(df_product, annotation_data, how = 'right', on='reviewBody', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')


    return  df_full
   
   
   
   
def insert_checked_annotation(df):
    

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

        
        df['reviewBody'] = df['reviewBody'].astype(str)
        df["annotation"] = df["annotation"].astype(str)
        df["ActionFlag"] = df["ActionFlag"].astype(str)
        df["ActionProbability"] = df["ActionProbability"].astype(float)
        df["Actions"] = df["Actions"].astype(str)
        df["Features"] = df["Features"].astype(str)
        df["Agent"] = df["Agent"].astype(str)
        df["Environment"] = df["Environment"].astype(str)
        df["Valence"] = df["Valence"].astype(str)
        df["Object"] =  df["Object"].astype(str)
        df["Ability"] = df["Ability"].astype(str)
        #df["User_description"] = df["User_description"].astype(str)
        df["annotation_md5"] = df["annotation_md5"].astype(str)
        df["checkedBy"] = df["checkedBy"].astype(str)
      
      
        cols = "`,`".join([str(i) for i in df.columns.tolist()])

    
        for i,row in df.iterrows():
              sql = "INSERT INTO `CheckedAnnotation` (`" + cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
              cursor.execute(sql, tuple(row))
              st.write("Annotation inserted into MYSQL")
             
              # the connection is not autocommitted by default, so we must commit to save our changes
              dbConnection.commit()
              


def main(df_annotation, annotator_name) -> None:
   

                                 
   
    actions = [' ', 'No_Action','Carry','Chat','Download','Game','Listen','Play','Stream','Teach','Watch','Work','Design','Draw','Exercise',
    'Multitask','Read','Study','Surf','Write','Attend','Browse','Call','Capture','Connect','Move','Scroll','Store','Text','Transfer','Travel',
    'Type','Unlock','Use','Edit','Meet','UsingVideo','Absorb','Access','Add','Break','Buy','Charge','Consume','Crack','Cruise','Do','Drop','Find',
    'Flicker','Flip','Fold','Hold','PlugIn','Purchase','Put','Rotate','Run','Send','Setup','Switch','Take','Touch','View','ch','Delete','Expect','Hear',
    'Install','Load','Looking','Open','Pay','Pickup','Produce','Realize','Reboot','Receive','Remove','Return','Save','Set','Support','Surprise',
    'Upgrade','Backup','Bend','Boot','Close','Communicate','Disconnect','Display','Fall','Improve','Lift','Light','Look','Navigate','Notify','Place',
    'Power','Press','Process','Project','Protect','Reduce','Reflect','Refresh','Respond','Scan','See','Select','Shake','Sign','Sketch','Start','Turn','Update',
    'Vege','Weight','Wipe','Code','Develop','Film','Note','Photograph','Compute','Create','Interact','Record', 'Add a new action']
    features = [' ','ScreenResolution', 'GraphicsCard', 'Performance', 'FPS',
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
    environments = [' ', 'Universal', 'Travel', 'University', 'Home', 'Work', 'Office','Room']
       
    agents = [' ', 'Person', 'Gamer', 'Employee', 'Son', 'Student', 'Artist',
       'Designer', 'Musician', 'GraphicDesigner', 'Daughter', 'Teacher',
       'Kid', 'Wife', 'Father', 'Psychotherapist', 'FilmMaker',
       'Freelancer', 'Developer', 'Photographer']
    valence = [' ', 'positive', 'negative', 'neutral']
    objects = [' ','Games', 'Media', 'Application', 'Movie', 'Pictures',
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

    features =list(map(lambda x: x.lower(), features))
    valence =list(map(lambda x: x.lower(), valence))
             
    

    def form(df_annotation, i, annotator_name):
       st.session_state = i
       #st.session_state.a_list = []
      
       
       
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
                    new_action = df_annotation["Actions"][i]
               else:
                    #st.caption("Please enter the correct action")
                    new_action = st.selectbox("Please select the correct Action.", actions, index= i)
                            
                    
                    if new_action == 'Add a new action':
                         otherOption = st.text_input("Enter your other option...")
                         new_action = otherOption+"Action"
                    else:  
                         new_action = new_action+"Action"
                    
               new_ability = new_action.replace("Action", "Ability")
                    

                  
               st.markdown("""---""")
 
               st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 10px;">Feature</p>', unsafe_allow_html=True)
               df_annotation["Features"][i] = df_annotation["Features"][i].replace("[", "")
               df_annotation["Features"][i] = df_annotation["Features"][i].replace("]", "")
               df_annotation["Features"][i] = df_annotation["Features"][i].replace("'", "")
               st.write(df_annotation["Features"][i])
               #st.caption("Please confirm machine results")
               checked_feature = st.radio(
                 "Is machine prediction correct?",
                 ('Yes', 'No'), key="feature"+ str(i))

               if checked_feature == "Yes":
                    new_feature = df_annotation["Features"][i]
               else:
                    #st.caption("Please enter the correct action")
                    new_feature = st.selectbox("Please select the correct Feature.", features, index= i)
                            


           with col2: 
               st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 10px;">Agent</p>', unsafe_allow_html=True)
               st.write(df_annotation["Agent"][i])
               #st.caption("Please confirm machine results")
               checked_agent = st.radio(
                 "Is machine prediction correct?",
                 ('Yes', 'No'), key="agent"+ str(i))

               if checked_agent == "Yes":
                    new_agent = df_annotation["Agent"][i]
               else:
                    #st.caption("Please enter the correct action")
                    new_agent = st.selectbox("Please select the correct Agent.", agents, index= i)
                            
               

                  
               st.markdown("""---""")


               st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 10px;">Valence</p>', unsafe_allow_html=True)
               st.write(df_annotation["Valence"][i])
               #st.caption("Please confirm machine results")
               checked_valence = st.radio(
                 "Is machine prediction correct?",
                 ('Yes', 'No'), key="valence"+ str(i))

               if checked_valence == "Yes":
                    new_valence = df_annotation["Valence"][i]
               else:
                    #st.caption("Please enter the correct action")
                    new_valence = st.selectbox("Please select the correct Valence.", valence , index= i)
                            
                  
            
                     


           with col3: 
               st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 10px;">Environment</p>', unsafe_allow_html=True)
               st.write(df_annotation["Environment"][i])
               #st.caption("Please confirm machine results")
               checked_env = st.radio(
                 "Is machine prediction correct?",
                 ('Yes', 'No'), key="environment"+ str(i))

               if checked_env == "Yes":
                    new_env = df_annotation["Environment"][i]
               else:
                    new_env = st.selectbox("Please select the correct Environment.", environments, index= df_annotation["Environment"][i])
                            
  
                  
               st.markdown("""---""")

               st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 10px;">Object</p>', unsafe_allow_html=True)
               st.write(df_annotation["Object"][i])
               #st.caption("Please confirm machine results")
               checked_obj = st.radio(
                 "Is machine prediction correct?",
                 ('Yes', 'No'), key="object"+ str(i))

               if checked_obj == "Yes":
                    new_obj = df_annotation["Object"][i]
               else:
                    #st.caption("Please enter the correct action")
                    new_obj = st.selectbox("Please select the correct Object.", objects, index= i)
                            
                  
                 
         

               confirmed_check = st.checkbox("Confirm annotation", key = i)
               df_checked_annotation = pd.DataFrame(columns = ["reviewBody","annotation", "ActionFlag", "ActionProbability", "Actions", "Features", "Agent", "Environment", "Valence", "Object", "Ability", "annotation_md5", "checkedBy"])
            
               df_checked_annotation.loc[i, 'reviewBody'] = df_annotation["reviewBody"][i]
               df_checked_annotation["annotation"] = df_annotation["annotation"][i]
               df_checked_annotation["ActionFlag"] = df_annotation["ActionFlag"][i]
               df_checked_annotation["ActionProbability"] = df_annotation["ActionProbability"][i]
               df_checked_annotation["Actions"] = new_action
               df_checked_annotation["Features"] = new_feature
               df_checked_annotation["Agent"] = new_agent
               df_checked_annotation["Environment"] = new_env
               df_checked_annotation["Valence"] = new_valence
               df_checked_annotation["Object"] = new_obj
               df_checked_annotation["Ability"] = new_ability
               df_checked_annotation["User_description"] = "The {0} has a {1} sentiment becuase he or she {2} using this product".format(new_agent, new_valence, new_action)
               df_checked_annotation["annotation_md5"] = df_annotation["annotation_md5"][i]
               df_checked_annotation["checkedBy"] = annotator_name
               
               #st.dataframe(df_checked_annotation)
               if confirmed_check:
                    if df_checked_annotation.loc[i]["Actions"] == "No_ActionAction":
                         df_checked_annotation["ActionFlag"] = "No Action Found"
                    insert_checked_annotation(df_checked_annotation)
                    


                  
       st.markdown("""---""")
       #return new_action, new_ability, new_feature, new_agent, new_env, new_valence, new_obj, i

               

            
           
    def no_form(df_annotation, i, annotator_name):
       st.session_state.a_list = []
       
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
                    new_action = new_action+"Action"
                    new_ability = new_action+"Ability"
                    

                    new_feature = st.selectbox(
                       "Please select the correct Feature.", features
                            )

                        
               with col2:
                    new_agent = st.selectbox(
                       "Please select the correct Agent.", agents
                            )

                    new_valence = st.selectbox(
                       "Please select the correct Valence.", valence
                            )


               with col3:
                    new_env = st.selectbox(
                       "Please select the correct Environment.", environments
                            )

                    new_obj = st.selectbox(
                       "Please select the correct Object.", objects
                            )


                    #st.write(df_checked_annotation)
                     
                    confirmed_check = st.checkbox("Confirm annotation", key = i)
                    df_checked_annotation = pd.DataFrame(columns = ["reviewBody","annotation", "ActionFlag", "ActionProbability", "Actions", "Features", "Agent", "Environment", "Valence", "Object", "Ability", "annotation_md5", "checkedBy"])
            
                    df_checked_annotation.loc[i, 'reviewBody'] = df_annotation["reviewBody"][i]
                    df_checked_annotation["annotation"] = df_annotation["annotation"][i]
                    df_checked_annotation["ActionFlag"] = df_annotation["ActionFlag"][i]
                    df_checked_annotation["ActionProbability"] = df_annotation["ActionProbability"][i]
                    df_checked_annotation["Actions"] = new_action
                    df_checked_annotation["Features"] = new_feature
                    df_checked_annotation["Agent"] = new_agent
                    df_checked_annotation["Environment"] = new_env
                    df_checked_annotation["Valence"] = new_valence
                    df_checked_annotation["Object"] = new_obj
                    df_checked_annotation["Ability"] = new_ability
                    df_checked_annotation["User_description"] = "The {0} has a {1} sentiment becuase he or she {2} using this product".format(new_agent, new_valence, new_action)
                
                    df_checked_annotation["annotation_md5"] = df_annotation["annotation_md5"][i]
                    df_checked_annotation["checkedBy"] = annotator_name
               
                    if confirmed_check:
                         if df_checked_annotation["Actions"].loc[0] == "No_ActionAction":
                            df_checked_annotation["ActionFlag"] = "No Action Found"
                         insert_checked_annotation(df_checked_annotation)
           
                     

       st.markdown("""---""")
       #return new_action, new_ability, new_feature, new_agent, new_env, new_valence, new_obj, i


    

    
      
    

    
    def review_container(i, df_annotation, annotator_name):
       
          st.session_state = i
          df_one_review = df_annotation.loc[df_annotation['reviewBody'] == i]
          df_one_review = df_one_review.drop_duplicates(subset=['annotation_md5'], keep='first')
          st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 20px;">Product Name:</p>', unsafe_allow_html=True)
          st.subheader(df_annotation["product_name"].unique())
          st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 20px;">Review Text:</p>', unsafe_allow_html=True)
          st.write(i)
          sorting_proba = st.checkbox("Sort annotations by machine scores", key = i)
          if sorting_proba:
             df_one_review = df_one_review.sort_values(by = ["ActionProbability"] , ascending=False)

       
          for row in df_one_review.index:
            st.write("The probability of this part of the review having an action is ", df_one_review["ActionProbability"][row])
            #df_checked_annotation = pd.DataFrame(columns = ["reviewBody","annotation", "ActionFlag", "ActionProbability", "Actions", "Features", "Agent", "Environment", "Valence", "Object", "Ability", "annotation_md5", "checkedBy"])
            
            if df_one_review["ActionFlag"][row] == "Action Exist":
                 form(df_one_review, row, annotator_name)

            else:
                 no_form(df_one_review, row, annotator_name)
                  
   
                  

    list_reviews = df_annotation["reviewBody"].unique()         

      
    for review in list_reviews:
        review_container(review, df_annotation, annotator_name)
   

    
 
    

if __name__ == "__main__":
   
    annotators = ["","Bma52", "Fz13", "Wk14"]
    annotator_name = st.selectbox("Please enter your name", annotators)
    if annotator_name != "":
         df = get_new_reviews_mysql(annotator_name)
         main(df,annotator_name)
         


    
    


    
         
