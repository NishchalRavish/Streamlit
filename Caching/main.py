import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier

COLS = ['class', 'odor', 'gill-size', 'gill-color', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'ring-type', 'spore-print-color']

@st.cache_data(show_spinner="Fetching data")
def read_data(cols):
    df = pd.read_csv('data/mushrooms.csv')
    df = df[cols]
    return df

@st.cache_resource #It's cache resource as LE can't be hashable(It keeps changing)
def get_target_encoder(data):
    le = LabelEncoder()
    le.fit(data)
    
    return le

@st.cache_resource
def get_feature_encoder(data):
    oe = OrdinalEncoder()
    X_cols = data.columns[1:]
    oe.fit(X_cols)
    
    return oe

@st.cache_data(show_spinner='Encoding Data')
def encode_data(data,_X_encoder,_y_encoder): # _X(will not be hashed) (The _ represents not to be cached as it keeps changing)
    data['class'] = _y_encoder.transform(data['class'])
    
    X_cols = data.columns[1:]
    data[X_cols] = _X_encoder.transform(data[X_cols])
    
    return data
    
@st.cache_resource(show_spinner='Model is Training')
def train_model(data):
    X = data.drop(['class'],axis=1)
    y = data['class']   
    
    gbc = GradientBoostingClassifier(max_depth=5, random_state=42)
    gbc.fir(X,y)
    
    return gbc

@st.cache_data(show_spinner='Making the Prediction')
def make_prediction(_model,_X_encoder,X_pred):
    features = [each[0] for each in X_pred]
    features = np.array(features).reshape(-1,1)
    
    encoded_features = _X_encoder.transform(features)
    
    pred = _model.predict(encoded_features)
    
    return pred[0]
    