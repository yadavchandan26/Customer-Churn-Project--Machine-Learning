import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path):
    data=pd.read_csv(data_path)
    return data

def preprocess(data):
    data=data.replace({
        'Yes':1,
        'No':0
    })
    data['gender']=data['gender'].replace({
        'Male':1,
        'Female':0
        })
    data=data.replace({
        'No internet service':0,
        'No phone service':0
    })
    ohe=['Contract','InternetService','PaymentMethod']
    data=pd.get_dummies(data,columns=ohe,drop_first=True)
    
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(0)

    return data
