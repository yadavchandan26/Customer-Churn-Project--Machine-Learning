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

def feature_selection(data):
    x=data.drop('Churn',axis=1)
    y=data['Churn']

    return x,y 

def final_loading(data_path):
    data=load_data(data_path)
    data=preprocess(data)
    x,y=feature_selection(data)

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=42)
    return x_train,x_test,y_train,y_test
    