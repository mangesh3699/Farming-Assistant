

from __future__ import print_function
from urllib import response
import markups
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
from sklearn.model_selection import cross_val_score
from . import fertilizer
from . import config
import requests
warnings.filterwarnings('ignore')

def crop_predict1(inp):
    df = pd.read_csv(r"C:/Users/mange/farming/Datasets/crop_recommendation.csv")
    features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
    target = df['label']
    #features = df[['temperature', 'humidity', 'ph', 'rainfall']]
    labels = df['label']
    # Initialzing empty lists to append all model's name and corresponding name
    acc = []
    model = []
        
    from sklearn.model_selection import train_test_split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
    #XGB
    import xgboost as xgb
    XB = xgb.XGBClassifier()
    XB.fit(Xtrain,Ytrain)

    predicted_values = XB.predict(Xtest)

    x = metrics.accuracy_score(Ytest, predicted_values)
    acc.append(x*100)
    model.append('XGBoost')
    print("XGBoost's Accuracy is: ", x)
    # Random Forest
    #from sklearn.ensemble import RandomForestClassifier
    
    #RF = RandomForestClassifier(n_estimators=20, random_state=0)
    #RF.fit(Xtrain,Ytrain)

    #predicted_values = RF.predict(Xtest)

    #x = metrics.accuracy_score(Ytest, predicted_values)
    #acc.append(x)
    #model.append('RF')
    #print("RF's Accuracy is: ", x)

    # Cross validation score (Random Forest)
    #score = cross_val_score(RF,features,target,cv=5)
    #score
    #CVS XGB
    score = cross_val_score(XB,features,target,cv=5)
    score
    ### Saving trained Random Forest model
    #import pickle
    #RF_pkl_filename = r"C:/Users/mange/farming/models/RandomForest.pkl"
    # Open the file to save as pkl file
    #RF_Model_pkl = open(RF_pkl_filename, 'wb')
    #pickle.dump(RF, RF_Model_pkl)
    # Close the pickle instances
    #RF_Model_pkl.close()
    # saving XGB
    import pickle
    # Dump the trained Naive Bayes classifier with Pickle
    XB_pkl_filename = 'C:/Users/mange/farming/models/XGBoost.pkl'
    # Open the file to save as pkl file
    XB_Model_pkl = open(XB_pkl_filename, 'wb')
    pickle.dump(XB, XB_Model_pkl)
    # Close the pickle instances
    XB_Model_pkl.close()
    ## Accuracy Comparison
    accuracy_models = dict(zip(model, acc))
    for k, v in accuracy_models.items():
        print(k, '-->', v)
    ## Making a prediction
    #data = np.array([[96, 54, 22, , 70.3, 6.96, 150.9]])
    print(inp)
    data = np.array(inp)
    probs = XB.predict_proba(data)
    prediction = np.argsort(probs)[:,-3:]
    prediction[:,[0, 2]] = prediction[:,[2, 0]]
    
    print(prediction)
    top_socs = XB.classes_[prediction]
    print(top_socs)
    return top_socs
    #print(prediction)



def crop_predict2(inp):
    state= inp[0][0]
    district= inp[0][1]
    district= district.strip()
    print(district)
    season = inp[0][2]
    #print(state)
    path = 'C:/Users/mange/farming/Datasets/crop_production.csv'
    df = pd.read_csv(path)
    df
    df_state = df[df["State_Name"]==state]
    print(df_state)
    df_district = df_state[df_state["District_Name"]==district ]
    print(df_district)
    #print(df_district[df_district["Season"]=="Kharif"])

    df_district = df_district.dropna()
    df_district["Season"].unique()
    df_season = df_district[df_district["Season"]== season ]
    df_season
    print(df_season)

    
    l = list(df_season["Crop"].unique())
    len(df_season)
    df_season = df_season.dropna()
    
    # To reset the indices
    df_season = df_season.reset_index(drop = True)
    from numpy.core.numeric import NaN
    data = {}
    for v in l:
      data[v]=[0,0]
      for i in range(len(df_season)):
        if(df_season.iloc[i]["Crop"]==v):
          data[v][0] = data[v][0] + df_season.iloc[i]["Production"]
          data[v][1] = data[v][1] + df_season.iloc[i]["Area"]
    data
    d = sorted(data.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
    print(d)
    #print(d[0],d[1],d[2])
    return d

def fertilizer_prediction(inp):
    crop_name = inp[0][3]
    N = inp[0][0]
    P = inp[0][1]
    K = inp[0][2]
    # ph = float(request.form['ph'])

    df = pd.read_csv('C:/Users/mange/farming/Datasets/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]
    
    n = nr.astype(int) - int(N)
    p = pr.astype(int) - int(P)
    k = kr.astype(int) - int(K)
    if n==0:
        key1 = 'NoneN'
    else:
        if n>0:
            n=n-5
            if n>0:
                key1 = 'Nlow'       
        else:
            n=n+5
            if n<0:
                key1 = "NHigh"
    if p==0:
        key2 = 'NoneP'
    else:
        if p>0:
            p=p-5
            if p>0:
                key2 = 'Plow'
        else:
            p=p+5
            if p<0:
                key2 = "PHigh"
    if k==0:
        key3 = 'NoneK'
    else:
        if k>0:
            k=k-5
            if k>0:
                key3 = 'Klow'
        else:
            k=k+5
            if k<0:
                key3 = "KHigh"
    
    #temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    # max_value = temp[max(temp.keys())]
    # if max_value == "N":
    #     if n < 0:
    #         key = 'NHigh'
    #     else:
    #         key = "Nlow"
    # elif max_value == "P":
    #     if p < 0:
    #         key = 'PHigh'
    #     else:
    #         key = "Plow"
    # else:
    #     if k < 0:
    #         key = 'KHigh'
    #     else:
    #         key = "Klow"
    
    markup = markups.ReStructuredTextMarkup()
    response1= markup.convert(fertilizer.fertilizer_dic[key1]).get_document_body()
    response2= markup.convert(fertilizer.fertilizer_dic[key2]).get_document_body()
    response3= markup.convert(fertilizer.fertilizer_dic[key3]).get_document_body()
    resp = [response1, response2, response3]
    #response = markup.convert(fertilizer.fertilizer_dic[key])
    print(resp)
    return resp

def weather_fetch(city_name):
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    print(complete_url)
    response = requests.get(complete_url)
    x = response.json()
    print(x)
    if x["cod"] != "404":
        y = x["main"]
        temperature = round((y["temp"] - 273.15), 2)
        print(temperature)
        humidity = y["humidity"]
        print(humidity)
        weather=[[temperature, humidity]]
        return weather
    else:
        return None

def weather_forecast(data):
    print(data)
    state = data[0][0]
    city = str(data[0][1])
    city = city.strip()
    print(city,state)
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/forecast?"
    complete_url = base_url + "q=" + city + "&units=metric" + "&appid=" + api_key 
    print(complete_url)
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        p=x['list']
        return p

       









       
    
