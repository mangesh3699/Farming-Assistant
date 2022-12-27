from django.shortcuts import render, redirect
from . import Mlmodel
import numpy as np
import pickle

# Create your views here.
from django.http import HttpResponse

def index(request):
     #return HttpResponse('#return render(request, "index.html")')
    return render(request, 'index.html')

def crop_prediction(request):
     return render(request, 'crop.html')

def crop_result(request):
     postdata = request.POST
     n = int(postdata.get('nitrogen'))
     p = int(postdata.get('phosphorous'))
     k = int(postdata.get('pottasium'))
     ph = float(postdata.get('ph'))
     rainfall = int(postdata.get('rainfall'))
     State = postdata.get('stt')
     city = postdata.get('city')
     temp_humi= Mlmodel.weather_fetch(city)
     if temp_humi == None:
      
          crop={
               'key': 'NONE',
               'crop1': 'Temperature',
               'crop2': 'Humidity cannot be fetched.',
               'crop3': ''
          }
     else:
          temperature = temp_humi[0][0]
     #print(temperature)
          humidity = temp_humi[0][1]
     #print(humidity)
          data=[[n, p, k, temperature, humidity, ph, rainfall ]]
          #result = Mlmodel.crop_predict1(data)
          
          # Dump the trained Naive Bayes classifier with Pickle
          XB_pkl_filename = 'C:/Users/mange/farming/models/XGBoost.pkl'
          model = pickle.load( open(XB_pkl_filename, 'rb'))
          # Open the file to save as pkl file
          
          data = np.array(data)
          probs = model.predict_proba(data)
          prediction = np.argsort(probs)[:,-3:]
          prediction[:,[0, 2]] = prediction[:,[2, 0]]
          
          print(prediction)
          top_socs = model.classes_[prediction]
          print(top_socs)
          result=top_socs
          crop ={
               'crop1': result[0][0],
               'crop2': result[0][1],
               'crop3': result[0][2]
          }

     
     
     return render(request, 'crop_result.html', crop)
     
def crop2_prediction(request):
     return render(request, 'crop2.html')

def crop2_result(request):
     postdata = request.POST
     State = postdata.get('stt')
     District = postdata.get('district')
     Season = postdata.get('Season')
     data =[[State, District, Season]]
     result = Mlmodel.crop_predict2(data)
     #print(result)
     n=len(result)
     print(n)
     #for i in range
     listing =[]
     for i in range(0,3):
          i1=result[i]
          t1=i1[0]
          t2=i1[1][0]
          t3=i1[1][1]
          list1=[t1,t2,t3]
          listing.append(list1)
     crop = {
               'crop1': listing[0],
               'crop2': listing[1],
               'crop3': listing[2]

     }
     

     print(crop)
     return render(request, 'crop_result2.html', {'keys':crop})

def contact(request):
     return render(request, 'contact.html')

def fertilizer_predict(request):
     return render(request, 'fertilizer.html')

def ferti_result(request):
     postdata = request.POST
     n = postdata.get('nitrogen')
     p = postdata.get('phosphorous')
     k = postdata.get('pottasium')
     crop = postdata.get('cropname')
     data =[[n, p, k, crop]]
     result= Mlmodel.fertilizer_prediction(data)
     print(result)
     f_result = {'fert1': result[0],
                 'fert2': result[1],
                 'fert3':result[2]
     }
     #f_result ={'fert': result}
     
     return render(request, 'fertilizer_result.html', f_result)


def weather_prediction(request):
     return render(request, 'weather.html')

def weather_prediction_result(request):
     postdata = request.POST
     State = postdata.get('stt')
     District = postdata.get('city')
     data =[[ State, District ]]
     list = Mlmodel.weather_forecast(data)
     length = len(list)
     print(length)

     listing = []

     for i in range(1,length,4):
          y=list[i]['main']
          temp = y['temp']
          temp_min = y['temp_min']
          temp_max = y['temp_max']
          pressure = y['pressure']
          humidity = y['humidity']
          z=list[i]['weather'][0]
          weather = z['main']
          sky = z['description']
          a=list[i]['wind']
          wind_speed = a['speed']
          degree = a['deg']
          gust = a['gust']
          b = list[i]
          visibility = b['visibility']
          date = b['dt_txt']
          list1 = [date,temp,temp_min,temp_max,pressure,humidity,weather,sky,wind_speed,degree,gust,visibility]
          print(list1)
          #listing[i-1] = [date,temp,temp_min,temp_max,pressure,humidity,weather,sky,wind_speed,degree,gust,visibility]
          listing.append(list1)
          #print(listing[i])
     print(listing)
         
     weather_dict ={'key0':listing[0],'key1':listing[1],'key2':listing[2],'key3':listing[3],'key4':listing[4],'key5':listing[5],'key6':listing[6],'key7':listing[7],'key8':listing[8],'key9':listing[9]}
     print(weather_dict)
       #print(list[i])
     #w_result ={'fert': result}
     return render(request, 'weather_forecast.html',{'keys': weather_dict} )



