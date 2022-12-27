from django.urls import path
from .import views

urlpatterns = [
    path('', views.index, name='index'),
    path('crop/', views.crop_prediction, name='crop_prediction'),
    path('crop/crop_result/', views.crop_result, name='crop_result'),
    path('contact/', views.contact, name='contact'),
    path('crop2/', views.crop2_prediction, name='crop2_prediction'),
    path('crop2/crop_result2/', views.crop2_result, name='crop2_result'),
    path('fertilizer/', views.fertilizer_predict, name='fertilizer'),
    path('fertilizer/fertilizer_result/',views.ferti_result, name='fertilizer_result'),
    path('weather/', views.weather_prediction, name='weather_prediction'),
    path('weather/weather_forecast/', views.weather_prediction_result, name='weather_prediction_result'),
]