from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),  # 👈 ADD THIS LINE
    path("index.html", views.index, name="index"),
    path("UserLogin.html", views.UserLogin, name="UserLogin"),
    path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
    path("Predict.html", views.Predict, name="Predict"),
    path("PredictAction", views.PredictAction, name="PredictAction"),
    path("LoadDataset", views.LoadDataset, name="LoadDataset"),
    path("TrainModel", views.TrainModel, name="TrainModel"),
    path("Register.html", views.Register, name="Register"),
    path("RegisterAction", views.RegisterAction, name="RegisterAction"),
    path("UserScreen", views.UserScreen, name='UserScreen'),
    path('PredictAction', views.PredictAction, name='PredictAction'),
    path('BatchPredictAction', views.BatchPredictAction, name='BatchPredictAction'),
    path('contact/', views.contact_view, name='Contact'),
    path('generate_watch_data/', views.generate_watch_data, name='generate_watch_data'),
]
