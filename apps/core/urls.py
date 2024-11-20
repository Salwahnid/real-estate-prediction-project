from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_price_api, name='predict_price_api'),
]
