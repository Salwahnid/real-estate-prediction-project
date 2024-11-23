from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.house_price_prediction, name='house_price_prediction'),
]
