import pickle
import pandas as pd
from rest_framework.decorators import api_view
from rest_framework.response import Response
from dateutil.relativedelta import relativedelta
from .serializers import HousePricePredictionSerializer
import numpy as np
from decimal import Decimal

# Charger le modèle sauvegardé
def load_model():
    with open('C:/Users/WIN/Desktop/IA_project/real-estate-prediction-project/apps/core/model/sarima_model.pkl', 'rb') as f:  # Assurez-vous que le chemin est correct
        model_fit = pickle.load(f)
    return model_fit



# Fonction de prédiction
def predict_house_price(current_price, date_to_predict):
    """
    Prédit le prix de la maison basé sur le prix actuel et la date à prédire
    
    :param current_price: Prix actuel de la maison
    :param date_to_predict: La date à laquelle vous voulez prédire le prix (format 'YYYY-MM-DD')
    :return: Le prix prédit pour la maison à la date donnée
    """
    
    # Convertir la date de prédiction en datetime
    date_to_predict = pd.to_datetime(date_to_predict)
    
    # Charger le modèle
    model_fit = load_model()

    # Récupérer les données d'entraînement pour la prédiction
    month_avg = model_fit.model.endog  # Accéder aux données originales pour la prédiction
    month_avg_dates = model_fit.model.data.orig_endog.index  # Utiliser l'index original pour obtenir les dates
    
    last_date_in_dataset = month_avg_dates[-1]  # Dernière date dans les données de l'index
    # Calculer le nombre de mois entre la dernière date et la date de prédiction
    diff = relativedelta(date_to_predict, last_date_in_dataset)
    steps_to_forecast = diff.years * 12 + diff.months
    
    # Effectuer la prédiction
    forecast = model_fit.get_forecast(steps=steps_to_forecast)
    
    # Récupérer le prix prédit
    predicted_price = forecast.predicted_mean.iloc[0]  # Première prévision

    # Convertir predicted_price en une valeur scalaire si nécessaire
    if isinstance(predicted_price, (np.ndarray, pd.Series)):
        predicted_price = predicted_price.item()  # Convertir ndarray ou pd.Series en valeur scalaire

    # Convertir current_price (Decimal) en float
    current_price_float = float(current_price)

    # Calculer le prix à la date de prédiction
    predicted_price_at_date = current_price_float * (predicted_price / month_avg[-1])

    # Convertir predicted_price_at_date en une valeur scalaire si nécessaire
    if isinstance(predicted_price_at_date, (np.ndarray, pd.Series)):
        predicted_price_at_date = predicted_price_at_date.item()

    return predicted_price_at_date





# Vue API pour la prédiction
@api_view(['POST'])
def predict_price_api(request):
    """
    Prédit le prix d'une maison basé sur le prix actuel et la date de prédiction via une API.
    """
    # Serializer pour valider les données de la requête
    serializer = HousePricePredictionSerializer(data=request.data)
    
    if serializer.is_valid():
        current_price = serializer.validated_data['current_price']
        date_to_predict = serializer.validated_data['date_to_predict']
        
        # Calculer la prédiction du prix
        predicted_price = predict_house_price(current_price, date_to_predict)
        
        return Response({'predicted_price': predicted_price}, status=200)
    
    return Response(serializer.errors, status=400)
