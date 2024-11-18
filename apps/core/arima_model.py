import joblib
import pandas as pd

# Charger les modèles pré-enregistrés
models = joblib.load('C:/Users/WIN/Desktop/Model/real-estate-prediction-project/apps/core/model/sarima_models.pkl')
best_para = joblib.load('C:/Users/WIN/Desktop/Model/real-estate-prediction-project/apps/core/model/best_parameters.pkl')

# Fonction de prédiction pour une maison donnée
def predict_house_value(zipcode, current_price, months_ahead=12):
    # Charger les données pour le code postal donné
    df = pd.read_csv('C:/Users/WIN/Desktop/Model/real-estate-prediction-project/apps/core/data/zillow_data.csv')
    
    # Vérifier les données et filtrer les valeurs invalides
    dfm = df[pd.to_datetime(df['Month'], errors='coerce').notna()]
    
    # Transformer les données en format long et les ajuster pour les séries temporelles
    dfm = pd.melt(dfm, id_vars=["RegionName"], var_name="Month", value_name="MeanValue")
    dfm['Month'] = pd.to_datetime(dfm['Month'], format='%Y-%m')
    dfm.set_index('Month', inplace=True)
    
    # Sélectionner les données du zipcode spécifié
    zipcode_df = dfm[dfm['RegionName'] == zipcode]
    
    # Ajouter le prix actuel comme entrée exogène
    zipcode_df['Current_Price'] = current_price
    
    # Récupérer le modèle SARIMA correspondant au zipcode
    model = models[zipcode]
    
    # Effectuer une prédiction avec l'exogène (prix actuel)
    exog = zipcode_df[['Current_Price']].tail(months_ahead)
    forecast = model.get_forecast(steps=months_ahead, exog=exog)
    
    # Obtenir la prédiction future pour la maison
    forecast_mean = forecast.predicted_mean[-1]  # Dernière prédiction de la période
    return forecast_mean

# Exemple de prédiction pour le zipcode 11220 avec un prix actuel de 500,000
zipcode_to_predict = 11220
current_price = 500000
predicted_value = predict_house_value(zipcode_to_predict, current_price, months_ahead=12)
print(f"Predicted value for zipcode {zipcode_to_predict} in 12 months (based on current price {current_price}): {predicted_value}")
