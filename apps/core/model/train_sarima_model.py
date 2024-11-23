import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle

# Charger les données
data = 'C:/Users/WIN/Desktop/IA_project/real-estate-prediction-project/apps/core/data/zillow_data.csv'
df = pd.read_csv(data, header=0, parse_dates=True)

# Transformer les données pour regrouper les prix
def Transform(dataframe):
    # Exclure 'RegionName' et d'autres colonnes non date dans value_vars
    date_columns = [c for c in dataframe.columns if c not in ['RegionName', 'RegionID', 'City', 'State', 'Metro', 'CountyName', 'SizeRank', 'District']]  # Ajouter d'autres colonnes ici
    melted = pd.melt(dataframe, id_vars=['RegionName'], var_name='Month', value_name='MeanPrice',
                     value_vars=date_columns)
    melted['Month'] = pd.to_datetime(melted['Month'], format='%Y-%m')
    melted = melted.dropna(subset=['MeanPrice'])
    return melted

# Appliquer la transformation
df_transformed = Transform(df)

# Filtrer les données pour n'utiliser que celles de 2011 et après
df_transformed = df_transformed[df_transformed['Month'] >= '2011-01-01']

# Calculer une moyenne globale des prix par mois
df_transformed.set_index('Month', inplace=True)
monthly_avg_prices = df_transformed.groupby('Month')['MeanPrice'].mean()

# Entraîner le modèle SARIMA sur les données globales
p, d, q = 1, 1, 1
P, D, Q, S = 1, 1, 1, 12

# Créer et entraîner le modèle
model = SARIMAX(monthly_avg_prices, order=(p, d, q), seasonal_order=(P, D, Q, S))
model_fit = model.fit(disp=False)

# Sauvegarder le modèle SARIMA global
model_path = 'C:/Users/WIN/Desktop/IA_project/real-estate-prediction-project/apps/core/model/sarima_models_global.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_fit, f)

print("Modèle global SARIMA entraîné et sauvegardé.")
