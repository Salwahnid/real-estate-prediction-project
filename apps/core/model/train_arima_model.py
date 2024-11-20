import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle

# Charger les données
data = 'C:/Users/WIN/Desktop/IA_project/real-estate-prediction-project/apps/core/data/zillow_data.csv'  # Remplacez par votre chemin de fichier
df = pd.read_csv(data, header=0, parse_dates=True)
area = pd.read_csv('C:/Users/WIN/Desktop/IA_project/real-estate-prediction-project/apps/core/data/manhattan_brooklyn_zip.csv')  # Remplacez par votre chemin de fichier
ny = df.loc[df['City'] == 'New York']
ny1 = pd.merge(ny, area, how='left', on='RegionName')

ny2 = ny1[ny1['District'].notnull()]
Brooklyn = ny2.loc[ny2['District'] == 'Brooklyn']

# Préparer les données
columns_to_drop = Brooklyn[['RegionID', 'City', 'State', 'Metro', 'CountyName', 'SizeRank', 'District']]
brooklyn = Brooklyn.drop(labels=columns_to_drop, axis='columns')

def Transform(dataframe):
    melted = pd.melt(dataframe, id_vars=['RegionName'], var_name='Month', value_name='MeanPrice')
    melted['Month'] = pd.to_datetime(melted['Month'], format='%Y-%m')
    melted = melted.dropna(subset=['MeanPrice'])
    return melted

brooklyn_data = Transform(brooklyn)
brooklyn_data.set_index(keys='Month', inplace=True)

# Utiliser les données de 2011 en avant pour la prédiction
df_2011 = brooklyn_data['2011':]
month_avg = df_2011.groupby(by=['Month']).mean()
month_avg = month_avg[['MeanPrice']]

# Appliquer un modèle SARIMA sur les données
p, d, q = 1, 1, 1
P, D, Q, S = 1, 1, 1, 12

# Créer le modèle SARIMA
model = SARIMAX(month_avg['MeanPrice'], order=(p, d, q), seasonal_order=(P, D, Q, S))
model_fit = model.fit(disp=False)

# Sauvegarder le modèle dans un fichier .pkl
with open('C:/Users/WIN/Desktop/IA_project/real-estate-prediction-project/apps/core/model/sarima_model.pkl', 'wb') as f:
    pickle.dump(model_fit, f)

print("Modèle formé et sauvegardé avec succès.")