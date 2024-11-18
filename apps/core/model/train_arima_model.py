import pandas as pd
import numpy as np
import statsmodels.api as sm
import joblib  # Pour sauvegarder le modèle
from itertools import product

# Charger les données
df = pd.read_csv('C:/Users/WIN/Desktop/Model/real-estate-prediction-project/apps/core/data/zillow_data.csv')

# Afficher les premières lignes pour vérifier la structure du DataFrame
print(df.head())

# Convertir les colonnes de dates (comme '2017-11', '2017-12', ...) en une colonne 'Month'
dfm = pd.melt(df, id_vars=["RegionID", "RegionName", "City", "State", "Metro", "CountyName"], 
              var_name="Month", value_name="MeanValue")

# Vérifier les premières lignes après le melt
print(dfm.head())

# Filtrer pour ne garder que les lignes où 'Month' contient des dates au format '%Y-%m'
dfm = dfm[dfm['Month'].str.match(r'\d{4}-\d{2}')]

# Convertir la colonne 'Month' en format datetime
dfm['Month'] = pd.to_datetime(dfm['Month'], format='%Y-%m')

# Vérifier que la conversion a bien fonctionné
print(dfm.head())

# Supprimer les lignes où les dates sont invalides (NaT)
dfm = dfm[pd.to_datetime(dfm['Month'], errors='coerce').notna()]

# Assurez-vous que 'RegionName' est unique pour chaque série temporelle
dfm.set_index('Month', inplace=True)

# Ajouter une colonne 'Current_Price' basée sur la dernière valeur de chaque région
dfm['Current_Price'] = dfm.groupby('RegionName')['MeanValue'].transform('last')

# Paramètres pour SARIMA
p = d = q = range(0, 2)
pdq = list(product(p, d, q))
pdqs = [(x[0], x[1], x[2], 12) for x in product(p, d, q)]

# Créer des DataFrames pour chaque code postal
zip_dfs = []
zip_list = dfm['RegionName'].unique()

for x in zip_list:
    zip_dfs.append(pd.DataFrame(dfm[dfm['RegionName'] == x][['MeanValue', 'Current_Price']].copy()))

# Ajuster les modèles SARIMA
def fit_sarima_models(zip_dfs, zip_list, pdq, pdqs):
    ans = []
    models = []
    
    for df, name in zip(zip_dfs, zip_list):
        for para1 in pdq:
            for para2 in pdqs:
                try:
                    # Inclure 'Current_Price' comme variable exogène
                    exog = df[['Current_Price']]
                    model = sm.tsa.statespace.SARIMAX(df['MeanValue'],
                                                      order=para1,
                                                      seasonal_order=para2,
                                                      exog=exog,
                                                      enforce_stationarity=False,
                                                      enforce_invertibility=False)
                    output = model.fit()
                    ans.append([name, para1, para2, output.aic])
                    models.append(output)
                except:
                    continue
    
    result = pd.DataFrame(ans, columns=['name', 'pdq', 'pdqs', 'AIC'])
    best_para = result.loc[result.groupby("name")["AIC"].idxmin()]
    
    return best_para, models

# Ajuster les modèles et obtenir les meilleurs paramètres
best_para, models = fit_sarima_models(zip_dfs, zip_list, pdq, pdqs)

# Sauvegarder les modèles et les meilleurs paramètres
joblib.dump(models, 'C:/Users/WIN/Desktop/Model/real-estate-prediction-project/apps/core/model/sarima_models.pkl')
joblib.dump(best_para, 'C:/Users/WIN/Desktop/Model/real-estate-prediction-project/apps/core/model/best_parameters.pkl')

print("Modèles SARIMA et meilleurs paramètres enregistrés avec succès.")
