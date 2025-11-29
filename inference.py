import numpy as np
import sys
import os
conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
os.environ['PROJ_LIB'] = os.path.join(conda_prefix, 'share', 'proj')
os.environ['GDAL_DATA'] = os.path.join(conda_prefix, 'share', 'gdal')
import geopandas as gpd
import pandas as pd
from shapely import wkt
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from utils import RedLluviaPipeline
from utils import aplicar_imputacion_numerica_guardada, codificador_ciclico,imputador_categorico_produccion
from utils import MESES, ESTACIONES, STATIONS_POINTS, DIRECCIONES_VIENTO

preprocesadores = joblib.load("preprocesadores.pkl")
scaler = preprocesadores['scaler']
ohe = preprocesadores['ohe']
encoder = preprocesadores['encoder']

dataTest = pd.read_csv('input.csv')
regiones_8 = gpd.read_file('./Shapefiles/NRM_clusters.shp')

col_num = dataTest.select_dtypes(include=np.number).columns.to_list()
dataTest['Date'] = pd.to_datetime(dataTest['Date'])
dataTest['Mes'] = dataTest['Date'].dt.month #type: ignore
dataTest['Temporada'] = dataTest['Mes'].apply(lambda x: ESTACIONES[x])

for col in col_num:
    dataTest = aplicar_imputacion_numerica_guardada(dataTest, col)
for cat in ['WindDir9am', 'WindDir3pm', 'WindGustDir']:
    dataTest = imputador_categorico_produccion(dataTest, cat)
    dataTest = codificador_ciclico(dataTest, f'{cat}_imputada', DIRECCIONES_VIENTO)


#Codificacion cíclica de meses
dataTest = codificador_ciclico(dataTest, 'Mes', MESES) #type: ignore

#Codificación e imputado de RainToday
dataTest['RainToday'] = dataTest['RainToday'].map({'Yes': 1, 'No': 0})
condicion_lluvia_positiva = (dataTest['RainToday'].isna()) & (dataTest['Rainfall_imputada'] >= 1)
dataTest.loc[condicion_lluvia_positiva, 'RainToday'] = 1
condicion_lluvia_negativa = (dataTest['RainToday'].isna()) & (dataTest['Rainfall_imputada'] < 1)
dataTest.loc[condicion_lluvia_negativa, 'RainToday'] = 0

#Mapeo de 'Location' a coordenadas espaciales para el spatial join
dataTest['gpd_coordenadas'] = dataTest['Location'].map(lambda x: STATIONS_POINTS[x])
dataTest['gpd_coordenadas'] = dataTest['gpd_coordenadas'].apply(wkt.loads) #type:ignore
geodataTest = gpd.GeoDataFrame(dataTest, geometry='gpd_coordenadas', crs="EPSG:4326")
regiones_8 = regiones_8.to_crs(geodataTest.crs) #type:ignore
dataTest = gpd.sjoin(geodataTest, regiones_8, how='left', predicate='within')
dataTest = dataTest.drop(columns=['index_right', 'OBJECTID', 'Shape_Leng', 'Shape_Area', 'code'], axis=1)
#Manejo del caso no contemplado - NorfolkIsland es Offshore
dataTest.loc[dataTest['Location'] == "NorfolkIsland", 'label'] = 'Offshore'
label = ohe.transform(dataTest[['label']]) 
label_df = pd.DataFrame(label, columns=ohe.get_feature_names_out(['label']))
dataTest.reset_index(drop=True, inplace=True)
dataTest = pd.concat([dataTest, label_df], axis=1)
temp = encoder.transform(dataTest[['Temporada']])
temp_df = pd.DataFrame(temp, columns=encoder.get_feature_names_out(['Temporada']))
dataTest.reset_index(drop=True, inplace=True)
dataTest = pd.concat([dataTest, temp_df], axis=1)

columnas_numericas_imputadas = ['MinTemp_imputada', 'MaxTemp_imputada',
       'Rainfall_imputada', 'Evaporation_imputada', 'Sunshine_imputada',
       'WindGustSpeed_imputada', 'WindSpeed9am_imputada',
       'WindSpeed3pm_imputada', 'Humidity9am_imputada', 'Humidity3pm_imputada',
       'Pressure9am_imputada', 'Pressure3pm_imputada', 'Cloud9am_imputada',
       'Cloud3pm_imputada', 'Temp9am_imputada', 'Temp3pm_imputada']
dataTest[columnas_numericas_imputadas] = scaler.transform(dataTest[columnas_numericas_imputadas])

dataTest = dataTest.drop(columns=col_num, axis=1)
dataTest = dataTest.drop(columns=['label','gpd_coordenadas','Temporada','Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'], axis=1)
modelo_cargado = joblib.load('modelo_red_neuronal.pkl')
prediccion = modelo_cargado.predict(dataTest)
probabilidades = modelo_cargado.predict_proba(dataTest)
for indice, probabilidad in enumerate(probabilidades):
    if prediccion[indice]:
        print(f"Para la observación {indice} la prediccion del modelo es 'Lloverá'")
    else:
        print(f"Para la observación {indice} la prediccion del modelo es 'No Lloverá'")
    print(f"La probabilidad de que llueva es: {probabilidad['clase_1_yes']*100}%")

print(f"El umbral de desición para determinar lluvia es: {modelo_cargado.best_threshold_:.2f}")