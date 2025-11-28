import numpy as np
import pandas as pd
import pickle
import os

#Mapeo de estaciones
ESTACIONES = {
    12: 'Verano', 1: 'Verano', 2: 'Verano',
    3: 'Otoño', 4: 'Otoño', 5: 'Otoño',
    6: 'Invierno', 7: 'Invierno', 8: 'Invierno',
    9: 'Primavera', 10: 'Primavera', 11: 'Primavera'
}
#Lista ordenada de meses para codificacion cíclica
MESES = np.arange(1,13)

#Mapeo de Location a Región
STATIONS_POINTS = {
    "Albury": "POINT(146.91583 -36.08056)",
    "BadgerysCreek": "POINT(150.75222 -33.87972)",
    "Cobar": "POINT(145.83194 -31.49972)",
    "CoffsHarbour": "POINT(153.11889 -30.30222)",
    "Moree": "POINT(149.83389 -29.46583)",
    "Newcastle": "POINT(151.75000 -32.91700)",
    "NorahHead": "POINT(151.57417 -33.28250)",
    "NorfolkIsland": "POINT(167.951564 -29.033794 )",
    "Penrith": "POINT(150.69450 -33.75150)",
    "Richmond": "POINT(150.78400 -33.58600)",
    "Sydney": "POINT(151.21000 -33.86778)",
    "SydneyAirport": "POINT(151.17722 -33.94611)",
    "WaggaWagga": "POINT(147.35900 -35.10900)",
    "Williamtown": "POINT(151.83444 -32.79500)",
    "Wollongong": "POINT(150.88300 -34.41700)",
    "Canberra": "POINT(149.12694 -35.29306)",
    "Tuggeranong": "POINT(149.08600 -35.40900)",
    "MountGinini": "POINT(148.95000 -35.47000)",
    "Ballarat": "POINT(143.85000 -37.55000)",
    "Bendigo": "POINT(144.28278 -36.75917)",
    "Sale": "POINT(147.05400 -38.10340)",
    "MelbourneAirport": "POINT(144.84479 -37.66371)",
    "Melbourne": "POINT(144.96306 -37.81361)",
    "Mildura": "POINT(142.15833 -34.18889)",
    "Nhil": "POINT(141.64722 -36.31083)",
    "Portland": "POINT(141.47111 -38.31806)",
    "Watsonia": "POINT(145.08300 -37.70800)",
    "Dartmoor": "POINT(141.28333 -37.93333)",
    "Brisbane": "POINT(153.02806 -27.46778)",
    "Cairns": "POINT(145.77330 -16.92330)",
    "GoldCoast": "POINT(153.40000 -28.01667)",
    "Townsville": "POINT(146.81580 -19.26220)",
    "Adelaide": "POINT(138.60072 -34.92866)",
    "MountGambier": "POINT(140.63444 -37.81028)",
    "Nuriootpa": "POINT(138.86500 -34.50100)",
    "Woomera": "POINT(136.81694 -31.14417)",
    "Albany": "POINT(117.88139 -35.02278)",
    "Witchcliffe": "POINT(115.09900 -34.02400)",
    "PearceRAAF": "POINT(116.01500 -31.66700)",
    "PerthAirport": "POINT(115.96700 -31.94030)",
    "Perth": "POINT(115.86000 -31.95000)",
    "SalmonGums": "POINT(121.64500 -32.98000)",
    "Walpole": "POINT(116.72300 -34.93100)",
    "Hobart": "POINT(147.32500 -42.88055)",
    "Launceston": "POINT(147.30440 -41.36050)",
    "AliceSprings": "POINT(133.87000 -23.70000)",
    "Darwin": "POINT(130.84111 -12.43806)",
    "Katherine": "POINT(132.26667 -14.46667)",
    "Uluru": "POINT(131.03611 -25.34500)"
}

#Lista ordenada de direcciones de viento para codificacion cíclica
DIRECCIONES_VIENTO = [
    'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'
]

def aplicar_imputacion_numerica_guardada(df_nuevo: pd.DataFrame, variable_a_imputar: str):
    '''
    Carga una lookup table con valores de mediana para la variable numérica a imputar, agrupados por 
    región y mes, junto con la mediana global obtenidas del conjunto de entrenamiento,
    para realizar la imputación de valores nulos.
    
    Argumentos necesarios:
    df: Dataframe donde se realizará la imputación
    variable_a_imputar: Identificador de la variable categórica a imputar
    
    Retorna:
    df - Dataframe con los valores nulos imputados
    '''
    carpeta_origen = 'imputadores'
    nombre_archivo = f"mediana_{variable_a_imputar}.pkl"
    ruta_completa = os.path.join(carpeta_origen, nombre_archivo)
    
    with open(ruta_completa, 'rb') as f:
        datos_cargados = pickle.load(f)
            
    lookup_medianas = datos_cargados['lookup']
    mediana_global = datos_cargados['global']
    variable_imputada = f'{variable_a_imputar}_imputada'

    def obtener_mediana(row):
        try:
            valor = lookup_medianas.loc[row['label'], row['Mes']]
            if not pd.isna(valor):
                return valor
        except KeyError:
            pass
        return mediana_global

    if variable_imputada not in df_nuevo.columns:
        df_nuevo[variable_imputada] = df_nuevo[variable_a_imputar].copy()
    
    valores_faltantes = df_nuevo[variable_imputada].isna()
    
    if valores_faltantes.any():
        valores_relleno = df_nuevo[valores_faltantes].apply(obtener_mediana, axis=1)
        df_nuevo.loc[valores_faltantes, variable_imputada] = valores_relleno
    
    df_nuevo[variable_imputada] = df_nuevo[variable_imputada].fillna(mediana_global)
    
    return df_nuevo

def imputador_categorico_produccion(df, variable_a_imputar):
    '''
    Carga la serie (.pkl) que contiene la probabilidad de aparición de la variable categórica a imputar como valor,
    y los valores únicos de la variable categórica como índice.
    Se utiliza para imputar los valores faltantes del dataframe provisto por input.csv.
    
    
    Argumentos necesarios:
    df: Dataframe donde se realizará la imputación
    variable_a_imputar: Identificador de la variable categórica a imputar
    
    Retorna:
    df - Dataframe con los valores nulos imputados
    '''
    np.random.seed(13)
    #Generación de de la ruta al imputador
    carpeta_origen = 'imputadores'
    nombre_archivo = f"frecuencia_{variable_a_imputar}.pkl"
    ruta_completa = os.path.join(carpeta_origen, nombre_archivo)
    #lectura del imputador
    with open(ruta_completa, 'rb') as f:
        frecuencias = pickle.load(f)

    variable_imputada = f'{variable_a_imputar}_imputada'
    
    if variable_imputada not in df.columns:
        df[variable_imputada] = df[variable_a_imputar].copy()
    
    #Referencia a filas con valores nulos (vector True/False)
    filas_a_imputar = df[variable_imputada].isnull()
    #Cantidad de nulos a imputar
    num_nulos = filas_a_imputar.sum()
    if num_nulos > 0:
        
        valores_imputados = np.random.choice(
            a=frecuencias.index,
            size=num_nulos,
            p=frecuencias.values
        )
        df.loc[filas_a_imputar, variable_imputada] = valores_imputados
    
    return df

def codificador_ciclico(df: pd.DataFrame, variable_a_codificar: str, categorias_ordenadas: list):
    """
    Realiza una codificación cíclica en una columna categórica ordenada.

    Argumentos necesarios:
    df - Dataframe donde se realizará la codificación
    variable_a_codificar - Identificador de la columna categórica a codificar cíclicamente
    categorías_ordenadas - Lista de categorías ordenada de manera que reflejen el orden cíclico
      ej: Meses - [1,2... 12] - El mes 12 precede al mes 1.

    Retorna:
    df- Dataframe con la columna codificada de manera cíclica, elimina la columna original
    """
    if variable_a_codificar not in df.columns:
        return df

    mapeo = {categoria: i for i, categoria in enumerate(categorias_ordenadas)}
    max_val = len(categorias_ordenadas)

    columna_mapeada = df[variable_a_codificar].map(mapeo)

    df[f'{variable_a_codificar}_sin'] = np.sin(2 * np.pi * columna_mapeada / max_val)
    df[f'{variable_a_codificar}_cos'] = np.cos(2 * np.pi * columna_mapeada / max_val)

    df.drop(columns=[variable_a_codificar], inplace=True)
    return df