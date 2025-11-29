import numpy as np
import pandas as pd
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

class RedLluviaPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.0005, epochs=100, batch_size=128,
                 capas_ocultas=[64, 64, 32], dropout_rate=0.25, random_state=None):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.capas_ocultas = capas_ocultas
        self.dropout_rate = dropout_rate
        #Threshold óptimo calculado a partir de la precision recall curve
        self.best_threshold_ = 0.5
        self.random_state = random_state
        #Modelo fiteado
        self.model_ = None
        #Necesario para graficar el error de entrenamiento vs. validación
        self.history_ = None
        #Clases aprendidas por el modelo
        self.classes_ = None

    def fit(self, X, y):
        """
        Entrena el modelo y encuentra el mejor threshold de que maximiza el f1_score.
        """
        #Se setea el random state en caso de ser proporcionado
        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
            np.random.seed(self.random_state)
            random.seed(self.random_state)
        #Cantidad de columnas (variables de entrada)
        input_dim = X.shape[1]
        #Clases únicas detectadas
        self.classes_ = np.unique(y)
        #Train-test split del conjunto de entrenamiento, podría considerarse una split para validación
        #útil para el fit del modelo y para el cálculo del mejor umbral para maximizar el f1_score
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        #Balanceo de pesos
        pesos = compute_class_weight(
            class_weight='balanced',
            classes=self.classes_,
            y=y_train
        )
        class_weights_dict = dict(enumerate(pesos))
        self._construir_modelo(input_dim)
        #Callbacks para early stopping (paciencia alta) para maximizar el f1-score
        #restore_best_weights permite 'guardar' los mejores pesos encontrados hasta el momento
        early_stopper = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=0)
        #Fit del modelo
        self.history_ = self.model_.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[early_stopper, reduce_lr],
            class_weight=class_weights_dict
        )

        #Una vez fiteado el modelo, hacemos las predicciones (utilizando el método predict del modelo de keras)
        #no confundirse con el método predict de la clase 'RedLluviaPipeline'
        val_probs = self.model_.predict(X_test, verbose=0)

        #Calculamos el umbral óptimo para F1
        precision, recall, thresholds = precision_recall_curve(y_test, val_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)

        mejor_f1_index = np.argmax(f1_scores)

        #Seteamos el umbral optimo como variable interna de la clase, si y solo si es un índice válido
        if mejor_f1_index < len(thresholds):
            self.best_threshold_ = thresholds[mejor_f1_index]
        else:
            self.best_threshold_ = 0.5

        return self
    def predict(self, X):
        """
        Método que realiza predicciones según el umbral óptimo
        """
        if self.model_ is None:
            raise RuntimeError("El modelo no ha sido entrenado todavía.")
        #Predict del modelo de keras
        probabilidades = self.model_.predict(X, verbose=0)
        #Utilizamos el mejor umbral calculado del fit para nuestras predicciones
        return (probabilidades > self.best_threshold_).astype(int).flatten()

    def predict_proba(self, X):
        """
        Devuelve una lista de diccionarios con las probabilidades para cada clase.
        Ejemplo: [{'negativa': 0.1, 'positiva': 0.9}, ...]
        El índice de la lista se corresponde fila a fila con el orden del dataframe.
        """
        if self.model_ is None:
            raise RuntimeError("El modelo no ha sido entrenado todavía.")

        # Obtenemos la probabilidad de la clase 1
        prob_positiva = self.model_.predict(X, verbose=0).flatten()
        
        # Obtenemos la probabilidad de la clase 0 = 1 - P('clase 1')
        prob_negativa = 1 - prob_positiva

        # Creamos una lista de diccionarios con las probabilidades
        resultados = []
        for p_neg, p_pos in zip(prob_negativa, prob_positiva):
            resultados.append({
                "clase_0_no": round(float(p_neg), 2),
                "clase_1_yes": round(float(p_pos), 2)
            })
            
        return resultados

    def _construir_modelo(self, input_dim):
        """Método interno de la clase para construir el modelo según los parámetros de inicialización"""
        
        #Modelo secuencial
        self.model_ = models.Sequential()
        #Para cada capa oculta seteamos una semilla. 
        #Si es la primera, definimos también la capa de entrada.
        #Se asigna el dropout fijo a las siguientes capas ocultas
        for i, neuronas in enumerate(self.capas_ocultas):
            seed_capa = self.random_state + i if self.random_state else None
            if i == 0:
                self.model_.add(layers.Dense(neuronas, activation='relu', input_shape=(input_dim,)))
            else:
                self.model_.add(layers.Dense(neuronas, activation='relu'))
            self.model_.add(layers.Dropout(self.dropout_rate, seed=seed_capa))

        #Salida sigmoidea porque necesitamos predicción 0 o 1
        self.model_.add(layers.Dense(1, activation='sigmoid'))

        optimizador = optimizers.Adam(learning_rate=self.learning_rate)
        self.model_.compile(optimizer=optimizador,
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
    
    def _graficar_evolucion(self):
        """Método para graficar la evolución del entrenamiento"""
        if self.history_ is None:
            raise RuntimeError("El modelo no ha sido entrenado todavía.")
        hist = self.history_.history
        loss = hist['loss']
        val_loss = hist['val_loss']
        accuracy = hist['accuracy']
        val_accuracy = hist['val_accuracy']

        rango_epocas = range(1, len(loss) + 1)

        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(rango_epocas, loss, label='Error de entrenamiento')
        plt.plot(rango_epocas, val_loss, label='Error de validación')
        plt.title('Evolución de la Pérdida')
        plt.xlabel('Época')
        plt.ylabel('Pérdida (Binary Crossentropy)')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.plot(rango_epocas, accuracy, label='Train Accuracy')
        plt.plot(rango_epocas, val_accuracy, label='Val Accuracy')
        plt.title('Evolución del Accuracy')
        plt.xlabel('Época')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()
    
    def __getstate__(self):
        """
        Prepara el objeto para ser guardado con Pickle/Joblib.
        TensorFlow no es compatible con pickle, así que guardamos los pesos manualmente.
        """
        state = self.__dict__.copy()
        if self.model_ is not None:
            state['model_weights_'] = self.model_.get_weights()
            state['input_dim_'] = self.model_.input_shape[1]

        if 'model_' in state:
            del state['model_']

        if 'history_' in state:
            del state['history_']
            
        return state

    def __setstate__(self, state):
        """
        Restaura el objeto desde Pickle/Joblib.
        Reconstruye el modelo de Keras y le carga los pesos.
        """
        self.__dict__.update(state)

        if 'model_weights_' in state:
            self._construir_modelo(state['input_dim_'])
            self.model_.set_weights(state['model_weights_'])
            del self.model_weights_
            del self.input_dim_
        else:
            self.model_ = None

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