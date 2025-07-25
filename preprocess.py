import pandas as pd
import numpy as np

def normalizar_nombres(df):
    df['Competición'] = df['Competición'].str.strip().str.title()
    df['Local'] = df['Local'].str.strip().str.title()
    df['Visitante'] = df['Visitante'].str.strip().str.title()
    return df

def preparar_datos_tabular(df):
    # Variables simples para Random Forest/XGBoost
    df['Resultado'] = np.where(df['Goles Local'] > df['Goles Visitante'], 2,
                        np.where(df['Goles Local'] < df['Goles Visitante'], 0, 1))
    X = df[['Goles Local', 'Goles Visitante']]  # Puedes agregar más features
    y = df['Resultado']
    return X, y

def preparar_datos_secuencial(df, secuencia=5):
    # Prepara secuencias para LSTM
    equipos = pd.unique(df[['Local', 'Visitante']].values.ravel('K'))
    X, y = [], []
    for equipo in equipos:
        partidos = df[(df['Local'] == equipo) | (df['Visitante'] == equipo)].sort_values('Fecha')
        goles = partidos[['Goles Local', 'Goles Visitante']].values
        resultado = np.where(partidos['Goles Local'] > partidos['Goles Visitante'], 2,
                      np.where(partidos['Goles Local'] < partidos['Goles Visitante'], 0, 1))
        for i in range(len(goles) - secuencia):
            X.append(goles[i:i+secuencia])
            y.append(resultado[i+secuencia])
    return np.array(X), np.array(y)
