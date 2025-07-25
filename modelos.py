from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def entrenar_random_forest(X, y):
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    return modelo

def entrenar_xgboost(X, y):
    modelo = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    modelo.fit(X, y)
    return modelo

def entrenar_lstm(X, y):
    n_timesteps, n_features = X.shape[1], X.shape[2]
    model = Sequential()
    model.add(LSTM(32, input_shape=(n_timesteps, n_features)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

def ensemble_predict(modelo_rf, modelo_lstm, X_tabular, X_seq):
    proba_rf = modelo_rf.predict_proba(X_tabular)
    proba_lstm = modelo_lstm.predict(X_seq)
    proba_final = 0.5 * proba_rf + 0.5 * proba_lstm
    return proba_final
