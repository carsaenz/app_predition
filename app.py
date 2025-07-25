import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = Flask(__name__)
app.secret_key = 'clave_secreta'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}

df_resultados = None
df_jugadores = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_columns(df, mapper):
    col_map = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for std_name, variants in mapper.items():
        for v in variants:
            v_lower = v.lower().strip()
            if v_lower in cols_lower:
                col_map[cols_lower[v_lower]] = std_name
                break
    df.rename(columns=col_map, inplace=True)
    return df

def normalizar_nombres_resultados(df):
    mapper = {
        'Competición': ['competición', 'competencia', 'competicion'],
        'Local': ['local'],
        'Visitante': ['visitante'],
        'Fecha': ['fecha'],
        'Goles Local': ['goles local', 'goles_local'],
        'Goles Visitante': ['goles visitante', 'goles_visitante'],
    }
    df = normalize_columns(df, mapper)
    df['Competición'] = df['Competición'].astype(str).str.strip().str.title()
    df['Local'] = df['Local'].astype(str).str.strip().str.title()
    df['Visitante'] = df['Visitante'].astype(str).str.strip().str.title()
    df['Goles Local'] = pd.to_numeric(df['Goles Local'], errors='coerce').fillna(0)
    df['Goles Visitante'] = pd.to_numeric(df['Goles Visitante'], errors='coerce').fillna(0)
    return df

def normalizar_nombres_jugadores(df):
    mapper = {
        'jugador': ['jugador'],
        'equipo': ['equipo'],
        'Competición': ['competición', 'competencia', 'competicion'],
        'goles': ['goles'],
        'asistencias': ['asistencias'],
        'partidos': ['partidos']
    }
    df = normalize_columns(df, mapper)
    df['jugador'] = df['jugador'].astype(str).str.strip().str.title()
    df['equipo'] = df['equipo'].astype(str).str.strip().str.title()
    df['Competición'] = df['Competición'].astype(str).str.strip().str.title()
    df['goles'] = pd.to_numeric(df['goles'], errors='coerce').fillna(0)
    df['asistencias'] = pd.to_numeric(df['asistencias'], errors='coerce').fillna(0)
    df['partidos'] = pd.to_numeric(df['partidos'], errors='coerce').fillna(0)
    return df

def prob_goleador(goles, partidos):
    if partidos > 0:
        lmbd = goles / partidos
        return round(100 * (1 - np.exp(-lmbd)), 2)
    return 0

def prob_asistencia(asistencias, partidos):
    if partidos > 0:
        lmbd = asistencias / partidos
        return round(100 * (1 - np.exp(-lmbd)), 2)
    return 0

def calcular_prob_jugador_con_gol_y_asistencia(equipo, jugador, liga_seleccionada):
    if jugador:
        datos = df_jugadores[(df_jugadores['equipo'] == equipo) & 
                             (df_jugadores['Competición'] == liga_seleccionada) & 
                             (df_jugadores['jugador'] == jugador)]
        if datos.empty:
            return None, None
        goles = datos['goles'].sum()
        asistencias = datos['asistencias'].sum()
        partidos = datos['partidos'].sum()
        return prob_goleador(goles, partidos), prob_asistencia(asistencias, partidos)
    return None, None

def preparar_datos_tabular(df):
    df['Resultado'] = np.where(df['Goles Local'] > df['Goles Visitante'], 2,
                              np.where(df['Goles Local'] < df['Goles Visitante'], 0, 1))
    X = df[['Goles Local', 'Goles Visitante']].fillna(0)
    y = df['Resultado'].astype(int)
    return X, y

def preparar_datos_secuencial(df, secuencia=5):
    df = df.sort_values('Fecha')
    equipos = pd.unique(df[['Local', 'Visitante']].values.ravel())
    X, y = [], []
    for equipo in equipos:
        partidos = df[(df['Local'] == equipo) | (df['Visitante'] == equipo)]
        goles = partidos[['Goles Local', 'Goles Visitante']].values
        resultados = np.where(partidos['Goles Local'] > partidos['Goles Visitante'], 2,
                             np.where(partidos['Goles Local'] < partidos['Goles Visitante'], 0, 1))
        for i in range(len(goles) - secuencia):
            X.append(goles[i:i+secuencia])
            y.append(resultados[i+secuencia])
    return np.array(X), np.array(y)

def entrenar_random_forest(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def entrenar_lstm(X, y):
    n_timesteps, n_features = X.shape[1], X.shape[2]
    model = Sequential()
    model.add(LSTM(32, input_shape=(n_timesteps, n_features)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

def ensemble_predict(model_rf, model_lstm, X_tab, X_seq):
    proba_rf = model_rf.predict_proba(X_tab)
    proba_lstm = model_lstm.predict(X_seq)
    proba_ensemble = 0.5 * proba_rf + 0.5 * proba_lstm
    return proba_ensemble

def goles_esperados_local_visitante(df, local, visitante):
    prom_goles_local_casa = df[df['Local'] == local]['Goles Local'].mean()
    prom_goles_visitante_fuera_encajados = df[df['Visitante'] == visitante]['Goles Local'].mean()
    lambda_local = np.mean([prom_goles_local_casa, prom_goles_visitante_fuera_encajados])

    prom_goles_visitante_fuera = df[df['Visitante'] == visitante]['Goles Visitante'].mean()
    prom_goles_local_casa_encajados = df[df['Local'] == local]['Goles Visitante'].mean()
    lambda_visitante = np.mean([prom_goles_visitante_fuera, prom_goles_local_casa_encajados])

    return round(lambda_local, 2), round(lambda_visitante, 2)

@app.route('/', methods=['GET', 'POST'])
def subir_archivos():
    global df_resultados, df_jugadores
    if request.method == 'POST':
        f_resultados = request.files.get('file_resultados')
        f_jugadores = request.files.get('file_jugadores')
        if not f_resultados or not f_jugadores:
            return "Debe subir ambos archivos."
        if allowed_file(f_resultados.filename) and allowed_file(f_jugadores.filename):
            path_r = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f_resultados.filename))
            path_j = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f_jugadores.filename))
            f_resultados.save(path_r)
            f_jugadores.save(path_j)
            df_resultados = pd.read_csv(path_r)
            df_resultados = normalizar_nombres_resultados(df_resultados)
            df_jugadores = pd.read_csv(path_j)
            df_jugadores = normalizar_nombres_jugadores(df_jugadores)
            return redirect(url_for('predicciones'))
        else:
            return "Solo archivos CSV permitidos."
    return render_template('upload.html')

@app.route('/equipos/<competicion>')
def obtener_equipos(competicion):
    if df_resultados is None:
        return jsonify([])
    competicion = competicion.strip().title()
    equipos_local = df_resultados[df_resultados['Competición'] == competicion]['Local'].unique()
    equipos_visitante = df_resultados[df_resultados['Competición'] == competicion]['Visitante'].unique()
    equipos = sorted(set(equipos_local) | set(equipos_visitante))
    return jsonify(equipos)

@app.route('/equipos_visitantes/<competicion>/<equipo_local>')
def obtener_equipos_visitantes(competicion, equipo_local):
    if df_resultados is None:
        return jsonify([])
    competicion = competicion.strip().title()
    equipo_local = equipo_local.strip().title()
    equipos_local = df_resultados[df_resultados['Competición'] == competicion]['Local'].unique()
    equipos_visitante = df_resultados[df_resultados['Competición'] == competicion]['Visitante'].unique()
    equipos = sorted(set(equipos_local) | set(equipos_visitante))
    equipos = [e for e in equipos if e != equipo_local]
    return jsonify(equipos)

@app.route('/jugadores', methods=['POST'])
def obtener_jugadores():
    global df_jugadores
    data = request.get_json()
    equipo = data.get('equipo', '').strip().title()
    competicion = data.get('competicion', '').strip().title()
    if df_jugadores is None:
        return jsonify([])
    jugadores = df_jugadores[(df_jugadores['equipo'] == equipo) & (df_jugadores['Competición'] == competicion)]['jugador'].unique()
    return jsonify(sorted(jugadores))

@app.route('/predicciones', methods=['GET', 'POST'])
def predicciones():
    global df_resultados, df_jugadores
    if df_resultados is None or df_jugadores is None:
        return redirect(url_for('subir_archivos'))

    ligas = sorted(df_resultados['Competición'].unique())
    equipos = []
    jugadores_local = []
    jugadores_visitante = []
    resultados = None
    liga_seleccionada = None
    equipo_local = None
    equipo_visitante = None
    jugador_local = None
    jugador_visitante = None

    if request.method == 'POST':
        liga_seleccionada = request.form.get('liga')
        equipo_local = request.form.get('equipo_local')
        equipo_visitante = request.form.get('equipo_visitante')
        jugador_local = request.form.get('jugador_local')
        jugador_visitante = request.form.get('jugador_visitante')

        if liga_seleccionada:
            equipos = sorted(df_resultados[df_resultados['Competición'] == liga_seleccionada]['Local'].unique())

        if equipo_local:
            jugadores_local = sorted(df_jugadores[
                (df_jugadores['equipo'] == equipo_local) & (df_jugadores['Competición'] == liga_seleccionada)
            ]['jugador'].unique())

        if equipo_visitante:
            jugadores_visitante = sorted(df_jugadores[
                (df_jugadores['equipo'] == equipo_visitante) & (df_jugadores['Competición'] == liga_seleccionada)
            ]['jugador'].unique())

        if liga_seleccionada and equipo_local and equipo_visitante:
            df_liga = df_resultados[df_resultados['Competición'] == liga_seleccionada]

            # Preparar datos para Random Forest y LSTM
            X_tab, y_tab = preparar_datos_tabular(df_liga)
            X_seq, y_seq = preparar_datos_secuencial(df_liga)

            modelo_rf = entrenar_random_forest(X_tab, y_tab)
            modelo_lstm = entrenar_lstm(X_seq, y_seq)

            X_pred_tab = np.array([[df_liga[df_liga['Local'] == equipo_local]['Goles Local'].mean() or 0,
                                    df_liga[df_liga['Visitante'] == equipo_visitante]['Goles Visitante'].mean() or 0]])

            X_pred_seq = X_seq[-1].reshape((1, X_seq.shape[1], X_seq.shape[2])) if len(X_seq) > 0 else np.zeros((1, 5, 2))

            proba = ensemble_predict(modelo_rf, modelo_lstm, X_pred_tab, X_pred_seq)[0]
            prob_victoria_local = proba[2]
            prob_empate = proba[1]
            prob_victoria_visitante = proba[0]

            goles_local, goles_visitante = goles_esperados_local_visitante(df_liga, equipo_local, equipo_visitante)

            prob_gol_local, prob_asi_local = calcular_prob_jugador_con_gol_y_asistencia(equipo_local, jugador_local, liga_seleccionada)
            prob_gol_visitante, prob_asi_visitante = calcular_prob_jugador_con_gol_y_asistencia(equipo_visitante, jugador_visitante, liga_seleccionada)

            prop1t, prop2t = 0.45, 0.55
            marcador_1t = f"{int(round(goles_local * prop1t))} - {int(round(goles_visitante * prop1t))}"
            marcador_2t = f"{int(round(goles_local * prop2t))} - {int(round(goles_visitante * prop2t))}"
            marcador_final = f"{int(round(goles_local))} - {int(round(goles_visitante))}"

            resultados = {
                'mensaje': f'Predicción avanzada para {liga_seleccionada}',
                'prob_victoria_local': round(prob_victoria_local * 100, 2),
                'prob_empate': round(prob_empate * 100,2),
                'prob_victoria_visitante': round(prob_victoria_visitante * 100, 2),
                'goles_esperados_local': goles_local,
                'goles_esperados_visitante': goles_visitante,
                'marcador_1t': marcador_1t,
                'marcador_2t': marcador_2t,
                'marcador_final': marcador_final,
                'jugador_local': jugador_local,
                'jugador_visitante': jugador_visitante,
                'prob_jugador_local_gol': prob_gol_local,
                'prob_jugador_local_asistencia': prob_asi_local,
                'prob_jugador_visitante_gol': prob_gol_visitante,
                'prob_jugador_visitante_asistencia': prob_asi_visitante,
            }

    return render_template('probabilidades.html',
                           ligas=ligas,
                           equipos=equipos,
                           jugadores_equipo_local=jugadores_local,
                           jugadores_equipo_visitante=jugadores_visitante,
                           resultados=resultados,
                           liga_seleccionada=liga_seleccionada,
                           equipo_local=equipo_local,
                           equipo_visitante=equipo_visitante,
                           jugador_local=jugador_local,
                           jugador_visitante=jugador_visitante)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

