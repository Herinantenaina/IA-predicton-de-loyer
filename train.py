import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
import folium
from streamlit_folium import st_folium
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv("original.csv")

#-------Remplace les NAN numriques par la mediane de leur colonne
df['superficie'] = df['superficie'].fillna(df['superficie'].median())

df['nombre_de_chambres'] = df['nombre_de_chambres'].fillna(df['nombre_de_chambres'].median())
# nombre_de_chambres
#-------Remplace les NAN categorielles par la valeur plus frequente
for col in ['quartier', 'douches_wc', 'type_d_acces', 'meublé', 'état_général']:
    df[col] = df[col].fillna(df[col].mode()[0])

#-----Numerisation des categories
df['meublé'] = df['meublé'].astype('category').cat.codes
df['douches_wc'] = df['douches_wc'].astype('category').cat.codes

etat_map = {"mauvais": 0, "moyen": 1, "bon": 2}
df["état_général"] = df["état_général"].map(etat_map)

# df = pd.get_dummies(df, columns=['quartier'], prefix='quartier') 
quartier_map = {
    '67ha': 0,
    'Ambanidia': 1,
    'Amboditsiry': 2,
    'Ambohimanarina': 3,
    'Ambohitrakely': 4,
    'Ambolokandrana': 5,
    'Ampasanimalo': 6,
    'Andoharanofotsy': 7,
    'Antsobolo': 8,
    'Bybass': 9,
    'Bypass': 10,
    'Iavoloha': 11,
    'Ifarihy': 12,
    'Mahamasina': 13,
    'Mausole': 14,
    'Tanjombato': 15
}
df['quartier'] = df['quartier'].map(quartier_map)

acces_map = {"sans": 0, "moto": 1, "voiture": 2}
df["type_d_acces"] = df["type_d_acces"].map(acces_map)

#-------Creation variables derives
#Confort
df['confortable'] = ((df['meublé'] == 1) & (df['état_général'] == 2)).astype(int)

#Espace ou trop de chambre
df['espace'] = (df['superficie'] / df['nombre_de_chambres']).round(2)



# Suppression des variables trop corrole
correlation = df.corr()

upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))

delete =  [col for col in upper.columns if any(upper[col] > 0.9)]

df = df.drop(columns=delete)

df.to_csv('data.csv', index=False)


# ------------Standardisation et Normalisation
# Standard
x = pd.read_csv('data.csv')

cols = x.select_dtypes(include=['float64', 'int64']).columns

scaler = StandardScaler()

x_standard = scaler.fit_transform(x[cols])

x_standard = pd.DataFrame(x_standard, columns=cols)

x_standard.to_csv('standard.csv', index=False)

# Normalisation
x = pd.read_csv('data.csv')

scaler = MinMaxScaler()

x_normal =  scaler.fit_transform(x[cols])

x_normal = pd.DataFrame(x_normal, columns=cols)

x_normal.to_csv('normal.csv', index=False)

X = df.drop(columns='loyer_mensuel')

y = df['loyer_mensuel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

joblib.dump(model, 'modele_regression.joblib')

y_pred = model.predict(X_test)

# Partie 3
# Backend Elimination: 
# Suppression des variables(features) inutiles(selon leur impact sur le model)(r2 eto)

features = list(X.columns)
best_r2 = -1
improved = True

while improved and len(features) > 1:
    improved = False
    temp_r2 = best_r2
    worst_feature = None
    
    for feature in features:
        trial_features = [f for f in features if f != feature]
        model = LinearRegression().fit(X_train[trial_features], y_train)
        y_pred = model.predict(X_test[trial_features])
        r2 = r2_score(y_test, y_pred)
        
        if r2 > temp_r2:
            temp_r2 = r2
            worst_feature = feature
            improved = True
    
    if improved:
        features.remove(worst_feature)
        best_r2 = temp_r2

# Upgrage leh modele
X = df[features]
y = df['loyer_mensuel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

joblib.dump(model, 'modele_regression.joblib')

# RFE: meme methode que backend mais selon l'importance de la variable calculee par le modele
rfe = RFE(estimator=model, n_features_to_select=5)


rfe.fit(X_train, y_train)

selected_features = X.columns[rfe.support_]

X_train_rfe = X_train[selected_features]
X_test_rfe = X_test[selected_features]

model_final = LinearRegression()
model_final.fit(X_train_rfe, y_train)

y_pred = model_final.predict(X_test_rfe)
print("MSE :", mean_squared_error(y_test, y_pred))
print("R² :", r2_score(y_test, y_pred))

# Sauvegarde (optionnel)
joblib.dump(model_final, 'modele_backend&rfe_optimise.joblib')


# Partie 4

model = joblib.load('modele_backend&rfe_optimise.joblib')

# features = ['quartier', 'nombre_de_chambres', 'douches_wc', 'meublé', 'espace']

features = X.columns

if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "quartier" not in st.session_state:
    st.session_state.quartier = None

st.title("Prédiction du loyer mensuel à Antananarivo")

with st.form("form_inputs"):
    st.write("Entrez les caractéristiques du logement :")
    
    
    quartier = st.selectbox("Quartier", 
                           options=["Ambanidia", "Andoharanofotsy", "Mahamasina", "Tanjombato", "67ha", "Bypass", "Ambohimanarina"])
    
    nombre_de_chambres = st.number_input("Nombre de chambres", min_value=1, max_value=10, value=2)
    douches_wc = st.selectbox("Douche/WC (intérieur=1, extérieur=0)", options=[1, 0])
    meuble = st.selectbox("Meublé (oui=1, non=0)", options=[1, 0])
    superficie = st.number_input("Superficie (m²)", min_value=10, max_value=500, value=50)
    
    espace = round(superficie / nombre_de_chambres, 2)
    
    submitted = st.form_submit_button("Prédire le loyer") 

if submitted:
    quartier_map = {
        '67ha': 0,
        'Ambanidia': 1,
        'Ambohimanarina': 3,
        'Andoharanofotsy': 7,
        'Bypass': 10,
        'Mahamasina': 13,
    }  

    input_data = pd.DataFrame({
    'quartier': [quartier_map[quartier]],
    'nombre_de_chambres': [nombre_de_chambres],
    'douches_wc': [douches_wc],
    'meublé': [meuble],
    'espace': [espace]
    })



    st.session_state.prediction = model.predict(input_data)[0]
    st.session_state.quartier = quartier


if st.session_state.prediction is not None:
    st.success(f"Le loyer mensuel prédit est : {st.session_state.prediction:.0f} Ar")
    
    st.subheader("Poids des variables dans le modèle")
    coef_df = pd.DataFrame({
        'Variable': features,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    st.table(coef_df)

    st.subheader("Localisation approximative sur la carte")

    quartier_coords = {
        "67ha": [-18.8895, 47.5359],
        "Ambanidia": [-19.053503, 47.728111],
        "Ambohimanarina": [-18.8671, 47.5167],
        "Andoharanofotsy": [-18.9737, 47.5430],
        "Bypass": [-18.8865, 47.5641],
        "Mahamasina": [-18.9130, 47.5147],
        "Tanjombato": [-18.9629, 47.5290]
    }

    coord = quartier_coords.get(st.session_state.quartier, [-18.8792, 47.5079])  # fallback centre

    m = folium.Map(location=coord, zoom_start=13)

    folium.Marker(
        location=coord,
        popup=f"Loyer prédit : {st.session_state.prediction:.0f} MGA\nQuartier : {st.session_state.quartier}",
        tooltip="Votre logement",
        icon=folium.Icon(color='red', icon='home')
    ).add_to(m)

    st_folium(m, width=700, height=450)
