
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

housing_toluca = pd.read_csv("avaluos_toluca_actualizado2026.csv")

st.write("# Valuación inmobiliaria — Toluca")

st.image("casatol.jfif", caption="Valuación inmobiliaria en Toluca")

st.header("Datos de la vivienda")

def user_input_features():
    delegacion = st.selectbox("Delegación:", housing_toluca["delegacion"].unique())
    superficie_terreno_m2 = st.number_input("Superficie de terreno (m²):", min_value=1.0, value=90.0)
    superficie_construccion_m2 = st.number_input("Superficie de construcción (m²):", min_value=1.0, value=90.0)
    recamaras = st.number_input("Recámaras:", min_value=0, max_value=10, value=3, step=1)
    banos = st.number_input("Baños:", min_value=0, max_value=10, value=2, step=1)
    estacionamientos = st.number_input("Estacionamientos:", min_value=0, max_value=10, value=2, step=1)
    niveles = st.number_input("Niveles:", min_value=1, max_value=10, value=2, step=1)
    antiguedad_anios = st.number_input("Antigüedad (años):", min_value=0, max_value=100, value=10, step=1)
    estado_conservacion = st.selectbox("Estado de conservación:", housing_toluca["estado_conservacion"].unique())
    uso_suelo = st.selectbox("Uso de suelo:", housing_toluca["uso_suelo"].unique())
    accesibilidad_vial = st.selectbox("Accesibilidad vial:", housing_toluca["accesibilidad_vial"].unique())
    distancia_centro_km = st.number_input("Distancia al centro (km):", min_value=0.0, value=5.0)
    distancia_vialidad_principal_km = st.number_input("Distancia a vialidad principal (km):", min_value=0.0, value=1.0)

    user_input_data = {
        "delegacion": delegacion,
        "superficie_terreno_m2": superficie_terreno_m2,
        "superficie_construccion_m2": superficie_construccion_m2,
        "recamaras": recamaras,
        "banos": banos,
        "estacionamientos": estacionamientos,
        "niveles": niveles,
        "antiguedad_anios": antiguedad_anios,
        "estado_conservacion": estado_conservacion,
        "uso_suelo": uso_suelo,
        "accesibilidad_vial": accesibilidad_vial,
        "distancia_centro_km": distancia_centro_km,
        "distancia_vialidad_principal_km": distancia_vialidad_principal_km
    }

    features = pd.DataFrame(user_input_data, index=[0])
    return features

df = user_input_features()

st.subheader("Datos ingresados")
st.write(df)

housing_toluca_features = ["delegacion", "superficie_terreno_m2", "superficie_construccion_m2", "recamaras", "banos", "estacionamientos", "niveles", "antiguedad_anios", "estado_conservacion", "uso_suelo", "accesibilidad_vial", "distancia_centro_km", "distancia_vialidad_principal_km"]

X = housing_toluca[housing_toluca_features]
X = pd.get_dummies(X)

df = pd.get_dummies(df)
df = df.reindex(columns=X.columns, fill_value=0)

Y = housing_toluca["precio"]

housing_model_toluca = DecisionTreeRegressor(random_state=0)
housing_model_toluca.fit(X, Y)

prediction = housing_model_toluca.predict(df)

st.subheader("Valor estimado de referencia")
st.write(f"${prediction[0]:,.2f} MXN")
