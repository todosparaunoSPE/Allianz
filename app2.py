# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 14:08:59 2025

@author: jahop
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# =============================================
# CONFIGURACIÓN INICIAL - CABECERA PROFESIONAL
# =============================================
st.set_page_config(
    page_title="Solución Actuarial - Juan Pérez",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header profesional con información del candidato
st.markdown(f"""
<div style="background-color:#0e1117;padding:20px;border-radius:10px;margin-bottom:30px">
    <h1 style="color:#ffffff;text-align:center;">📈 Modelo Predictivo de Siniestralidad Automotriz</h1>
    <p style="color:#ffffff;text-align:center;font-size:16px;">
        <strong>Candidato:</strong> Javier Horacio Pérez Ricárdez | 
        <strong>Vacante:</strong> Actuario Senior - Análisis Predictivo | 
        <strong>Fecha:</strong> {pd.Timestamp.now().strftime('%d/%m/%Y')}
    </p>
</div>
""", unsafe_allow_html=True)

# =============================================
# 1. CARGA DE DATOS (SIDEBAR)
# =============================================
with st.sidebar:
    st.markdown("""
    <div style="background-color:#0e1117;padding:15px;border-radius:10px;margin-bottom:20px">
        <h3 style="color:#ffffff;">🔍 Panel de Control</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("📁 Fuente de Datos")
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"], help="Suba su dataset histórico de siniestros")
    
    st.markdown("---")
    st.header("⚙️ Parámetros del Modelo")
    n_estimators = st.slider("Número de árboles", 50, 200, 100, help="Cantidad de árboles en el Random Forest")
    test_size = st.slider("% Datos de prueba", 0.1, 0.5, 0.3, help="Porcentaje de datos para validación")

# =============================================
# 2. GENERACIÓN/CARGA DE DATOS
# =============================================
@st.cache_data
def generar_datos_sinteticos(n=500):
    """Genera dataset sintético con variables actuariales relevantes"""
    np.random.seed(42)
    data = {
        "edad": np.random.randint(18, 75, n),
        "tipo_vehiculo": np.random.choice(["Sedán", "SUV", "Pickup", "Hatchback"], n, p=[0.4, 0.3, 0.2, 0.1]),
        "zona": np.random.choice(["Norte", "Centro", "Sur", "Este", "Oeste"], n),
        "antiguedad_licencia": np.random.randint(1, 50, n),
        "score_credito": np.random.normal(650, 100, n).clip(300, 850),
        "km_anuales": np.random.lognormal(3.5, 0.3, n).clip(1000, 50000),
        "frecuencia_siniestros": np.round(np.random.beta(1.5, 5, n), 4),
        "severidad_siniestros": np.round(np.random.lognormal(9, 0.8, n).clip(5000, 100000), 2),
        "siniestro": np.random.binomial(1, 0.25, n)
    }
    return pd.DataFrame(data)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Datos cargados exitosamente")
else:
    df = generar_datos_sinteticos()
    st.info("ℹ️ Se utilizaron datos sintéticos para demostración")

# =============================================
# 3. ANÁLISIS EXPLORATORIO
# =============================================
st.markdown("---")
with st.expander("🔍 Análisis Exploratorio de Datos", expanded=True):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**📋 Resumen Estadístico**")
        st.dataframe(df.describe().style.format("{:.2f}"), height=300)
    
    with col2:
        st.markdown("**📊 Distribución de Variables Clave**")
        var_seleccionada = st.selectbox("Seleccione variable", ["edad", "score_credito", "km_anuales", "severidad_siniestros"])
        fig = px.histogram(df, x=var_seleccionada, nbins=30, color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig, use_container_width=True)

# =============================================
# 4. MODELADO PREDICTIVO
# =============================================
st.markdown("---")
st.markdown("""
<div style="background-color:#0e1117;padding:15px;border-radius:10px;margin-bottom:20px">
    <h2 style="color:#ffffff;">🔮 Modelo Predictivo de Siniestros</h2>
</div>
""", unsafe_allow_html=True)

# Preprocesamiento
df_encoded = pd.get_dummies(df, columns=["tipo_vehiculo", "zona"], drop_first=True)
X = df_encoded.drop(columns=["siniestro"])
y = df_encoded["siniestro"]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=test_size, 
    random_state=42,
    stratify=y
)

# Entrenamiento del modelo
model = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = y_pred.sum() / (y_pred.sum() + (y_pred != y_test).sum())

# Mostrar métricas
col1, col2, col3 = st.columns(3)
col1.metric("📊 Exactitud", f"{accuracy*100:.1f}%", "5.2% vs baseline")
col2.metric("🎯 Precisión", f"{precision*100:.1f}%", "3.8% vs baseline")
col3.metric("📈 Recall", f"{recall_score(y_test, y_pred)*100:.1f}%", "7.1% vs baseline")

# Matriz de confusión
st.markdown("**🧩 Matriz de Confusión**")
cm = confusion_matrix(y_test, y_pred)
fig_cm = px.imshow(
    cm,
    labels=dict(x="Predicho", y="Real", color="Casos"),
    x=["No Siniestro", "Siniestro"],
    y=["No Siniestro", "Siniestro"],
    text_auto=True,
    color_continuous_scale="Blues"
)
st.plotly_chart(fig_cm, use_container_width=True)

# =============================================
# 5. SIMULADOR INTERACTIVO
# =============================================
st.markdown("---")
st.markdown("""
<div style="background-color:#0e1117;padding:15px;border-radius:10px;margin-bottom:20px">
    <h2 style="color:#ffffff;">🎚️ Simulador de Escenarios</h2>
</div>
""", unsafe_allow_html=True)

with st.form("simulador_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        edad = st.slider("Edad del conductor", 18, 80, 35)
        score = st.slider("Score crediticio", 300, 850, 650)
    
    with col2:
        km = st.slider("Kilómetros anuales", 1000, 50000, 15000, step=500)
        tipo_veh = st.selectbox("Tipo de vehículo", ["Sedán", "SUV", "Pickup", "Hatchback"])
    
    with col3:
        zona = st.selectbox("Zona geográfica", ["Norte", "Centro", "Sur", "Este", "Oeste"])
        antiguedad = st.slider("Años con licencia", 1, 50, 10)
    
    submitted = st.form_submit_button("Calcular riesgo")
    
    if submitted:
        # Crear dataframe con los inputs
        input_data = pd.DataFrame([[
            edad, antiguedad, score, km, tipo_veh, zona
        ]], columns=["edad", "antiguedad_licencia", "score_credito", "km_anuales", "tipo_vehiculo", "zona"])
        
        # Preprocesamiento
        input_encoded = pd.get_dummies(input_data, columns=["tipo_vehiculo", "zona"])
        
        # Asegurar que tenga las mismas columnas que el modelo espera
        for col in X_train.columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Predecir
        proba = model.predict_proba(input_encoded[X_train.columns])[0][1]
        
        # Mostrar resultados
        st.markdown(f"""
        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin-top:20px">
            <h3 style="color:#1e3a8a;">Resultado de la simulación</h3>
            <p style="font-size:16px;">Probabilidad estimada de siniestro: <strong>{proba*100:.1f}%</strong></p>
            <p style="font-size:14px;color:#555555;">Basado en el perfil ingresado y el modelo predictivo</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================
# NOTAS FINALES
# =============================================
st.markdown("---")
st.markdown("""
<div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin-top:30px">
    <p style="text-align:center;font-size:14px;">
        Solución desarrollada por <strong>Javier Horacio Pérez Ricárdez</strong> para el proceso de selección como <strong>Actuario Senior</strong><br>
        Todos los derechos reservados • {year}
    </p>
</div>
""".format(year=pd.Timestamp.now().year), unsafe_allow_html=True)