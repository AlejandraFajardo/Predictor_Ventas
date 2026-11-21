# -*- coding: utf-8 -*-
"""
Predictor de Ventas para E-commerce — App (Fase 3)
Autora: Luz Dany Alejandra Fajardo Mora
Tecnologías: Python + Prophet + Streamlit
"""

import streamlit as st
import pandas as pd
from prophet import Prophet

# ----------------------------------------------------------------------
# Configuración de la página
# ----------------------------------------------------------------------
st.set_page_config(page_title="Predictor de Ventas", page_icon="📈", layout="centered")

st.title("📈 Predictor de Ventas para E-commerce")
st.write(
    "Esta herramienta predice las ventas futuras de una tienda online a partir de su "
    "historial, para ayudar a decidir **cuánto preparar** y no quedarse sin stock ni "
    "gastar de más."
)

# ----------------------------------------------------------------------
# 1) Cargar y limpiar los datos (se guarda en caché para no repetirlo)
# ----------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    url = "https://raw.githubusercontent.com/SMU-ExEd/python-for-analytics-data/main/online_retail.csv.gz"
    df = pd.read_csv(url)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]          # quitar devoluciones
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]              # solo ventas válidas
    df = df.dropna(subset=["Description"]).drop_duplicates()           # sin nulos ni duplicados
    df["Ingresos"] = df["Quantity"] * df["UnitPrice"]
    serie = df.groupby(df["InvoiceDate"].dt.date)["Ingresos"].sum().reset_index()
    serie.columns = ["ds", "y"]
    serie["ds"] = pd.to_datetime(serie["ds"])
    return serie

# ----------------------------------------------------------------------
# 2) Entrenar el modelo (también en caché: solo se entrena una vez)
# ----------------------------------------------------------------------
@st.cache_resource
def entrenar_modelo(serie):
    modelo = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    modelo.fit(serie)
    return modelo

with st.spinner("Cargando datos y entrenando el modelo... (solo la primera vez)"):
    serie = cargar_datos()
    modelo = entrenar_modelo(serie)

# ----------------------------------------------------------------------
# 3) Control: cuántas semanas predecir
# ----------------------------------------------------------------------
st.subheader("¿Cuánto quieres predecir?")
semanas = st.slider("Semanas hacia el futuro", min_value=1, max_value=8, value=4)
dias = semanas * 7

# ----------------------------------------------------------------------
# 4) Predecir
# ----------------------------------------------------------------------
futuro = modelo.make_future_dataframe(periods=dias)
prediccion = modelo.predict(futuro)

ultimo_dia = serie["ds"].max()
futuro_solo = prediccion[prediccion["ds"] > ultimo_dia].copy()
futuro_solo["yhat"] = futuro_solo["yhat"].clip(lower=0)   # nada de ventas negativas

venta_total = futuro_solo["yhat"].sum()
venta_diaria = futuro_solo["yhat"].mean()

# ----------------------------------------------------------------------
# 5) Mostrar resultados
# ----------------------------------------------------------------------
col1, col2 = st.columns(2)
col1.metric(f"Ventas estimadas ({semanas} semanas)", f"£{venta_total:,.0f}")
col2.metric("Venta diaria promedio", f"£{venta_diaria:,.0f}")

st.success(
    f"💡 **Recomendación:** prepárate para vender alrededor de **£{venta_total:,.0f}** "
    f"en las próximas {semanas} semanas. Asegura inventario y personal para cubrir esa demanda."
)

# ----------------------------------------------------------------------
# 6) Gráfica: historial + predicción
# ----------------------------------------------------------------------
st.subheader("Historial y predicción")
fig = modelo.plot(prediccion)
st.pyplot(fig)
st.caption("Los puntos son las ventas reales; la línea azul es la predicción, con su rango probable.")

# ----------------------------------------------------------------------
# 7) Tabla con la predicción día a día
# ----------------------------------------------------------------------
with st.expander("Ver la predicción día a día"):
    tabla = futuro_solo[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    tabla.columns = ["Fecha", "Predicción (£)", "Mínimo (£)", "Máximo (£)"]
    tabla["Fecha"] = tabla["Fecha"].dt.date
    st.dataframe(tabla.round(0), use_container_width=True, hide_index=True)

st.caption("Datos: Online Retail (tienda online real, Reino Unido) · Modelo: Prophet · Autora: Luz Dany Alejandra Fajardo Mora")
