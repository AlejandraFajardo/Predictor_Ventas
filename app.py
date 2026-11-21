# -*- coding: utf-8 -*-
"""
Predictor de Ventas para E-commerce — App (Fase 3)
Autora: Luz Dany Alejandra Fajardo Mora
Tecnologías: Python + Prophet + Streamlit

Esta versión permite:
  (1) usar los datos de ejemplo, o
  (2) subir tus propios datos de ventas.
"""

import streamlit as st
import pandas as pd
from prophet import Prophet

st.set_page_config(page_title="Predictor de Ventas", page_icon="📈", layout="centered")

st.title("📈 Predictor de Ventas para E-commerce")
st.write(
    "Esta herramienta predice las ventas futuras de una tienda a partir de su historial, "
    "para ayudar a decidir **cuánto preparar** y no quedarse sin stock ni gastar de más."
)

# ----------------------------------------------------------------------
# 1) Elegir la fuente de datos
# ----------------------------------------------------------------------
st.subheader("1. Elige los datos")
fuente = st.radio(
    "¿Qué datos quieres usar?",
    ["Datos de ejemplo (tienda demo)", "Subir mis propios datos"],
)

# --- Función para los datos de ejemplo (en caché) ---
@st.cache_data
def cargar_ejemplo():
    url = "https://raw.githubusercontent.com/SMU-ExEd/python-for-analytics-data/main/online_retail.csv.gz"
    df = pd.read_csv(url)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df.dropna(subset=["Description"]).drop_duplicates()
    df["Ingresos"] = df["Quantity"] * df["UnitPrice"]
    serie = df.groupby(df["InvoiceDate"].dt.date)["Ingresos"].sum().reset_index()
    serie.columns = ["ds", "y"]
    serie["ds"] = pd.to_datetime(serie["ds"])
    return serie

serie = None

if fuente == "Datos de ejemplo (tienda demo)":
    serie = cargar_ejemplo()

else:
    st.write("Sube un archivo (CSV o Excel) con al menos una columna de **fecha** y una de **ventas**.")
    archivo = st.file_uploader("Subir archivo", type=["csv", "xlsx"])

    if archivo is None:
        st.info("Esperando tu archivo…")
        st.stop()   # no sigue hasta que suba algo

    # Leer el archivo según su tipo
    if archivo.name.endswith(".csv"):
        datos = pd.read_csv(archivo)
    else:
        datos = pd.read_excel(archivo)

    st.write("Vista previa de tus datos:")
    st.dataframe(datos.head(), use_container_width=True)

    # La persona indica qué columna es la fecha y cuál la venta
    col1, col2 = st.columns(2)
    col_fecha = col1.selectbox("Columna de FECHA", datos.columns)
    col_valor = col2.selectbox("Columna de VENTAS", datos.columns)

    # Construir la serie en el formato de Prophet
    serie = datos[[col_fecha, col_valor]].copy()
    serie.columns = ["ds", "y"]
    serie["ds"] = pd.to_datetime(serie["ds"], errors="coerce")   # fechas inválidas -> vacío
    serie["y"] = pd.to_numeric(serie["y"], errors="coerce")
    serie = serie.dropna()
    serie = serie.groupby("ds")["y"].sum().reset_index()         # sumar por día

    if len(serie) < 20:
        st.warning("Se necesitan al menos ~20 días de datos para predecir. Revisa tu archivo.")
        st.stop()

# ----------------------------------------------------------------------
# 2) Entrenar el modelo (en caché por conjunto de datos)
# ----------------------------------------------------------------------
@st.cache_resource
def entrenar_modelo(serie):
    modelo = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    modelo.fit(serie)
    return modelo

with st.spinner("Entrenando el modelo… (puede tardar un poco la primera vez)"):
    modelo = entrenar_modelo(serie)

# ----------------------------------------------------------------------
# 3) Elegir cuánto predecir
# ----------------------------------------------------------------------
st.subheader("2. ¿Cuánto quieres predecir?")
semanas = st.slider("Semanas hacia el futuro", min_value=1, max_value=8, value=4)
dias = semanas * 7

# ----------------------------------------------------------------------
# 4) Predecir
# ----------------------------------------------------------------------
futuro = modelo.make_future_dataframe(periods=dias)
prediccion = modelo.predict(futuro)

ultimo_dia = serie["ds"].max()
futuro_solo = prediccion[prediccion["ds"] > ultimo_dia].copy()
futuro_solo["yhat"] = futuro_solo["yhat"].clip(lower=0)

venta_total = futuro_solo["yhat"].sum()
venta_diaria = futuro_solo["yhat"].mean()

# ----------------------------------------------------------------------
# 5) Mostrar resultados
# ----------------------------------------------------------------------
st.subheader("3. Resultados")
c1, c2 = st.columns(2)
c1.metric(f"Ventas estimadas ({semanas} semanas)", f"{venta_total:,.0f}")
c2.metric("Venta diaria promedio", f"{venta_diaria:,.0f}")

st.success(
    f"💡 **Recomendación:** prepárate para vender alrededor de **{venta_total:,.0f}** "
    f"en las próximas {semanas} semanas. Asegura inventario y personal para cubrir esa demanda."
)

st.subheader("Historial y predicción")
fig = modelo.plot(prediccion)
st.pyplot(fig)
st.caption("Los puntos son las ventas reales; la línea azul es la predicción, con su rango probable.")

with st.expander("Ver la predicción día a día"):
    tabla = futuro_solo[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    tabla.columns = ["Fecha", "Predicción", "Mínimo", "Máximo"]
    tabla["Fecha"] = tabla["Fecha"].dt.date
    st.dataframe(tabla.round(0), use_container_width=True, hide_index=True)

st.caption("Modelo: Prophet · Autora: Luz Dany Alejandra Fajardo Mora")
