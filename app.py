import streamlit as st
import pandas as pd
from datetime import datetime
from mantenimiento import MantenimientoPredictivo

st.set_page_config(page_title="Sistema Predictivo de Mantenimiento", layout="centered")
st.title("🛠️ Sistema Predictivo de Mantenimiento para Flota Minera")

uploaded_file = st.file_uploader("📁 Carga tu archivo CSV con los datos del equipo", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Archivo cargado correctamente.")

    data['Fecha'] = pd.to_datetime(data['Fecha'])
    data.rename(columns={
        'Tiempo de funcionamiento (horas)': 'TIEMPO_FUNCIONAMIENTO',
        'Tiempo de trabajo (horas)': 'TIEMPO_TRABAJO',
        'Tiempo en vacío (horas)': 'TIEMPO_EN_VACIO',
        'Último horómetro conocido (Horas)': 'HOROMETRO'
    }, inplace=True)

    modelo = st.text_input("🔧 Modelo del equipo")
    serie = st.text_input("🔢 Serie del equipo")

    horo_mant = st.number_input("🔩 Horómetro del último mantenimiento", min_value=0.0, step=0.1)
    fecha_mant = st.date_input("📅 Fecha del último mantenimiento", format="YYYY-MM-DD")

    if st.button("🚀 Ejecutar análisis"):
        try:
            predictor = MantenimientoPredictivo(data, modelo, serie)
            st.subheader("📊 Gráfica de utilización del equipo")
            fig_util = predictor.graficar_utilizacion_web()
            st.pyplot(fig_util)

            st.subheader("🔮 Predicción de mantenimiento")
            fig_line, fig_bar, resumen_texto = predictor.predecir_mantenimiento_web(horo_mant, fecha_mant)
            st.pyplot(fig_line)
            st.pyplot(fig_bar)
            st.success(resumen_texto)
        except Exception as e:
            st.error(f"❌ Error durante la predicción: {e}")
else:
    st.info("⬆️ Sube un archivo CSV para comenzar.")