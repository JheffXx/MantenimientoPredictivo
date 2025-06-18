import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler
import os
import time

class MantenimientoPredictivo:
    def __init__(self, data, modelo, serie):
        self.data = data.copy()
        self.modelo = modelo
        self.serie = serie
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.training_path = "Training"

    def graficar_utilizacion_web(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data['Fecha'], self.data['TIEMPO_FUNCIONAMIENTO'], label='Tiempo de funcionamiento', color='blue')
        ax.plot(self.data['Fecha'], self.data['TIEMPO_TRABAJO'], label='Tiempo de trabajo', color='green')
        ax.plot(self.data['Fecha'], self.data['TIEMPO_EN_VACIO'], label='Tiempo en vacío', color='orange')
        ax.axhline(y=self.data['TIEMPO_FUNCIONAMIENTO'].max(), color='red', linestyle='--', label='Máximo')
        ax.axhline(y=self.data['TIEMPO_FUNCIONAMIENTO'].mean(), color='gray', linestyle='--', label='Promedio')
        ax.set_title(f'Utilización del equipo: {self.modelo} - Serie: {self.serie}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Horas')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def predecir_mantenimiento_web(self, horo_mant, fecha_mant):
        fecha_mant = pd.to_datetime(fecha_mant)
        self.data['DIA'] = (self.data['Fecha'] - self.data['Fecha'].min()).dt.days
        data_limpia = self.data.dropna(subset=['HOROMETRO', 'DIA'])
        data_limpia.sort_values('Fecha', inplace=True)

        X = data_limpia[['DIA']].values
        y = data_limpia['HOROMETRO'].values.reshape(-1, 1)

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        model = Sequential([
            Input(shape=(1,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        import time
        time.sleep(1.5)  # Simulación de espera (barrita de carga)

        model.fit(X_scaled, y_scaled, epochs=500, verbose=0)

        os.makedirs(self.training_path, exist_ok=True)
        model.save(os.path.join(self.training_path, 'modelo_entrenado.h5'))
        np.save(os.path.join(self.training_path, 'scaler_X_min.npy'), self.scaler_X.data_min_)
        np.save(os.path.join(self.training_path, 'scaler_X_max.npy'), self.scaler_X.data_max_)
        np.save(os.path.join(self.training_path, 'scaler_y_min.npy'), self.scaler_y.data_min_)
        np.save(os.path.join(self.training_path, 'scaler_y_max.npy'), self.scaler_y.data_max_)

        # Datos actuales
        fecha_actual = data_limpia['Fecha'].iloc[-1]
        horometro_actual = data_limpia['HOROMETRO'].iloc[-1]
        dias_trans = (fecha_actual - fecha_mant).days
        horas_trans = horometro_actual - horo_mant
        horas_diarias = horas_trans / dias_trans if dias_trans > 0 else 10

        # 🧠 Predicción del siguiente horómetro usando la red neuronal
        dia_pred = data_limpia['DIA'].iloc[-1] + 10
        X_pred_scaled = self.scaler_X.transform([[dia_pred]])
        horometro_pred = self.scaler_y.inverse_transform(model.predict(X_pred_scaled))[0][0]

        # ✅ Calcular fecha estimada en base a ritmo real de uso
        horas_restantes = horometro_pred - horometro_actual
        dias_estimados = int(np.ceil(horas_restantes / horas_diarias)) if horas_diarias > 0 else 1
        fecha_estimada = fecha_actual + timedelta(days=dias_estimados)
        siguiente_horometro = horometro_pred

        # 📈 Gráfico de líneas
        fig_line, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(data_limpia['Fecha'], y, label="Horómetro actual", color='blue')
        ax1.axvline(fecha_estimada, color='orange', linestyle='--', label="Fecha estimada")
        ax1.axhline(siguiente_horometro, color='orange', linestyle='--')
        ax1.scatter([fecha_estimada], [siguiente_horometro], color='orange')
        ax1.set_title(f"Predicción de Mantenimiento - {self.modelo} {self.serie}")
        ax1.set_xlabel("Fecha")
        ax1.set_ylabel("Horómetro")
        ax1.legend()
        ax1.grid(True)
        plt.tight_layout()

        # 📊 Gráfico de barras
        fig_bar, ax2 = plt.subplots(figsize=(8, 5))
        etiquetas = ['Actual', 'Mantenimiento']
        valores = [horometro_actual, siguiente_horometro]
        colores = ['#1f77b4', '#ff7f0e']
        bars = ax2.bar(etiquetas, valores, color=colores)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, height + 5, f'{height:.2f}', ha='center', va='bottom')
        ax2.set_title(f"Horómetro Actual vs Mantenimiento - {self.modelo} {self.serie}")
        ax2.set_ylabel("Valor Horómetro")
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 📝 Texto resumen
        resumen = f"📋 Horómetro último mantenimiento: {horo_mant:.2f}\n" \
                f"🔍 Horómetro actual: {horometro_actual:.2f}\n" \
                f"🪚 Estimado para mantenimiento: {siguiente_horometro:.2f}\n" \
                f"📅 Fecha estimada: {fecha_estimada.strftime('%d/%m/%Y')}"

        return fig_line, fig_bar, resumen
    
##Para guardar la predicción en un historial CSV para entrenamiento posterior
    def guardar_prediccion_en_historial(modelo, serie, fecha_actual, horo_actual, horo_estimado, fecha_estimada, horas_diarias, feedback=None):
        historial_path = "Training/historial_predicciones.csv"
        os.makedirs("Training", exist_ok=True)

        nueva_fila = {
            "Modelo": modelo,
            "Serie": serie,
            "Fecha actual": fecha_actual.strftime('%Y-%m-%d'),
            "Horómetro actual": round(horo_actual, 2),
            "Horómetro estimado": round(horo_estimado, 2),
            "Fecha estimada": fecha_estimada.strftime('%Y-%m-%d'),
            "Promedio horas/día": round(horas_diarias, 2),
            "Feedback": feedback or "Pendiente"
        }

        if os.path.exists(historial_path):
            df_historial = pd.read_csv(historial_path)
            df_historial = pd.concat([df_historial, pd.DataFrame([nueva_fila])], ignore_index=True)
        else:
            df_historial = pd.DataFrame([nueva_fila])

        df_historial.to_csv(historial_path, index=False)
