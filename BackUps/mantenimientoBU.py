import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import os
import shutil

class MantenimientoPredictivo:
    def __init__(self, ruta_csv, modelo, serie):
        self.ruta_csv = ruta_csv
        self.modelo = modelo
        self.serie = serie
        self.data = self._cargar_datos()

    def _cargar_datos(self):
        data = pd.read_csv(self.ruta_csv)
        data['Fecha'] = pd.to_datetime(data['Fecha'])
        data.sort_values('Fecha', inplace=True)
        data.rename(columns={
            'Tiempo de funcionamiento (horas)': 'TIEMPO_FUNCIONAMIENTO',
            'Tiempo de trabajo (horas)': 'TIEMPO_TRABAJO',
            'Tiempo en vacío (horas)': 'TIEMPO_EN_VACIO',
            'Último horómetro conocido (Horas)': 'HOROMETRO'
        }, inplace=True)
        return data

    def graficar_utilizacion(self):
        data = self.data
        plt.figure(figsize=(12, 6))
        plt.plot(data['Fecha'], data['TIEMPO_FUNCIONAMIENTO'], label='Tiempo de funcionamiento', color='blue')
        plt.plot(data['Fecha'], data['TIEMPO_TRABAJO'], label='Tiempo de trabajo', color='green')
        plt.plot(data['Fecha'], data['TIEMPO_EN_VACIO'], label='Tiempo en vacío', color='orange')
        plt.axhline(y=data['TIEMPO_FUNCIONAMIENTO'].max(), color='red', linestyle='--', label='Máximo')
        plt.axhline(y=data['TIEMPO_FUNCIONAMIENTO'].mean(), color='gray', linestyle='--', label='Promedio')
        plt.title(f'Utilización del equipo: {self.modelo} - Serie: {self.serie}')
        plt.xlabel('Fecha')
        plt.ylabel('Horas')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def predecir_mantenimiento(self, horo_mant, fecha_mant_str):
        data = self.data
        fecha_mant = pd.to_datetime(fecha_mant_str)
        data['DIA'] = (data['Fecha'] - data['Fecha'].min()).dt.days
        data_limpia = data.dropna(subset=['HOROMETRO', 'DIA'])

        X = data_limpia[['DIA']].values
        y = data_limpia['HOROMETRO'].values.reshape(-1, 1)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        model = Sequential([
            Input(shape=(1,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        model.fit(X_scaled, y_scaled, epochs=500, verbose=0, callbacks=[early_stop])

        # Guardar modelo y parámetros
        training_dir = "/Users/jherson.lg/Desktop/Proyecto/Tesis/Training"
        os.makedirs(training_dir, exist_ok=True)
        model.save(os.path.join(training_dir, "modelo_entrenado.h5"))
        np.save(os.path.join(training_dir, "scaler_X_min.npy"), scaler_X.data_min_)
        np.save(os.path.join(training_dir, "scaler_X_max.npy"), scaler_X.data_max_)
        np.save(os.path.join(training_dir, "scaler_y_min.npy"), scaler_y.data_min_)
        np.save(os.path.join(training_dir, "scaler_y_max.npy"), scaler_y.data_max_)

        print(f"\n✅ Modelo y parámetros almacenados correctamente en: {training_dir}")

        horometro_actual = data_limpia['HOROMETRO'].iloc[-1]
        fecha_actual = data_limpia['Fecha'].iloc[-1]
        dias_trans = (fecha_actual - fecha_mant).days
        horas_trans = horometro_actual - horo_mant
        horas_diarias = horas_trans / dias_trans if dias_trans > 0 else 10

        if horas_diarias > 10:
            intervalo = 200
        elif horas_diarias < 7:
            intervalo = 260
        else:
            intervalo = 240

        horas_restantes = intervalo - horas_trans
        dias_estimados = int(np.ceil(horas_restantes / horas_diarias))
        siguiente_horometro = horo_mant + intervalo
        fecha_estimada = fecha_mant + timedelta(days=dias_estimados)

        # Gráfico de líneas
        dia_pred = data_limpia['DIA'].iloc[-1] + dias_estimados
        X_pred_scaled = scaler_X.transform([[dia_pred]])
        horometro_pred = scaler_y.inverse_transform(model.predict(X_pred_scaled))[0][0]

        plt.figure(figsize=(10, 5))
        plt.plot(data_limpia['Fecha'], y, label="Horómetro actual", color='blue')
        plt.axvline(fecha_estimada, color='orange', linestyle='--', label="Fecha estimada")
        plt.axhline(siguiente_horometro, color='orange', linestyle='--')
        plt.scatter([fecha_estimada], [siguiente_horometro], color='orange')
        plt.title(f"Predicción - {self.modelo} {self.serie}")
        plt.xlabel("Fecha")
        plt.ylabel("Horómetro")
        plt.legend()
        plt.tight_layout()
        plt.figtext(0.5, -0.08, f"🔧 Estimado: {siguiente_horometro:.2f} hrs\n📅 {fecha_estimada.strftime('%d/%m/%Y')}", ha="center")
        plt.show()

        print(f"\n📋 Horómetro último mantenimiento: {horo_mant:.2f}")
        print(f"🔍 Horómetro actual: {horometro_actual:.2f}")
        print(f"🪚 Estimado para mantenimiento: {siguiente_horometro:.2f}")
        print(f"📅 Fecha estimada: {fecha_estimada.strftime('%d/%m/%Y')}")