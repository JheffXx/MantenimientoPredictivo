class EntrenadorDinamico:
    def __init__(self, historial_path="Training/historial_predicciones.csv"):
        self.historial_path = historial_path

    def entrenar_modelo_con_historial(self):
        if not os.path.exists(self.historial_path):
            print("❌ No existe historial de predicciones.")
            return None

        df = pd.read_csv(self.historial_path)
        df = df[df['Feedback'] == "Sí"]  # Solo usar datos validados

        if df.empty:
            print("⚠️ No hay datos validados para entrenar.")
            return None

        # Preparar entrenamiento
        df['Fecha actual'] = pd.to_datetime(df['Fecha actual'])
        df['DIA'] = (df['Fecha actual'] - df['Fecha actual'].min()).dt.days
        X = df[['DIA']].values
        y = df[['Horómetro estimado']].values

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
        model.fit(X_scaled, y_scaled, epochs=500, verbose=0)

        # Guardar el nuevo modelo entrenado
        os.makedirs("Training", exist_ok=True)
        model.save("Training/modelo_refinado.h5")
        np.save("Training/scaler_hist_X_min.npy", scaler_X.data_min_)
        np.save("Training/scaler_hist_X_max.npy", scaler_X.data_max_)
        np.save("Training/scaler_hist_y_min.npy", scaler_y.data_min_)
        np.save("Training/scaler_hist_y_max.npy", scaler_y.data_max_)
        print("✅ Modelo entrenado con historial guardado correctamente.")