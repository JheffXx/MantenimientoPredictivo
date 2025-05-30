##from mantenimiento import MantenimientoPredictivo
from BackUps.mantenimientoBU import MantenimientoPredictivo

ruta = "/Users/jherson.lg/Desktop/Proyecto/Tesis/data.csv"
modelo = input("Ingrese el modelo del equipo: ")
serie = input("Ingrese la serie del equipo: ")

sistema = MantenimientoPredictivo(ruta, modelo, serie)
sistema.graficar_utilizacion()

horo_mant = float(input("🔧 Ingrese el horómetro del último mantenimiento: "))
fecha_mant = input("📅 Ingrese la fecha del último mantenimiento (YYYY-MM-DD): ")

sistema.predecir_mantenimiento(horo_mant, fecha_mant)

# Ejecutar el script con: python3 Sistema.py