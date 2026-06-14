from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_FILE = BASE_DIR / "models" / "modelo_churn.pkl"

# Crear la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Churn",
    version="2",
    description="API para predecir si un cliente abandonará el servicio"
)

# Definir el formato de los datos que recibirá la API
class Cliente(BaseModel):
    edad: int
    antiguedad_meses: int
    saldo_promedio: float
    reclamos: int
    usa_app: int

def cargar_modelo():
    """Carga el modelo entrenado si existe"""
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


@app.get("/")
def inicio():
    return {
        "mensaje": "Servicio ML-Ops activo",
        "estado": "ok",
        "autor": "Ruben Baltazar Balderrama"
    }

@app.get("/health")
def health():
    """Endpoint para verificar el estado del servicio"""
    return {
        "estado": "ok",
        "modelo_disponible": MODEL_FILE.exists()
    }

@app.post("/predict")
def predict(cliente: Cliente):
    """Endpoint para predecir si un cliente hará churn"""
    modelo = cargar_modelo()
    
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo aún no está disponible. Primero debe entrenar el modelo."
        )
    
    # Convertir los datos a DataFrame
    datos = pd.DataFrame([cliente.model_dump()])
    
    # Hacer predicción
    prediccion = int(modelo.predict(datos)[0])
    
    # Obtener probabilidad (si está disponible)
    probabilidad = None
    if hasattr(modelo, "predict_proba"):
        probabilidad = float(modelo.predict_proba(datos)[0][1])
    
    return {
        "churn_predicho": prediccion,
        "probabilidad_churn": probabilidad
    }