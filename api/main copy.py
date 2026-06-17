# ============================================================
# BLOQUE 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================
from collections import Counter
from pathlib import Path
from threading import Lock
from time import perf_counter
import logging
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

# ============================================================
# BLOQUE 2. CONFIGURACIÓN GENERAL DEL PROYECTO
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ruta del modelo serializado (tu modelo original)
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_churn.pkl"

LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "monitor_api.log"

VERSION_MODELO = "modelo_churn_v1"
AUTOR = "Ruben Baltazar Balderrama"  # 👈 Tu nombre completo

# ============================================================
# BLOQUE 3. RANGOS HISTÓRICOS DE REFERENCIA (para 5 variables)
# ============================================================
RANGOS_HISTORICOS = {
    "edad": (18, 100),
    "antiguedad_meses": (0, 240),
    "saldo_promedio": (0, 50000),
    "reclamos": (0, 50),
    "usa_app": (0, 1),
}

# ============================================================
# BLOQUE 4. LOGGING A ARCHIVO Y CONSOLA
# ============================================================
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("api_churn")

# ============================================================
# BLOQUE 5. VERIFICACIÓN Y CARGA DEL MODELO
# ============================================================
if not MODEL_PATH.exists():
    raise RuntimeError(
        f"No se encontró el modelo serializado en {MODEL_PATH}. "
        "Ejecute primero el entrenamiento del modelo."
    )

modelo = joblib.load(MODEL_PATH)
logger.info("Modelo cargado correctamente: %s", VERSION_MODELO)

# ============================================================
# BLOQUE 6. CONTADORES DE MÉTRICAS EN MEMORIA
# ============================================================
metricas = {
    "solicitudes_totales": 0,
    "errores_validacion": 0,
    "errores_internos": 0,
    "predicciones_validas": 0,
    "predicciones_alto_riesgo": 0,
    "predicciones_bajo_riesgo": 0,
    "solicitudes_con_anomalias": 0,
    "latencia_acumulada_ms": 0.0,
    "latencia_maxima_ms": 0.0,
    "codigos_http": Counter(),
}

metricas_lock = Lock()

# ============================================================
# BLOQUE 7. MODELOS DE DATOS Y VALIDACIÓN DE ENTRADAS (5 VARIABLES)
# ============================================================
class ClienteEntrada(BaseModel):
    edad: int = Field(..., ge=18, le=100, description="Edad del cliente en años")
    antiguedad_meses: int = Field(..., ge=0, le=240, description="Meses como cliente")
    saldo_promedio: float = Field(..., ge=0, le=50000, description="Saldo promedio mensual")
    reclamos: int = Field(..., ge=0, le=50, description="Número de reclamos")
    usa_app: int = Field(..., ge=0, le=1, description="Usa la app (0=No, 1=Si)")

class PrediccionSalida(BaseModel):
    prediccion: str
    probabilidad: float
    version_modelo: str
    autor: str
    alertas_datos: list[str]

# ============================================================
# BLOQUE 8. DETECCIÓN DE VALORES FUERA DEL RANGO HISTÓRICO
# ============================================================
def detectar_anomalias(datos: ClienteEntrada) -> list[str]:
    alertas: list[str] = []
    valores = datos.model_dump()
    
    for variable, valor in valores.items():
        if variable in RANGOS_HISTORICOS:
            minimo, maximo = RANGOS_HISTORICOS[variable]
            if valor < minimo or valor > maximo:
                alertas.append(
                    f"{variable}={valor} fuera del rango histórico [{minimo}, {maximo}]"
                )
    return alertas

# ============================================================
# BLOQUE 9. PREPARACIÓN DEL RESUMEN DE MÉTRICAS
# ============================================================
def resumen_metricas() -> dict:
    with metricas_lock:
        total = metricas["solicitudes_totales"]
        latencia_promedio = (metricas["latencia_acumulada_ms"] / total) if total else 0.0
        
        return {
            "version_modelo": VERSION_MODELO,
            "autor": AUTOR,
            "solicitudes_totales": total,
            "errores_validacion": metricas["errores_validacion"],
            "errores_internos": metricas["errores_internos"],
            "predicciones_validas": metricas["predicciones_validas"],
            "predicciones_alto_riesgo": metricas["predicciones_alto_riesgo"],
            "predicciones_bajo_riesgo": metricas["predicciones_bajo_riesgo"],
            "solicitudes_con_anomalias": metricas["solicitudes_con_anomalias"],
            "latencia_promedio_ms": round(latencia_promedio, 3),
            "latencia_maxima_ms": round(metricas["latencia_maxima_ms"], 3),
            "codigos_http": dict(metricas["codigos_http"]),
        }

# ============================================================
# BLOQUE 10. CREACIÓN DE LA APLICACIÓN FASTAPI
# ============================================================
app = FastAPI(
    title="API de predicción de churn - Ruben Baltazar",
    description="Servicio académico ML-Ops con métricas, logs y detección de anomalías.",
    version="2.0.0",
)

# ============================================================
# BLOQUE 11. MIDDLEWARE PARA MEDIR LATENCIA Y CONTAR SOLICITUDES
# ============================================================
@app.middleware("http")
async def registrar_solicitud(request: Request, call_next):
    inicio = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        with metricas_lock:
            metricas["errores_internos"] += 1
        logger.exception("Error interno no controlado en %s", request.url.path)
        raise
        
    latencia_ms = (perf_counter() - inicio) * 1000
    
    with metricas_lock:
        metricas["solicitudes_totales"] += 1
        metricas["latencia_acumulada_ms"] += latencia_ms
        metricas["latencia_maxima_ms"] = max(metricas["latencia_maxima_ms"], latencia_ms)
        metricas["codigos_http"][str(response.status_code)] += 1
        
    logger.info(
        "Solicitud | metodo=%s | ruta=%s | estado=%s | latencia_ms=%.3f",
        request.method,
        request.url.path,
        response.status_code,
        latencia_ms,
    )
    
    response.headers["X-Process-Time-ms"] = f"{latencia_ms:.3f}"
    return response

# ============================================================
# BLOQUE 12. MANEJO DE ERRORES DE VALIDACIÓN
# ============================================================
@app.exception_handler(RequestValidationError)
async def registrar_error_validacion(request: Request, exc: RequestValidationError):
    with metricas_lock:
        metricas["errores_validacion"] += 1
        
    logger.warning(
        "Error de validación | ruta=%s | detalle=%s",
        request.url.path,
        exc.errors(),
    )
    return await request_validation_exception_handler(request, exc)

# ============================================================
# BLOQUE 13. ENDPOINT DE INICIO
# ============================================================
@app.get("/")
def inicio() -> dict[str, str]:
    return {
        "mensaje": "Servicio ML-Ops activo",
        "estado": "ok",
        "autor": AUTOR,
    }
@app.get("/info")
async def get_info():
    return {
        "proyecto": "Churn Prediction API",
        "autor": "Ruben Baltazar Balderrama",
        "version": "1.0.0"
        
    }
# ============================================================
# BLOQUE 14. ENDPOINT DE SALUD
# ============================================================
@app.get("/health")
def health() -> dict[str, str]:
    return {
        "estado": "ok",
        "modelo": VERSION_MODELO,
        "monitoreo": "activo",
    }

# ============================================================
# BLOQUE 15. ENDPOINT GET /METRICS (MEJORA PERSONAL)
# ============================================================
@app.get("/metrics")
def metrics() -> dict:
    """Endpoint con métricas de uso de la API - MEJORA PERSONAL"""
    return resumen_metricas()

# ============================================================
# BLOQUE 16. ENDPOINT POST /PREDICT
# ============================================================
@app.post("/predict", response_model=PrediccionSalida)
def predict(datos: ClienteEntrada) -> PrediccionSalida:
    try:
        # Paso 1. Detectar datos atípicos.
        alertas = detectar_anomalias(datos)
 
        # Paso 2. Preparar la entrada matricial (5 variables)
        X = np.array([[
            datos.edad,
            datos.antiguedad_meses,
            datos.saldo_promedio,
            datos.reclamos,
            datos.usa_app
        ]])
 
        # Paso 3. Calcular la probabilidad de abandono (clase 1)
        probabilidad = float(modelo.predict_proba(X)[0][1])
 
        # Paso 4. Aplicar el umbral de decisión del 50 %
        etiqueta = "alto_riesgo" if probabilidad >= 0.50 else "bajo_riesgo"
 
        # Paso 5. Actualizar las métricas de predicción
        with metricas_lock:
            metricas["predicciones_validas"] += 1
            metricas[f"predicciones_{etiqueta}"] += 1
            if alertas:
                metricas["solicitudes_con_anomalias"] += 1
 
        # Paso 6. Registrar logs
        if alertas:
            logger.warning("Valores fuera de rango histórico: %s", alertas)
            
        logger.info(
            "Predicción | resultado=%s | probabilidad=%.4f | alertas=%s",
            etiqueta,
            probabilidad,
            len(alertas),
        )
 
        # Paso 7. Devolver la respuesta estructurada
        return PrediccionSalida(
            prediccion=etiqueta,
            probabilidad=round(probabilidad, 4),
            version_modelo=VERSION_MODELO,
            autor=AUTOR,
            alertas_datos=alertas,
        )
 
    except Exception as exc:
        with metricas_lock:
            metricas["errores_internos"] += 1
        logger.exception("No fue posible generar la predicción")
        raise HTTPException(
            status_code=500,
            detail="No fue posible generar la predicción.",
        ) from exc