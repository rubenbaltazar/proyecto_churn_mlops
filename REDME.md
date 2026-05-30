# Proyecto Churn MLOps

Este proyecto predice qué clientes pueden abandonar un servicio (churn).

## Estructura del proyecto
- **data**: Datos del proyecto (CSV)
- **src**: Código fuente (preparación, entrenamiento, evaluación)
- **models**: Modelo entrenado (.pkl)
- **api**: API con FastAPI
- **tests**: Pruebas automáticas
- **docs**: Métricas y documentación

## Cómo ejecutar
1. Construir Docker: `docker-compose up -d --build`
2. Entrenar: `docker exec churn_mlops_api python src/preparar_datos.py`
3. Entrenar modelo: `docker exec churn_mlops_api python src/entrenar_modelo.py`
4. Evaluar: `docker exec churn_mlops_api python src/evaluar_modelo.py`
5. Abrir API: `http://localhost:8000/docs`