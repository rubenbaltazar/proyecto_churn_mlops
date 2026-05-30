from fastapi.testclient import TestClient
from api.main import app

# Crear un cliente de prueba
client = TestClient(app)

def test_inicio():
    """Prueba que el endpoint raíz funcione"""
    response = client.get("/")
    assert response.status_code == 200
    assert "mensaje" in response.json()

def test_health():
    """Prueba que el endpoint de salud funcione"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "estado" in response.json()
    assert "modelo_disponible" in response.json()