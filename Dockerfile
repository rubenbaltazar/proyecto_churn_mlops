# Usar Python 3.12.10 como base (versión oficial liviana)
FROM python:3.12.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de dependencias
COPY requirements.txt .

# Instalar las dependencias dentro del contenedor
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código del proyecto
COPY . .

# Exponer el puerto 8000 para la API
EXPOSE 8000

# Comando para ejecutar la API cuando el contenedor inicie
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
