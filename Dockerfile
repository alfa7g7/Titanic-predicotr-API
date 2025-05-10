FROM python:3.9-slim

WORKDIR /app

# Copiar los requerimientos y las dependencias
COPY requirements.txt ./

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY ./app ./app

# Al iniciar el contenedor, entrenar el modelo (si es necesario)
RUN python -m app.models.titanic_model

# Exponer el puerto que utiliza la API
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 