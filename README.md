# API de Predicción del Titanic

Este proyecto expone un modelo de predicción de supervivencia en el Titanic como un servicio API utilizando FastAPI y Uvicorn, containerizado con Docker.

## Estructura del Proyecto

```
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── titanic_model.py
│   ├── __init__.py
│   └── main.py
├── Dockerfile
└── requirements.txt
```

## Funcionalidades

- API RESTful para predecir la supervivencia de pasajeros del Titanic
- Modelo de predicción basado en RandomForest
- Documentación automática de la API con Swagger UI
- Containerización con Docker

## Instrucciones de Uso

### Construcción de la Imagen Docker

```bash
docker build -t titanic-predictor-api .
```

### Ejecución del Contenedor

```bash
docker run -p 8000:8000 titanic-predictor-api
```

### Acceso a la API

Una vez que el contenedor está en ejecución, puedes acceder a:

- Documentación de la API: [http://localhost:8000/docs](http://localhost:8000/docs)
- Endpoint de verificación de salud: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
- Endpoint de predicción: [http://localhost:8000/api/v1/predict](http://localhost:8000/api/v1/predict) (POST)

### Ejemplo de Uso

Puedes hacer una predicción enviando una solicitud POST al endpoint `/api/v1/predict` con los siguientes datos:

```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22.0,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}
```

## Despliegue en Producción

Para desplegar esta API en producción, considera:

1. Cambiar las configuraciones de CORS para restringir los orígenes permitidos
2. Añadir autenticación/autorización
3. Configurar un balanceador de carga si se esperan muchas solicitudes
4. Monitorizar el rendimiento de la API

## Tecnologías Utilizadas

- FastAPI: Framework web para crear APIs con Python
- Uvicorn: Servidor ASGI de alto rendimiento
- scikit-learn: Biblioteca para machine learning
- Docker: Plataforma de containerización 