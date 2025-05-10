from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from app.api.endpoints import router as api_router

# Crear la aplicación FastAPI
app = FastAPI(
    title="API de Predicción del Titanic",
    description="API para predecir la supervivencia de pasajeros del Titanic basada en sus características",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, restringe a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(api_router, prefix="/api/v1")

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "mensaje": "Bienvenido a la API de predicción de supervivencia del Titanic",
        "documentación": "/docs",
        "health_check": "/api/v1/health"
    }

# Ejecutar la aplicación con Uvicorn si se llama directamente
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True) 