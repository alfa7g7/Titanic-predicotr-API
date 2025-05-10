from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from app.models.titanic_model import TitanicModel

# Inicializar el router
router = APIRouter()

# Inicializar el modelo
model = TitanicModel()
model.load()

# Definir el esquema de datos para la predicción
class PassengerData(BaseModel):
    Pclass: int = Field(..., description="Clase del pasajero (1 = 1ra clase, 2 = 2da clase, 3 = 3ra clase)", ge=1, le=3)
    Sex: str = Field(..., description="Género del pasajero ('male' o 'female')")
    Age: Optional[float] = Field(None, description="Edad del pasajero en años")
    SibSp: int = Field(0, description="Número de hermanos/cónyuges a bordo")
    Parch: int = Field(0, description="Número de padres/hijos a bordo")
    Fare: Optional[float] = Field(None, description="Tarifa del pasajero")
    Embarked: Optional[str] = Field(None, description="Puerto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)")

    class Config:
        schema_extra = {
            "example": {
                "Pclass": 3,
                "Sex": "male",
                "Age": 22.0,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 7.25,
                "Embarked": "S"
            }
        }

# Definir el esquema de respuesta
class PredictionResponse(BaseModel):
    survived: int = Field(..., description="Predicción de supervivencia (0 = No sobrevive, 1 = Sobrevive)")
    survival_probability: float = Field(..., description="Probabilidad de supervivencia")
    passenger_data: Dict[str, Any] = Field(..., description="Datos del pasajero utilizados para la predicción")

# Endpoint para la predicción
@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_survival(passenger: PassengerData):
    try:
        # Convertir el objeto Pydantic a diccionario
        passenger_dict = passenger.dict()
        
        # Realizar la predicción
        prediction, probability = model.predict(passenger_dict)
        
        # Crear la respuesta
        response = {
            "survived": prediction,
            "survival_probability": probability,
            "passenger_data": passenger_dict
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

# Endpoint para verificar la salud del servicio
@router.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "message": "El servicio de predicción del Titanic está funcionando correctamente"} 