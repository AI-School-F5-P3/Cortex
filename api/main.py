from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from pymongo import MongoClient
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de MongoDB
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["telco_db"]
customers_collection = db["customers"]

class CustomerInput(BaseModel):
    region: int
    tenure: int
    age: int
    marital: int
    address: int
    income: float
    ed: int
    employ: int
    retire: int
    gender: int
    reside: int

    @validator('region')
    def validate_region(cls, v):
        if not 1 <= v <= 3:
            raise ValueError('Region must be between 1 and 3')
        return v

    @validator('age')
    def validate_age(cls, v):
        if not 18 <= v <= 100:
            raise ValueError('Age must be between 18 and 100')
        return v

    @validator('ed')
    def validate_education(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Education level must be between 1 and 5')
        return v

    @validator('marital', 'retire', 'gender')
    def validate_binary(cls, v):
        if v not in [0, 1]:
            raise ValueError('Value must be 0 or 1')
        return v

# Cargar modelo y escalador
try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("knn_model.pkl")
    logger.info("Modelo y escalador cargados exitosamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo o escalador: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error en la inicialización del servidor: {str(e)}")

# Cargar el PCA
try:
    pca = joblib.load("pca.pkl")
    logger.info("PCA cargado exitosamente")
except Exception as e:
    logger.error(f"Error al cargar el PCA: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error en la inicialización del servidor: {str(e)}")

# Verificar las características esperadas
expected_features = ['region', 'tenure', 'age', 'marital', 'address', 'income', 
                     'ed', 'employ', 'retire', 'gender', 'reside']
logger.info("Características esperadas verificadas manualmente.")

@app.post("/predict")
async def predict_category(customer: CustomerInput):
    try:
        # Convertir input a DataFrame
        input_data = pd.DataFrame([customer.dict()])
        logger.info(f"Datos recibidos: \n{input_data}")

        # Verificar columnas
        missing_cols = set(expected_features) - set(input_data.columns)
        if missing_cols:
            error_msg = f"Faltan columnas requeridas: {missing_cols}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Reordenar columnas para coincidir con el orden del modelo
        input_data = input_data[expected_features]
        logger.info(f"Datos ordenados: \n{input_data}")

        # Escalar datos
        data_scaled = scaler.transform(input_data)
        logger.info(f"Datos escalados: \n{data_scaled}")

        # Reducir dimensionalidad con PCA
        try:
            data_pca = pca.transform(data_scaled)
            logger.info(f"Datos transformados con PCA: \n{data_pca}")
        except Exception as e:
            logger.error(f"Error al aplicar PCA: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al aplicar PCA: {str(e)}")

        # Realizar predicción
        try:
            prediction = model.predict(data_pca)[0]
            logger.info(f"Predicción realizada: {prediction}")
        except Exception as e:
            logger.error(f"Error en la predicción: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

        # Guardar en MongoDB
        customer_data = customer.dict()
        customer_data["custcat"] = int(prediction)
        try:
            customers_collection.insert_one(customer_data)
            logger.info("Datos guardados en MongoDB")
        except Exception as e:
            logger.error(f"Error al guardar en MongoDB: {str(e)}")
            # No levantamos una excepción aquí para no interrumpir la respuesta

        return {
            "prediction": int(prediction),
            "message": "Predicción realizada exitosamente",
            "input_features": customer.dict()
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")

@app.get("/api/test")
async def test_endpoint():
    return {"message": "API funcionando correctamente", "status": "OK"}

@app.get("/")
async def root():
    return {
        "message": "API de Predicción de Categorías de Clientes",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Realizar predicción",
            "/api/test": "GET - Probar conexión"
        }
    }
