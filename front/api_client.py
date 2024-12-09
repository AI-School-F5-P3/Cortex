import requests
from typing import Dict, Any, Optional
import streamlit as st
from config import get_api_url
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionAPI:
    """Cliente para interactuar con la API de predicciones."""

    @staticmethod
    def predict_category(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Envia los datos a la API para predecir la categoría.

        Args:
            data (Dict[str, Any]): Datos del cliente para realizar la predicción.

        Returns:
            Optional[Dict[str, Any]]: Respuesta de la API con la predicción o None si ocurre un error.
        """
        try:
            logger.info(f"Enviando datos a la API: {data}")
            
            # Verificar campos requeridos antes de realizar la solicitud
            required_fields = ['region', 'tenure', 'age', 'marital', 'address', 
                               'income', 'ed', 'employ', 'retire', 'gender', 'reside']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                error_msg = f"Faltan campos requeridos: {missing_fields}"
                st.error(error_msg)
                logger.error(error_msg)
                return None

            # Realizar la solicitud POST a la API
            api_url = get_api_url("predict")
            logger.info(f"Llamando a la API en {api_url}")
            response = requests.post(api_url, json=data)

            # Procesar la respuesta
            logger.info(f"Respuesta del servidor: {response.status_code} - {response.text}")
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Predicción exitosa: {result}")
                return result
            else:
                error_detail = response.json().get('detail', 'Error desconocido')
                error_msg = f"Error en la predicción: {error_detail}"
                st.error(error_msg)
                logger.error(error_msg)
                return None

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Error de conexión con el servidor: {str(e)}"
            st.error(error_msg)
            logger.error(error_msg)
            return None
        except requests.exceptions.RequestException as e:
            error_msg = f"Error en la solicitud a la API: {str(e)}"
            st.error(error_msg)
            logger.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"Error inesperado: {str(e)}"
            st.error(error_msg)
            logger.error(error_msg)
            return None
