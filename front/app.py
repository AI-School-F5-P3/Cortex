import streamlit as st
import pandas as pd
from api_client import PredictionAPI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_category_name(prediction: int) -> str:
    """Mapea el ID de la categoría al nombre descriptivo."""
    category_map = {
        1: "Basic Service",
        2: "E-Service",
        3: "Plus Service",
        4: "Total Service"
    }
    return category_map.get(prediction, "Categoría desconocida")

def main():
    st.title("Predicción de Categoría de Servicio")
    st.markdown("Esta aplicación utiliza un modelo de machine learning para predecir la categoría de servicio de un cliente basado en sus características.")

    # Contenedor para mensajes de estado
    status_container = st.empty()

    # Formulario para ingresar los datos del cliente
    with st.form("prediction_form"):
        st.header("Ingrese los datos del cliente")

        col1, col2 = st.columns(2)
        
        # Column 1 Inputs
        with col1:
            form_data = {
                "region": st.number_input("Región (1-3)", min_value=1, max_value=3, value=1),
                "tenure": st.number_input("Tiempo de permanencia (meses)", min_value=0, value=0),
                "age": st.number_input("Edad", min_value=18, max_value=100, value=30),
                "marital": st.selectbox(
                    "Estado Civil", 
                    options=[0, 1], 
                    format_func=lambda x: "Soltero" if x == 0 else "Casado"
                ),
                "address": st.number_input("Tiempo en dirección actual (años)", min_value=0, value=0),
                "income": st.number_input("Ingresos anuales (miles)", min_value=0.0, value=0.0)
            }
        
        # Column 2 Inputs
        with col2:
            form_data.update({
                "ed": st.selectbox(
                    "Nivel de Educación", 
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: {
                        1: "Sin estudios",
                        2: "Estudios básicos",
                        3: "Educación secundaria/bachillerato",
                        4: "Educación superior",
                        5: "Posgrados"
                    }[x]
                ),
                "employ": st.number_input("Años de empleo", min_value=0, value=0),
                "retire": st.selectbox(
                    "Retirado", 
                    options=[0, 1], 
                    format_func=lambda x: "No" if x == 0 else "Sí"
                ),
                "gender": st.selectbox(
                    "Género", 
                    options=[0, 1], 
                    format_func=lambda x: "Femenino" if x == 0 else "Masculino"
                ),
                "reside": st.number_input("Número de residentes", min_value=1, max_value=8, value=1)
            })

        # Botón de envío
        submitted = st.form_submit_button("Predecir Categoría")

    # Si el formulario se envió
    if submitted:
        with st.spinner('Realizando predicción...'):
            logger.info("Datos enviados para predicción: %s", form_data)
            
            # Mostrar los datos enviados en un cuadro expandible
            with st.expander("Ver datos a enviar"):
                st.json(form_data)
            
            # Llamada al cliente de la API para predecir
            try:
                result = PredictionAPI.predict_category(form_data)
                if result and 'prediction' in result:
                    prediction = result['prediction']
                    category_name = get_category_name(prediction)

                    # Mostrar el resultado
                    st.success(f"✅ Predicción exitosa")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Categoría predicha", category_name)
                    with col2:
                        st.metric("ID de categoría", prediction)
                    
                    # Mostrar detalles adicionales
                    with st.expander("Ver detalles completos"):
                        st.json(result)
                else:
                    st.error("❌ No se pudo realizar la predicción. Verifique los datos de entrada o intente nuevamente.")
            except Exception as e:
                logger.error(f"Error durante la predicción: {str(e)}")
                st.error("❌ Hubo un problema al conectar con la API. Inténtelo nuevamente más tarde.")

if __name__ == "__main__":
    main()
