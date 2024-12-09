import os

# URL base de la API (permite usar variables de entorno)
API_URL = os.getenv("API_URL", "http://localhost:8000")

def get_api_url(endpoint: str) -> str:
    """
    Construye la URL completa para un endpoint específico.

    Args:
        endpoint (str): El endpoint de la API.

    Returns:
        str: URL completa del endpoint.
    """
    return f"{API_URL}/{endpoint}"
