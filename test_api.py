import os
from google import genai
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Forzamos la llave nueva aquí por si el .env no se actualizó bien
api_key = os.environ.get("GEMINI_API_KEY") 
client = genai.Client(api_key=api_key)

print("Enviando petición de prueba...")

try:
    # Intenta con el 1.5-flash usando la llave NUEVA
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents="Hola, esto es una prueba de conexión. Responde 'Conexión exitosa'."
    )
    print("EXITO:", response.text)
except Exception as e:
    print("ERROR:", e)