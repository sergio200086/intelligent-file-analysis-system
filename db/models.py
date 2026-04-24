import chromadb
import uuid
from google import genai
import os
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

client_genai = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

client = chromadb.PersistentClient(path="./my_models")
collection = client.get_or_create_collection(
    name="documents"
)

def insert_document(text, id):
    collection.add(
        documents=[text],
        ids=[id]
    )

def query_document(text, results=3):
    results = collection.query(
        query_texts=[text],
        n_results=results
    )
    return results["documents"][0] # type: ignore


def answer_question(text):
    results = query_document(text)
    context = "\n".join(results)

    response = client_genai.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=f"Basándote en estos documentos:\n\n{context}\n\nResponde: {text}"
    )

    return response.text


# insert_document("Invoice 001 from Acme Inc for $5000 for consulting services", str(uuid.uuid4()))
# insert_document("Lease agreement valid for 12 months starting March 1st", str(uuid.uuid4()))
# insert_document("Your payment of $3200 has been confirmed", str(uuid.uuid4()))
# answer_question("my car Radiator doesn't work") 


# print(answer_question("¿Cuánto facturó Acme Inc?"))
# print(answer_question("¿Cuál es la duración del contrato?"))
# print(answer_question("¿Fue confirmado el pago?"))
