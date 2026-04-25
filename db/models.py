import chromadb
import uuid
from google import genai
import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI


load_dotenv(find_dotenv())

client_genai = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# client_deepseek = OpenAI(
#     api_key=os.environ.get('DEEPSEEK_API_KEY'),
#     base_url="https://api.deepseek.com")

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
    # was working before
    response = client_genai.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=f"Basándote en estos documentos:\n\n{context}\n\nResponde: {text}"
    )
    
    
    # response = client_deepseek.chat.completions.create(
    #     model="deepseek-v4-pro",
    #     messages=[
    #         {"role": "system", "content": f"Eres el encargado de administrar reportes y entregarlos a la personas encargada, cada vez que te hagan una pregunta debes contestar de forma precisa y directa. Teniendo esto en cuenta: Basándote en estos documentos{context}, responde: {text}"}
    #     ],
    #     stream=False,
    #     reasoning_effort="high",
    #     extra_body={"thinking":{"type": "enabled"}}
    # )

    return response.text


# insert_document("Invoice 001 from Acme Inc for $5000 for consulting services", str(uuid.uuid4()))
# insert_document("Lease agreement valid for 12 months starting March 1st", str(uuid.uuid4()))
# insert_document("Your payment of $3200 has been confirmed", str(uuid.uuid4()))
# answer_question("my car Radiator doesn't work") 


# print(answer_question("¿Cuánto facturó Acme Inc?"))
# print(answer_question("¿Cuál es la duración del contrato?"))
# print(answer_question("¿Fue confirmado el pago?"))
