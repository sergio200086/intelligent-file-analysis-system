import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./final_model_ready"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

test_text = "La aplicación se cierra sola cuando intento subir una foto de perfil."

inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
    
with torch.no_grad():
    outputs = model(**inputs)
    prediccion = torch.argmax(outputs.logits, dim=-1).item()

severity = "Crítico" if prediccion == 1 else "Menor"
print(f"Received report: '{test_text}'")
print(f"AI prediction: {severity}")