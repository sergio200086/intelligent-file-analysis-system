import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("./model_classificator_final")
model = AutoModelForSequenceClassification.from_pretrained("./model_classificator_final", dtype="auto", device_map="auto")
print("✅ Model loaded successfully")

labels_map = {
    0: "contract",
    1:"email",
    2:"invoice",
    3: "report"
}

def classify_document(text):
    inputs = tokenizer(
        text,
        truncation = True,
        max_length = 128,
        padding = 'max_length',
        return_tensors = "pt"
    )

    with torch.no_grad():
        output = model(**inputs)

    logits = output.logits
    max_value = torch.argmax(logits, dim= 1).item()
    label = labels_map[max_value] # type: ignore

    probs = torch.softmax(logits, dim=1)[0]
    confidence = probs[max_value].item() # type: ignore

    return {
        "classification": label,
        "confidence": f"{confidence:.2%}",
        "probs": {
            labels_map[i]: f"{prob:.2%}" for i, prob in enumerate(probs)
        }
    }



if __name__ == "__main__":
    print("=" * 60)
    print("TESTING CLASSIFIER")
    print("=" * 60)

    examples =[
        "Invoice 001 from Acme Inc for $5000 for consulting services",
        "Lease agreement valid for 12 months starting March 1st",
        "Your payment of $3200 has been confirmed",
        "Quarterly report: Sales analysis for Q1 2024",
        "This is a totally new document I've never seen before",
        "Just in at NielsenIQ: This week's employee reviews and more",
        "Updating Netlify's privacy statement"
    ]

    for i, doc in enumerate(examples, 1):
        print(f"The input is: {doc}")
        output = classify_document(doc)
        print(f"Output: {output["classification"]}")
        print(f"confidence: {output["confidence"]}")
        print(f"All probabilities: ")        
        for label, prob in output["probs"].items():
            print(f"{label} - {prob}")