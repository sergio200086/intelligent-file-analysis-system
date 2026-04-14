from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def main ():
    data = {
        "text":[
            "Caída total del servidor de base de datos en producción.",
            "El texto del footer tiene un error ortográfico.",
            "Excepción de puntero nulo al procesar pagos con tarjeta.",
            "El color del botón no cambia cuando se activa el modo oscuro.",
            "Latencia de 5000ms en el endpoint de autenticación.",
            "La imagen de perfil tarda un poco en cargar la primera vez."
        ],
        "label":[1,0,1,0,1,0]
    }

    dataset = Dataset.from_dict(data).train_test_split(test_size=0.2)

    base_model= "distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels = 2)

    def tokenizing(examples):
        return tokenizer(examples["text"], padding="max_length", truncation = True, max_length = 128)
    
    tokenized_dataset = dataset.map(tokenizing, batched = True)

    arguments = TrainingArguments(
        output_dir="./my_trained_model",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        num_train_epochs=5,
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    print ("Starting fine-tuning")
    trainer.train()

    trainer.save_model("./final_model_ready")
    tokenizer.save_pretrained("./final_model_ready")
    print("model saved")
    
if __name__ == "__main__":
    main()