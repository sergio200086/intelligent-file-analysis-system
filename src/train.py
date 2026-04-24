from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np 
import pandas as pd

def main ():
    #read csv
    df = pd.read_csv("training_data_expanded.csv")
    print(f"{len(df)} examples loaded")

    label_encoder = LabelEncoder()
    df['label_id'] = label_encoder.fit_transform(df['label'])

    print(f"\n Labels mapping: ")
    for i, label in enumerate(label_encoder.classes_):
        print(f"{i} = {label}")
    
    text_train, text_test, labels_train, labels_test = train_test_split(
        df['text'].values.tolist(),
        df['label_id'].values.tolist(),
        test_size=0.2,
        random_state=42,
    )

    print(f"\n✅ Data divided:")
    print(f"   Train: {len(text_train)} examples")
    print(f"   Test: {len(text_test)} examples")

    #use model
    base_model= "distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels = len(label_encoder.classes_), problem_type="single_label_classification")

    def tokenizing(examples):
        return tokenizer(
            examples['text'], 
            truncation = True, 
            max_length = 128,
            padding = 'max_length',
            return_tensors = None
        )
    
    dataset_train = Dataset.from_dict({
        'text': text_train,
        'labels': labels_train
    })

    dataset_test = Dataset.from_dict({
        'text': text_test,
        'labels': labels_test
    })


    #tokenize data
    train_tokenized = dataset_train.map(tokenizing, batched=True)
    test_tokenized = dataset_test.map(tokenizing, batched=True)
    
    training_args = TrainingArguments(
        output_dir="./model_clasificator",       
        num_train_epochs=20,                        
        per_device_train_batch_size=16,            
        per_device_eval_batch_size=16,
        learning_rate=5e-5,                       
        weight_decay=0.01,                         
        eval_strategy="epoch",                    
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=2,
        seed=42,
    )

    def compute_metrics(eval_pred):
        """calculate accuracy after each epoch"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    print("Training...")
    trainer.train()

    print("✅ Fine-tuning conmpleted!")

    eval_result = trainer.evaluate()
    print(f"\n✅ Results in Test Set:")
    print(f"   Accuracy: {eval_result['eval_accuracy']:.2%}")
    print(f"   Loss: {eval_result['eval_loss']:.4f}")

    predictions = trainer.predict(test_tokenized)
    pred_ids = np.argmax(predictions.predictions, axis = 1)
    pred_labels = label_encoder.inverse_transform(pred_ids)
    true_labels = label_encoder.inverse_transform(labels_test)

    print(f"\nPredictions vs True Labels: ")
    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        match = "✅" if true == pred else "❌"
        print(f"    {match} Example {i+1}: True {true}, Predicted = {pred}")

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(true_labels, pred_labels, labels=label_encoder.classes_)
    
    #show matrix 
    print("\n" + " " * 15 + "Predicted")
    print(" " * 5 + " ".join(f"{label:12}" for label in label_encoder.classes_))
    for i, label in enumerate(label_encoder.classes_):
        print(f"{label:12} {' '.join(f'{cm[i][j]:12}' for j in range(len(label_encoder.classes_)))}")

    # 4. GUARDAR MODELO
    print("\n[6/6] Saving model...")
    model.save_pretrained("./model_classificator_final")
    tokenizer.save_pretrained("./model_classificator_final")
    print("✅ Model saved in './model_classificator_final'")


    print("\n" + "=" * 60)
    print("¡FINE-TUNING COMPLETED!")
    print("=" * 60)
    
if __name__ == "__main__":
    main()