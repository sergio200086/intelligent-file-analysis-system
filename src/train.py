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
import numpy as np 
import pandas as pd

def main ():

    df = pd.read_csv("training_data.csv")
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

    # dataset = Dataset.from_dict(data).train_test_split(test_size=0.2)

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

    train_tokenized = dataset_train.map(tokenizing, batched=True)
    test_tokenized = dataset_test.map(tokenizing, batched=True)
    

    training_args = TrainingArguments(
        output_dir="./model_clasificator",       
        num_train_epochs=3,                        
        per_device_train_batch_size=8,            
        per_device_eval_batch_size=8,
        learning_rate=2e-5,                       
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
        """Calcula accuracy después de cada epoch"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    # Crear Trainer
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


    print("\n" + "=" * 60)
    print("¡FINE-TUNING COMPLETED!")
    print("=" * 60)
    
if __name__ == "__main__":
    main()