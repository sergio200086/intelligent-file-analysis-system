# 📄 Intelligent Document Analysis System

An AI-powered system that automatically classifies documents, extracts information, and answers questions based on document content using Fine-tuning, RAG, and APIs.

## ✨ Features

- **Automatic Document Classification** (98% accuracy)
  - Invoices, Contracts, Emails, Reports
- **RAG (Retrieval-Augmented Generation)** for intelligent queries
  - Search documents by meaning, not keywords
  - Answer questions based on your content
- **Production-Ready REST API**
  - 3 functional endpoints
  - Interactive Swagger documentation
- **Fine-tuned DistilBERT** specialized in your domain

## 🏗️ Architecture

```
Document PDF/Text
    ↓
[Preprocessing]
    ↓
[Fine-tuned Classifier] ← Determines document type
    ↓
[ChromaDB + Embeddings] ← Stores intelligently
    ↓
[RAG Pipeline] ← User asks question
    ↓
[Gemini/LLM] ← Analyzes and responds
    ↓
Intelligent Response
```

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **ML/NLP:** Transformers (DistilBERT), Sentence-Transformers
- **Vector Database:** ChromaDB
- **LLM:** Google Gemini / DeepSeek / Ollama
- **Classification:** Scikit-learn, PyTorch
- **APIs:** Anthropic, Google GenAI

## 📦 Installation

### Requirements

- Python 3.10+
- pip
- Git

### Steps

1. **Clone the repository**

```bash
git clone https://github.com/sergio200086/intelligent-file-analysis-system
cd intelligent-document-analysis
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
   Create a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```

5. **Run the API**

```bash
uvicorn routes.api:app --reload
```

6. **Access documentation**
   Open in your browser: `http://localhost:8000/docs`

## 🚀 Usage

### 1. Classify a Document

**Endpoint:** `POST /predict-type`

```bash
curl -X POST "http://localhost:8000/predict-type?document=Invoice%20001%20from%20Acme%20Inc%20for%20%245000"
```

**Response:**

```json
{
  "classification": "invoice",
  "confidence": "95.32%",
  "probs": {
    "contract": "2.15%",
    "email": "1.23%",
    "invoice": "95.32%",
    "report": "1.30%"
  }
}
```

### 2. Upload a Document

**Endpoint:** `POST /upload-document`

```bash
curl -X POST "http://localhost:8000/upload-document?document=Invoice%20001%20from%20Acme%20Inc%20for%20%245000"
```

**Response:**

```json
{
  "status": "ok"
}
```

### 3. Ask a Question (RAG)

**Endpoint:** `POST /ask`

```bash
curl -X POST "http://localhost:8000/ask?question=How%20much%20did%20Acme%20Inc%20invoice"
```

**Response:**

```json
{
  "answer": "Acme Inc invoiced $5000 for consulting services according to Invoice 001.",
  "status": "ok"
}
```

## 📊 Results

### Classifier Fine-tuning Performance

| Metric            | Value                                |
| ----------------- | ------------------------------------ |
| Test Set Accuracy | **98.08%**                           |
| Training Data     | 260 examples                         |
| Categories        | 4 (Invoice, Contract, Email, Report) |
| Base Model        | DistilBERT Multilingual              |
| Training Epochs   | 20                                   |
| Final Loss        | 0.0048                               |

### Confusion Matrix

```
Correct predictions per category:
- Invoices: 12/13 (92%)
- Contracts: 16/16 (100%)
- Emails: 12/12 (100%)
- Reports: 11/11 (100%)
```

## 📁 Project Structure

```
intelligent-file-analysis-system/
├── src/
│   └── predictor.py              # Fine-tuned classifier
├── db/
│   ├── models.py                 # RAG + Vector DB
│   └── my_models/                # ChromaDB storage
├── routes/
│   └── api.py                    # FastAPI endpoints
├── modelo_clasificador_final/    # Trained model
├── training_data_expanded.csv    # Training dataset (260 examples)
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables
└── README.md                     # This file
```

## 🧠 How It Works

### 1. Fine-tuning (Training)

- Take DistilBERT (pre-trained model)
- Train it with 260 real document examples
- After 20 epochs, achieved 98% accuracy
- Save the trained model in `modelo_clasificador_final/`

### 2. Classification (Prediction)

- User uploads a document
- Pass it through the fine-tuned classifier
- Returns: document type + confidence + probabilities

### 3. RAG (Intelligent Answers)

- Documents are converted to embeddings (numbers representing meaning)
- Stored in ChromaDB (vector database)
- User asks a question
- System finds similar documents by meaning
- Passes context to the LLM (Gemini/DeepSeek)
- LLM analyzes and responds

## 🔑 Key Concepts

### Fine-tuning

Specialization of a pre-trained model with your own data. In this case, DistilBERT learned to classify specific business documents.

### RAG (Retrieval-Augmented Generation)

Combination of intelligent search + text generation. The LLM doesn't generate from memory, but based on relevant documents it finds.

### Vector Database

Database that stores embeddings (numeric representations of meaning). Allows semantic search instead of exact keyword matching.

## 📈 Future Improvements

- [ ] Streamlit dashboard for visual interface
- [ ] NER (Named Entity Recognition) for data extraction
- [ ] Multi-language support
- [ ] WhatsApp/Facebook Messenger integration
- [ ] SQL database for document persistence
- [ ] Authentication and authorization
- [ ] Rate limiting and caching
- [ ] Sentiment analysis
- [ ] Automatic alerts

## 🤝 Contributing

Contributions are welcome. For major changes, please open an issue first.

## 📄 License

MIT License - Feel free to use and modify

**Built with ❤️ in Bogotá, Colombia**

## 📚 Learning Documentation

This project was developed as part of an AI and Machine Learning learning journey. It documents:

- Fine-tuning Transformer models
- RAG implementation from scratch
- Vector Database integration
- REST API creation with FastAPI
- ML model evaluation

---

**Last Updated:** April 2024
