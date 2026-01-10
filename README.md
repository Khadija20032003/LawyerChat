# 📊 SEC Risk Factors RAG Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot that helps users explore and understand **SEC 10-K risk factor disclosures** through natural language conversations.

Built with LangChain, ChromaDB, and Gradio, this application combines semantic search over SEC filings with large language models to provide accurate, context-aware answers about corporate risk factors.

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/FatimaZh/LawyerChat)

---

## ✨ Features

- 🔍 **Semantic Search** over SEC 10-K risk factor sections
- 🤖 **LLM-Powered Answers** using Retrieval-Augmented Generation
- 📄 **PDF Upload Support** - Query your own documents alongside SEC data
- 🎯 **Dual-Source Retrieval** - Search both SEC database and uploaded PDFs simultaneously
- 💬 **Interactive Chat Interface** built with Gradio
- 🔄 **Automated Dataset Updates** via GitHub Actions
- 🧱 **Clean Architecture** - Domain logic separated from UI
- ⚡ **Lightweight Models** - Runs on CPU with 1B parameter LLM
- 🔒 **Privacy-Focused** - All processing happens locally/in-space

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       User Query                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Query Embedding (all-MiniLM-L6-v2)              │
└────────────────────┬────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐  ┌──────────────────────┐
│  SEC Database    │  │  Uploaded PDF        │
│  (ChromaDB)      │  │  (Temporary Store)   │
│  Persistent      │  │  In-Memory           │
└─────────┬────────┘  └──────────┬───────────┘
          │                      │
          └──────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  Top-K Documents      │
         │  (k=5)      │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  Context Formation    │
         │  + Prompt Template    │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  LLM Generation       │
         │  (Llama-3.2-1B)      │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  Generated Answer     │
         └───────────────────────┘
```

### Component Breakdown

**Data Layer**
- **ChromaDB Vector Store**: Persistent database of pre-processed SEC 10-K risk factors
- **PDF Processing Pipeline**: On-the-fly text extraction, chunking, and indexing

**Model Layer**
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional)
- **LLM**: `unsloth/Llama-3.2-1B-Instruct` (1B parameters, instruction-tuned)

**Processing Layer**
- **RAG Chain**: Retrieval → Context formation → Prompted generation
- **Dual-Source Search**: Parallel queries to SEC database and uploaded PDFs

**Interface Layer**
- **Gradio Blocks UI**: Chat interface with file upload and status indicators

---



## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- (Optional) CUDA-compatible GPU for faster inference

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/FatimaZh/LawyerChat.git
cd LawyerChat
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python chatbot/app.py
```

The Gradio interface will launch automatically in your browser at `http://localhost:7860`

### First Run Behavior

On first launch, the application will:
1. Download the SEC 10-K vector database from HuggingFace (~100MB)
2. Load the embeddings model (~80MB)
3. Load the LLM model (~2.5GB)

**This may take 5-10 minutes depending on your connection speed.**

Subsequent runs will be faster as models are cached locally.

---

## 🔧 Configuration

All settings are in `config.py`:

### Models

```python
LLM_MODEL = "unsloth/Llama-3.2-1B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### Generation Parameters

```python
MAX_NEW_TOKENS = 512      # Maximum length of generated answers
TEMPERATURE = 0.7         # Sampling temperature 
TOP_P = 0.9              # Nucleus sampling threshold
RETRIEVAL_K = 5          # Number of documents to retrieve
```

### Device Selection

```python
DEVICE = "cpu"  # or "cuda"
```

---

## 💡 Usage Examples

### Basic Queries

Ask questions about SEC risk factors:

```
❓ What are the main cybersecurity risks companies report?
❓ Which companies mention climate change in their risk factors?
❓ What regulatory risks do financial institutions face?
❓ Tell me about supply chain disruption risks
❓ What emerging technology risks are reported?
```

### PDF Upload Workflow

1. Click **"Upload Your PDF"** in the left panel
2. Select a text-based PDF document
3. Wait for processing confirmation
4. Ask questions that combine both sources:
   ```
   ❓ How do the risks in my contract compare to SEC filings?
   ❓ Does my company's risk disclosure align with industry standards?
   ```

---

## 🧠 How It Works

### Retrieval-Augmented Generation (RAG)

Instead of relying solely on the LLM's training data:

1. **Embed Query** → Convert question to 384-dimensional vector
2. **Search Database** → Find semantically similar text chunks (cosine similarity)
3. **Retrieve Top-K** → Get most relevant passages (k=5 or 4+1 for dual-source)
4. **Format Context** → Structure retrieved text for the prompt
5. **Generate Answer** → LLM produces response grounded in retrieved documents

**Benefits**:
- ✅ **Accuracy**: Answers based on actual documents, not hallucinations
- ✅ **Transparency**: Can trace answers back to source filings
- ✅ **Up-to-date**: Database can be refreshed with new filings
- ✅ **Domain-specific**: Focused expertise on SEC risk factors


---

## 🔄 Automated Updates

The system includes automated dataset refreshes via GitHub Actions.

### Workflow: `.github/workflows/daily_updta.yml`

Triggers daily (configurable) to:
1. Fetch latest SEC 10-K filings
2. Extract risk factors using `sec_risk_factors/`
3. Update vector database
4. Push to HuggingFace dataset

---

## 📊 Technical Details

### Vector Database

- **Backend**: ChromaDB (SQLite + HNSW index)
- **Collection**: `sec_10k_risk_factors`
- **Source**: HuggingFace Dataset `FatimaZh/sec-10k-chroma`
- **Persistence**: Local disk (`./data/chroma_sec/`)

### Text Processing

- **PDF Extraction**: `pdfplumber` 
- **Chunking Strategy**: RecursiveCharacterTextSplitter
  - Chunk size: 1000 characters
  - Overlap: 200 characters
- **Embedding Dimension**: 384

---

## 🌐 Deployment

### HuggingFace Spaces

This app is designed to run on HuggingFace Spaces:

1. Create a new Space with **Gradio SDK**
2. Upload all files to the Space
3. Space will auto-deploy

**Important**: On HuggingFace Spaces free tier, the database is re-downloaded on every cold start, which can take several minutes.

### Local Deployment

The application runs on `localhost:7860` by default. To change:

```python
demo.launch(
    server_name="0.0.0.0",  # Allow external connections
    server_port=7860,
    share=True  
)
```

---

## 🛠️ Tech Stack

- **Framework**: [LangChain](https://python.langchain.com/) 0.3+
- **UI**: [Gradio](https://gradio.app/) 4.40+
- **LLM**: [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- **Embeddings**: [Sentence-Transformers](https://www.sbert.net/)
- **Vector DB**: [ChromaDB](https://www.trychroma.com/)
- **PDF Processing**: [pdfplumber](https://github.com/jsvine/pdfplumber)


---

## 📝 Example Use Cases

### For Investors
- Compare risk disclosures across companies
- Track emerging risk trends over time
- Due diligence on specific sectors

### For Compliance Teams
- Benchmark your risk disclosures against peers
- Identify common risk language patterns
- Research regulatory risk trends

### For Researchers
- Analyze corporate risk communication
- Study industry-specific risk factors
- Dataset for NLP/finance research

---

## 📄 License

This project is provided for **educational and research purposes**.

SEC 10-K filings are public domain documents (U.S. government data).

---

## 👥 Authors

### Fatima Zohra  

---

### Khadija Eladnani  



---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star!**

</div>