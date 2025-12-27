# ⚖️ SEC Risk Factors RAG System

AI-powered query system for analyzing corporate risk factors from SEC 10-K filings using Retrieval-Augmented Generation (RAG).

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Overview

This system automatically ingests, processes, and analyzes "Item 1A - Risk Factors" sections from SEC 10-K filings. It uses semantic search and local LLMs to answer questions about corporate risks across thousands of companies.

### Key Features

- 📊 **Automated Data Ingestion**: Fetch and process 10-K filings from SEC EDGAR
- 🔍 **Semantic Search**: Find relevant risk factors using sentence transformers
- 🤖 **AI-Powered Q&A**: Query risk factors using local LLMs (Ollama)
- 📈 **Daily Updates**: Automatic ingestion of new filings
- 💻 **Multiple Interfaces**: Jupyter notebooks, Streamlit app, or Python API
- 🔒 **Privacy-First**: All processing done locally, no data sent to cloud APIs

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) (for LLM functionality)

### Installation

````bash

Voici le code complet du README à copier directement:
markdown
# ⚖️ SEC Risk Factors RAG System

AI-powered query system for analyzing corporate risk factors from SEC 10-K filings using Retrieval-Augmented Generation (RAG).

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Overview

This system automatically ingests, processes, and analyzes "Item 1A - Risk Factors" sections from SEC 10-K filings. It uses semantic search and local LLMs to answer questions about corporate risks across thousands of companies.

### Key Features

- 📊 **Automated Data Ingestion**: Fetch and process 10-K filings from SEC EDGAR
- 🔍 **Semantic Search**: Find relevant risk factors using sentence transformers
- 🤖 **AI-Powered Q&A**: Query risk factors using local LLMs (Ollama)
- 📈 **Daily Updates**: Automatic ingestion of new filings
- 💻 **Multiple Interfaces**: Jupyter notebooks, Streamlit app, or Python API
- 🔒 **Privacy-First**: All processing done locally, no data sent to cloud APIs

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) (for LLM functionality)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/LawyerChat.git
cd LawyerChat

# Install dependencies
pip install -e .

# Install Ollama (macOS)
brew install ollama

# Or download from https://ollama.com for other platforms
````

### Initial Setup

```bash
# 1. Pull the LLM model
ollama pull llama3.1

# 2. Start Ollama server (in a separate terminal)
ollama serve

# 3. Ingest initial data (processes 50 companies as a test)
python -c "
from pathlib import Path
from sec_risk import SecClient, init_chroma, load_seen_accessions
from sec_risk.pipeline import ingest_many_ciks, IngestConfig

USER_AGENT = 'YourName your.email@example.com'  # Required by SEC
sec = SecClient(user_agent=USER_AGENT)

BASE = Path('./data')
vectordb = init_chroma(str(BASE / 'chroma_sec'), collection_name='sec_10k_risk_factors')
seen = load_seen_accessions(BASE / 'retrieved_10k' / 'manifest.jsonl')

ciks = sec.get_all_ciks()
cfg = IngestConfig(top_n_per_cik=3, chunk_size=1500, chunk_overlap=200)

df, skips = ingest_many_ciks(
    sec=sec,
    vectordb=vectordb,
    ciks=ciks,
    seen_accessions=seen,
    artifact_dir=BASE / 'retrieved_10k',
    manifest_path=BASE / 'retrieved_10k' / 'manifest.jsonl',
    cfg=cfg,
    limit=50
)

print(f'✓ Ingested {len(df)} filings')
"
```

## 💡 Usage

### Option 1: Streamlit Web App (Recommended)

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Option 2: Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Available notebooks:

- `01_explore_database.ipynb` - Explore ingested data
- `02_semantic_search.ipynb` - Test semantic search
- `03_rag_with_ollama.ipynb` - Interactive RAG queries
- `04_analysis_examples.ipynb` - Advanced analytics

### Option 3: Python API

```python
from pathlib import Path
from sec_risk import init_chroma, ask_with_rag

# Load vector database
BASE = Path("./data")
vectordb = init_chroma(
    str(BASE / "chroma_sec"),
    collection_name="sec_10k_risk_factors"
)

# Query
response = ask_with_rag(
    vectordb,
    "What are the main cybersecurity risks companies face?",
    n_context=5
)

print(response['answer'])

# View sources
for source in response['sources']:
    print(f"- {source['company']} ({source['date']})")
```

## 📊 Data Pipeline

```
SEC EDGAR
    ↓
Fetch 10-K Filings
    ↓
Extract Item 1A (Risk Factors)
    ↓
Text Chunking (1500 chars, 200 overlap)
    ↓
Generate Embeddings (all-MiniLM-L6-v2)
    ↓
Store in ChromaDB
    ↓
Query with RAG + Ollama
```

## 🔄 Daily Ingestion

Set up automatic daily updates:

```bash
# Create cron job (runs daily at 9 AM)
crontab -e

# Add:
0 9 * * * cd /path/to/LawyerChat && /path/to/.venv/bin/python daily_ingest.py >> logs/cron.log 2>&1
```

Or run manually:

```bash
python daily_ingest.py
```

## 📁 Project Structure

```
LawyerChat/
├── src/
│   └── sec_risk/
│       ├── __init__.py           # Package initialization
│       ├── sec_client.py         # SEC EDGAR API client
│       ├── extractors.py         # HTML parsing & Item 1A extraction
│       ├── artifacts.py          # File storage utilities
│       ├── store.py              # ChromaDB management
│       ├── pipeline.py           # Ingestion pipeline
│       └── rag_query.py          # RAG query functions
├── notebooks/
│   ├── 01_explore_database.ipynb
│   ├── 02_semantic_search.ipynb
│   ├── 03_rag_with_ollama.ipynb
│   └── 04_analysis_examples.ipynb
├── data/
│   ├── chroma_sec/              # Vector database (not in git)
│   └── retrieved_10k/           # Downloaded filings (not in git)
├── app.py                        # Streamlit web interface
├── daily_ingest.py              # Daily ingestion script
├── requirements.txt
└── README.md
```

## 🛠️ Configuration

### User Agent

SEC requires a valid user agent with contact info:

```python
USER_AGENT = "YourName your.email@example.com"
```

### Embedding Model

Default: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)

To change:

```python
vectordb = init_chroma(
    persist_dir="./data/chroma_sec",
    collection_name="sec_10k_risk_factors",
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)
```

### LLM Model

Default: `llama3.1`

Other options:

```bash
ollama pull qwen2.5:3b    # Faster, good for RAG
ollama pull mistral       # Alternative
ollama pull gemma2:2b     # Smallest, fastest
```

## 📈 Example Queries

**Cybersecurity risks:**

```
"What are the main cybersecurity threats companies face?"
```

**Supply chain:**

```
"What supply chain disruptions do companies mention?"
```

**Climate change:**

```
"How do companies describe climate-related risks?"
```

**Industry comparison:**

```
"Compare regulatory risks in tech vs financial services"
```

**Trend analysis:**

```
"What AI-related risks have emerged in recent filings?"
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Test specific component
pytest tests/test_extractors.py
```

## 📊 Database Statistics

After ingestion, check your database:

```python
from sec_risk import init_chroma

vectordb = init_chroma("./data/chroma_sec", "sec_10k_risk_factors")

print(f"Total vectors: {vectordb._collection.count()}")

# Get unique companies
results = vectordb._collection.get(include=['metadatas'])
companies = set(m.get('company') for m in results['metadatas'])
print(f"Unique companies: {len(companies)}")
```

## 🐛 Troubleshooting

### "Cannot connect to Ollama"

Make sure Ollama is running:

```bash
ollama serve
```

### "No module named 'sec_risk'"

Install in editable mode:

```bash
pip install -e .
```

### "Rate limit exceeded"

The SEC limits requests to 10 per second. The client automatically handles this with a 0.15s delay between requests.

### "ChromaDB not found"

Initialize the database first:

```bash
python -c "from sec_risk import init_chroma; init_chroma('./data/chroma_sec', 'sec_10k_risk_factors')"
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SEC EDGAR** for providing free access to corporate filings
- **Anthropic** for Claude (used in development)
- **Ollama** for local LLM inference
- **ChromaDB** for vector database
- **Hugging Face** for sentence transformers

## 📧 Contact

Khadija Adnani - khadijaadnani2000@gmail.com

Project Link: [https://github.com/yourusername/LawyerChat](https://github.com/yourusername/LawyerChat)

## 🎓 Technical Details

### Technologies Used

- **Python 3.9+**: Core programming language
- **LangChain**: RAG framework
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: Text embeddings (all-MiniLM-L6-v2)
- **Ollama**: Local LLM inference (llama3.1)
- **BeautifulSoup**: HTML parsing
- **Streamlit**: Web interface
- **Jupyter**: Interactive notebooks
- **Pandas**: Data manipulation

### System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB+ for full database
- **CPU**: Modern multi-core processor (GPU optional)
- **OS**: macOS, Linux, or Windows with WSL

### Performance

- **Query time**: 20-30 seconds per question (CPU)
- **Ingestion rate**: ~100 filings/hour (respects SEC rate limits)
- **Database size**: ~50MB per 1000 filings
- **Embedding dimension**: 384 (all-MiniLM-L6-v2)

## 🔮 Future Enhancements

- [ ] Support for additional SEC forms (10-Q, 8-K, S-1)
- [ ] GPU acceleration for faster inference
- [ ] Multi-language support
- [ ] Export reports to PDF
- [ ] Advanced analytics dashboard
- [ ] REST API endpoint
- [ ] Docker containerization
- [ ] Cloud deployment option

## 📚 Resources

- [SEC EDGAR Documentation](https://www.sec.gov/edgar)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)

---

**Built with ❤️ for financial analysis and corporate risk assessment**
