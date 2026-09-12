# RAG with LangChain and ChromaDB

A production-ready Retrieval Augmented Generation (RAG) system that combines document processing, vector embeddings, and large language models (LLMs) to answer questions based on PDF content.

## 📋 Project Overview

This project implements a complete RAG pipeline that:

1. **Extracts** text and tables from PDF documents
2. **Chunks** text intelligently while preserving context
3. **Embeds** chunks using semantic models
4. **Stores** embeddings in a vector database (ChromaDB)
5. **Retrieves** relevant documents based on queries
6. **Generates** contextual LLM responses based on retrieved documents

### Key Features

- ✅ **Multiple LLM Providers**: Groq, OpenAI, Lamini (easily switchable via config)
- ✅ **Flexible PDF Extraction**: pdfplumber and pypdf support
- ✅ **Robust Text Chunking**: Recursive character splitting with overlap
- ✅ **Vector Storage**: ChromaDB with configurable embedding models
- ✅ **CLI Interface**: Full command-line support for all operations
- ✅ **Browser Dashboard**: Local HTML/CSS interface for upload, status, and queries
- ✅ **Modular Architecture**: Clean separation of concerns
- ✅ **Error Handling**: Comprehensive logging and exception handling
- ✅ **Configuration Management**: Environment-based configuration
- ✅ **Production Ready**: Security best practices (no hardcoded secrets)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                             │
└─────────────────────────────────────────────────────────────┘
              │
              ├─→ PDF Extraction (pdfplumber/pypdf)
              │     └─→ Text Extractor / Table Extractor
              │
              ├─→ Text Chunking (LangChain)
              │     └─→ RecursiveCharacterTextSplitter
              │
              ├─→ Embedding Generation (Sentence Transformers)
              │     └─→ all-MiniLM-L6-v2 (or custom model)
              │
              ├─→ Vector Storage (ChromaDB)
              │     └─→ Persistent document embeddings
              │
              ├─→ Retrieval (Semantic Search)
              │     └─→ Top-K document retrieval
              │
              └─→ LLM Response Generation
                    ├─→ Groq (fast inference)
                    ├─→ OpenAI (GPT models)
                    └─→ Lamini (custom models)
```

### Project Structure

```
rag_langchain_chromadb/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── utils.py                  # Logging utilities
│   ├── text_chunker.py           # Text chunking logic
│   ├── rag_pipeline.py           # Main RAG orchestration
│   ├── pdf/
│   │   ├── __init__.py
│   │   ├── text_extractor.py     # PDF text extraction
│   │   └── table_extractor.py    # PDF table extraction
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py             # LLM provider abstractions
│   └── embeddings/
│       ├── __init__.py
│       └── chroma_store.py       # ChromaDB vector store
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_pdf_extraction.py
│   ├── test_text_chunker.py
│   ├── test_vector_store.py
│   ├── test_llm_client.py
│   └── test_rag_pipeline.py
├── data/
│   ├── input/                    # Input PDF files
│   ├── output/                   # Output files
│   └── chroma/                   # ChromaDB persistence
├── .env.example                  # Environment template
├── .gitignore
├── main.py                       # CLI entry point
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository_url>
cd rag_langchain_chromadb

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Browser dashboard

Start the local web interface after installation:

```bash
python web_server.py
```

Then open http://127.0.0.1:5000. Upload a PDF in the **Ingest** panel and ask
questions in the **Query** panel. The status panel reflects the live pipeline,
including whether ChromaDB has indexed any chunks.

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# Required:
# - LLM_PROVIDER (groq, openai, or lamini)
# - LLM_API_KEY
# - LLM_MODEL (optional, has defaults)
```

### 3. Get API Keys

Choose one LLM provider:

**Groq** (Recommended - Fast and free tier available)
- Get key: https://console.groq.com/
- Free tier includes usage limits
- Model: `llama3-8b-8192`

**OpenAI**
- Get key: https://platform.openai.com/api-keys
- Requires paid account
- Model: `gpt-3.5-turbo` or `gpt-4`

**Lamini**
- Get key: https://www.lamini.ai/
- Requires account setup
- Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`

### 4. Example Usage

```bash
# Create data directories
mkdir -p data/input data/output

# Copy your PDF to data/input/
cp your_document.pdf data/input/

# Ingest PDF into vector store
python main.py ingest --input data/input/your_document.pdf

# Query the documents
python main.py query --query "What is the main topic of the document?"

# Extract tables (if any)
python main.py extract-tables --input data/input/your_document.pdf --output data/output/tables.json

# Check pipeline status
python main.py status
```

## 🔧 Configuration

### Environment Variables

All configuration is managed through environment variables in `.env`:

#### LLM Provider
```bash
LLM_PROVIDER=groq              # "groq", "openai", or "lamini"
LLM_API_KEY=<your_api_key>
LLM_MODEL=llama3-8b-8192      # Provider-specific model ID
LLM_TEMPERATURE=0.7            # Sampling temperature (0-2)
LLM_MAX_TOKENS=2048            # Max response length
```

#### Embeddings
```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=32
```

#### ChromaDB
```bash
CHROMA_COLLECTION_NAME=documents
CHROMA_PERSIST_DIR=./data/chroma
```

#### PDF Processing
```bash
PDF_EXTRACTION_METHOD=pdfplumber    # "pdfplumber" or "pypdf"
PDF_CHUNK_SIZE=500                  # Characters per chunk
PDF_CHUNK_OVERLAP=0                 # Overlap between chunks
```

#### Directories
```bash
INPUT_DIR=./data/input
OUTPUT_DIR=./data/output
LOG_LEVEL=INFO                      # DEBUG, INFO, WARNING, ERROR
```

### Customization

#### Change Text Chunk Size
```bash
PDF_CHUNK_SIZE=1000          # Larger chunks for broad context
PDF_CHUNK_OVERLAP=100        # Overlap to preserve boundaries
```

#### Change Embedding Model
```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2    # Fast, general-purpose
# Or:
EMBEDDING_MODEL=all-mpnet-base-v2   # Better quality, slower
```

#### Use Different LLM Provider
```bash
# For OpenAI
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL=gpt-3.5-turbo

# For Lamini
LLM_PROVIDER=lamini
LLM_API_KEY=...
LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
```

## 📖 CLI Usage

### Ingest Command

Ingest a PDF document into the vector store:

```bash
python main.py ingest --input document.pdf
python main.py ingest --input document.pdf --recreate  # Recreate collection
```

Options:
- `--input`: Path to PDF file (required)
- `--recreate`: Delete and recreate collection (optional)

### Query Command

Query the ingested documents:

```bash
python main.py query --query "What is the main topic?"
python main.py query --query "..." --top-k 10  # Retrieve 10 documents
python main.py query --query "..." --temperature 0.5  # Lower temperature
python main.py query --query "..." --max-tokens 1024  # Shorter responses
```

Options:
- `--query`: Query text (required)
- `--top-k`: Number of documents to retrieve (default: 5)
- `--temperature`: LLM temperature 0-2 (default: 0.7)
- `--max-tokens`: Max response tokens (default: 2048)

### Extract Tables Command

Extract structured data from PDFs:

```bash
python main.py extract-tables --input document.pdf
python main.py extract-tables --input document.pdf --output tables.json
```

Options:
- `--input`: Path to PDF file (required)
- `--output`: Output JSON file path (optional)

### Status Command

Show pipeline configuration and status:

```bash
python main.py status
```

## 🧪 Testing

Run unit tests:

```bash
# All tests
pytest

# Specific test file
pytest tests/test_config.py

# With coverage
pytest --cov=app tests/

# Verbose output
pytest -v
```

### Test Files

- `test_config.py`: Configuration loading and validation
- `test_pdf_extraction.py`: PDF text and table extraction
- `test_text_chunker.py`: Text chunking logic
- `test_vector_store.py`: ChromaDB operations
- `test_llm_client.py`: LLM provider clients
- `test_rag_pipeline.py`: Complete pipeline integration

## 🔐 Security

### API Key Management

⚠️ **NEVER commit `.env` to version control!**

1. `.env` is in `.gitignore` - won't be accidentally committed
2. Use `.env.example` as template with placeholder values
3. Copy to `.env` and fill in real credentials locally only
4. Each user/environment has their own `.env`

### Best Practices

- Use environment variables for all secrets
- Rotate API keys regularly
- Use read-only API keys when possible
- Monitor API usage for unauthorized access
- Don't log API keys or responses containing sensitive data

## 🐛 Troubleshooting

### Common Issues

**Issue**: `LLM_API_KEY environment variable is required`
```
Solution: Create .env file from .env.example and add your API key
```

**Issue**: `FileNotFoundError: PDF file not found`
```
Solution: Check file path is correct and file exists in data/input/
```

**Issue**: `chromadb import error`
```
Solution: pip install chromadb
```

**Issue**: `pdfplumber fails on certain PDFs`
```
Solution: Try PDF_EXTRACTION_METHOD=pypdf as fallback
```

**Issue**: LLM response is slow
```
Solution: Reduce PDF_CHUNK_SIZE or use faster LLM provider (Groq)
```

**Issue**: Out of memory with large PDFs
```
Solution: 
- Increase PDF_CHUNK_SIZE to fewer chunks
- Reduce PDF_CHUNK_OVERLAP
- Process PDFs in batches
```

### Debug Mode

Enable debug logging:

```bash
LOG_LEVEL=DEBUG python main.py status
```

### Check Configuration

```bash
python -c "from app.config import load_config; print(load_config())"
```

## 📚 Workflow Examples

### Example 1: Simple Question Answering

```bash
# 1. Place PDF in data/input/
# 2. Ingest it
python main.py ingest --input data/input/document.pdf

# 3. Ask questions
python main.py query --query "What are the key findings?"
python main.py query --query "Who are the authors?"
python main.py query --query "What is the methodology?"
```

### Example 2: Tabular Data Extraction

```bash
# Extract tables from PDF
python main.py extract-tables \
  --input data/input/report.pdf \
  --output data/output/tables.json

# Use tables in external application
# (CSV, database import, etc.)
```

### Example 3: Custom Analysis

```bash
# Python script using pipeline directly
from app.config import load_config, validate_config
from app.pdf.text_extractor import PDFTextExtractor
from app.text_chunker import TextChunker
from app.embeddings.chroma_store import ChromaVectorStore
from app.llm.client import LLMFactory
from app.rag_pipeline import RAGPipeline

config = load_config()
validate_config(config)

# Initialize pipeline
pipeline = RAGPipeline(
    pdf_extractor=PDFTextExtractor(config.pdf.extraction_method),
    text_chunker=TextChunker(config.pdf.chunk_size, config.pdf.chunk_overlap),
    vector_store=ChromaVectorStore(
        config.chroma.collection_name,
        config.chroma.persist_directory,
        config.chroma.embedding_model
    ),
    llm_client=LLMFactory.create(
        config.llm.provider,
        config.llm.api_key,
        config.llm.model
    )
)

# Use pipeline
pipeline.ingest_pdf("document.pdf")
result = pipeline.rag_query("Your question here?")
print(result["response"])
```

## 🔄 Workflow Overview

### Document Ingestion

```
PDF File
  ↓
Text Extraction (pdfplumber/pypdf)
  ↓
Text Chunking (RecursiveCharacterTextSplitter)
  ├─ Splits by: paragraphs → sentences → words
  ├─ Preserves context with overlap
  └─ Creates semantic chunks
  ↓
Embedding Generation (Sentence Transformers)
  ├─ Converts text to vectors
  └─ Captures semantic meaning
  ↓
Vector Storage (ChromaDB)
  ├─ Indexes embeddings
  └─ Enables fast retrieval
```

### Query Execution

```
User Query
  ↓
Embedding Generation (same model)
  ↓
Semantic Search (cosine similarity)
  ├─ Retrieves top-K similar chunks
  └─ Ranks by relevance
  ↓
Context Construction
  ├─ Combines retrieved documents
  └─ Builds prompt with context
  ↓
LLM Generation
  ├─ Groq: Fast, free tier available
  ├─ OpenAI: High quality, paid
  └─ Lamini: Customizable, flexible
  ↓
Response
```

## 🛠️ Development

### Adding a New LLM Provider

1. Create client class in `app/llm/client.py`:

```python
class MyLLMClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        # Initialize client
        pass
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Generate response
        pass
    
    def get_model_info(self) -> dict:
        # Return model info
        pass
```

2. Register in `LLMFactory.PROVIDERS`:

```python
PROVIDERS = {
    "groq": GroqClient,
    "openai": OpenAIClient,
    "lamini": LaminiClient,
    "myprovider": MyLLMClient,  # Add here
}
```

3. Update `.env.example` with new provider config

4. Add tests in `tests/test_llm_client.py`

### Adding a New Extraction Method

1. Create extractor in `app/pdf/text_extractor.py`:

```python
@staticmethod
def _extract_with_mymethod(pdf_path: Path) -> List[str]:
    # Implementation
    pass
```

2. Update `extract_text()` method to support new method

3. Add configuration option to config.py

4. Add tests

## 📊 Performance Considerations

### Optimization Tips

| Aspect | Slow | Fast |
|--------|------|------|
| **LLM Provider** | OpenAI | Groq |
| **Chunk Size** | 500 | 1000+ |
| **Embedding Model** | all-mpnet-base-v2 | all-MiniLM-L6-v2 |
| **Retrieval Count** | 10+ | 3-5 |
| **Temperature** | 0 (deterministic) | 0.7+ (creative) |

### Scaling

For large document collections:

1. **Batch Processing**: Ingest PDFs in parallel
2. **Pagination**: Retrieve results in pages
3. **Filtering**: Use metadata for pre-filtering
4. **Caching**: Cache embeddings and responses
5. **Indexing**: ChromaDB auto-indexes for speed

## 📖 API Reference

### RAGPipeline

```python
pipeline = RAGPipeline(pdf_extractor, text_chunker, vector_store, llm_client)

# Ingest PDF
pipeline.ingest_pdf(pdf_path, collection_name=None, force_recreate=False)

# Retrieve documents
documents = pipeline.retrieve(query, n_results=5)

# Generate response
response = pipeline.generate_response(query, retrieved_documents)

# Full RAG query
result = pipeline.rag_query(query, n_retrieve=5, temperature=0.7)

# Get status
status = pipeline.get_status()
```

### PDFTextExtractor

```python
extractor = PDFTextExtractor(method="pdfplumber")

# Extract texts
texts = extractor.extract_text(pdf_path)

# Extract with metadata
texts_meta = extractor.extract_text_with_pages(pdf_path)
```

### TextChunker

```python
chunker = TextChunker(chunk_size=500, chunk_overlap=0)

# Chunk single text
chunks = chunker.chunk_text(text)

# Chunk multiple texts
all_chunks = chunker.chunk_texts(texts)

# Chunk with metadata
chunks_meta = chunker.chunk_with_metadata(text, source="doc.pdf", page=1)
```

### ChromaVectorStore

```python
store = ChromaVectorStore(
    collection_name="documents",
    persist_directory="./data/chroma"
)

# Create collection
store.create_collection(force_recreate=False)

# Add documents
store.add_documents(texts, ids=None, metadata=None)

# Query
results = store.query(query_texts, n_results=5)

# Get info
info = store.get_collection_info()

# Clear/delete
store.clear_collection()
store.delete_collection()
```

## 📝 Configuration Reference

See `.env.example` for all available configuration options with descriptions.

## 🤝 Contributing

Contributions welcome! Please:

1. Create a feature branch
2. Make changes with tests
3. Submit pull request
4. Follow existing code style

## 📄 License

[Add your license here]

## 📞 Support

For issues and questions:

1. Check troubleshooting section
2. Review debug logs (`LOG_LEVEL=DEBUG`)
3. Check configuration with `python main.py status`
4. Open an issue with error details and configuration

## 🎯 Roadmap

- [ ] Web interface (Gradio/Streamlit)
- [ ] Batch processing CLI
- [ ] Advanced filtering and metadata search
- [ ] Response caching
- [ ] Multi-document comparison
- [ ] Citation generation
- [ ] Conversation history
- [ ] Fine-tuning pipelines
- [ ] API server (FastAPI)
- [ ] Docker containerization

## 🙏 Acknowledgments

- LangChain for document processing framework
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- All LLM providers (Groq, OpenAI, Lamini)
- pdfplumber and pypdf for PDF processing

---

**Last Updated**: 2024
**Version**: 1.0.0
