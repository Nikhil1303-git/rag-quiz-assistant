# Quick Start Guide

Get up and running with the RAG LangChain ChromaDB project in 5 minutes.

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- An API key for at least one LLM provider (Groq, OpenAI, or Lamini)

## 1. Setup (2 minutes)

```bash
# Clone or navigate to project directory
cd rag_langchain_chromadb

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Configuration (1 minute)

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your credentials
# Open .env in your editor and fill in:
# - LLM_PROVIDER (groq, openai, or lamini)
# - LLM_API_KEY (your actual API key)
```

### Get a Free API Key

**Option 1: Groq (Recommended - Free tier available)**
- Visit: https://console.groq.com/
- Sign up/login
- Get your API key
- Paste into LLM_API_KEY in .env

**Option 2: OpenAI (Requires payment)**
- Visit: https://platform.openai.com/api-keys
- Create account with payment method
- Generate API key
- Paste into .env

**Option 3: Lamini (Sign up required)**
- Visit: https://www.lamini.ai/
- Create account
- Get API key
- Paste into .env

## 3. Prepare Data (1 minute)

```bash
# Create data directories (if not exists)
mkdir -p data/input data/output

# Copy your PDF files to data/input/
# Example: Copy "mydocument.pdf" to data/input/
```

## 4. Run (1 minute)

### Open the browser dashboard

Install the dependencies, then start the local web app:

```bash
python web_server.py
```

Open http://127.0.0.1:5000 in your browser. The dashboard lets you upload a PDF,
watch ingestion results, check pipeline health, and ask questions against the
indexed document. Keep the terminal open while using it so backend errors are
visible in the server log.

### Ingest a PDF

```bash
python main.py ingest --input data/input/mydocument.pdf
```

Expected output:
```
✓ Successfully ingested PDF
  Pages: 10
  Chunks: 45
  Collection: documents
```

### Ask a Question

```bash
python main.py query --query "What is the main topic?"
```

Expected output:
```
============================================================
Query: What is the main topic?
============================================================

Retrieved 5 documents:

[Document 1]
The document discusses artificial intelligence and machine learning...

============================================================
Response:
============================================================
The main topic of the document is exploring applications of artificial
intelligence in modern business contexts, with a focus on machine learning
techniques...
```

### Check Status

```bash
python main.py status
```

## Complete Example

```bash
# Step 1: Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Step 2: Configure
cp .env.example .env
# Edit .env and add your LLM_API_KEY

# Step 3: Prepare data
mkdir -p data/input
# Copy your PDF to data/input/

# Step 4: Ingest
python main.py ingest --input data/input/myfile.pdf

# Step 5: Query
python main.py query --query "What are the key findings?"

# Step 6: More queries
python main.py query --query "Who are the authors?"
python main.py query --query "What is the methodology?" --top-k 10

# Step 7: Extract tables (if any)
python main.py extract-tables --input data/input/myfile.pdf --output data/output/tables.json
```

## Common Options

```bash
# Query with more documents retrieved
python main.py query --query "..." --top-k 10

# Query with shorter responses
python main.py query --query "..." --max-tokens 512

# Query with lower temperature (more focused)
python main.py query --query "..." --temperature 0.3

# Recreate collection (delete previous data)
python main.py ingest --input file.pdf --recreate
```

## Troubleshooting

**Error: `LLM_API_KEY environment variable is required`**
- Solution: Make sure you edited `.env` and added your API key

**Error: `PDF file not found`**
- Solution: Make sure the file exists in the path you specified

**Error: `CUDA out of memory` or slow execution**
- Solution: Increase PDF_CHUNK_SIZE in .env (e.g., 1000 instead of 500)

**Error: Connection error to LLM service**
- Solution: Check your API key is correct and has quota/credits remaining

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) for technical details
- Run tests: `pytest tests/`
- Explore CLI help: `python main.py --help`

## File Structure

```
├── main.py                 # Run this to use the CLI
├── requirements.txt        # Install dependencies
├── .env.example           # Copy to .env and edit
├── README.md              # Full documentation
├── app/
│   ├── config.py          # Configuration management
│   ├── rag_pipeline.py    # Main workflow
│   ├── pdf/               # PDF processing
│   ├── llm/               # LLM providers
│   └── embeddings/        # Vector store
├── tests/                 # Test suite
└── data/
    ├── input/             # Put PDFs here
    ├── output/            # Extracted tables
    └── chroma/            # Vector database
```

## Support

- Full documentation: See [README.md](README.md)
- Configuration options: See [.env.example](.env.example)
- Technical details: See [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)
- Run tests: `pytest tests/ -v`

Happy querying! 🚀
