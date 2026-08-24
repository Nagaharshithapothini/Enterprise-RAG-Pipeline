# Enterprise RAG Pipeline

A production-style **Retrieval-Augmented Generation (RAG)** application built with Python, FastAPI, Sentence Transformers, ChromaDB, Cross-Encoder reranking, and an LLM.

The project demonstrates an end-to-end enterprise document intelligence workflow including document ingestion, text processing, semantic chunking, embedding generation, vector indexing, semantic retrieval, reranking, grounded response generation, and source attribution.

---

## Architecture

```text
                   ┌──────────────────┐
                   │  PDF / TXT File  │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Document Loader  │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Cleaning &       │
                   │ Chunking         │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Sentence         │
                   │ Embeddings       │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ ChromaDB Vector  │
                   │ Store            │
                   └────────┬─────────┘
                            │
User Question ──────────────┤
                            ▼
                   ┌──────────────────┐
                   │ Semantic Vector  │
                   │ Retrieval        │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Cross-Encoder    │
                   │ Reranking        │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Context          │
                   │ Construction     │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Large Language   │
                   │ Model            │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Grounded Answer  │
                   │ + Sources        │
                   └──────────────────┘
```

---

## Features

- PDF and TXT document ingestion
- Text cleaning and chunking
- Chunk overlap for context preservation
- Metadata preservation
- Sentence Transformer embeddings
- ChromaDB persistent vector database
- Semantic similarity retrieval
- Cross-Encoder reranking
- Context-aware response generation
- Source attribution
- Hallucination-reduction prompt controls
- FastAPI REST endpoints
- Swagger API documentation
- Environment-based configuration

---

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| API | FastAPI |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| Reranking | Cross-Encoder |
| LLM Integration | OpenAI API |
| Document Processing | PyPDF |
| Configuration | python-dotenv |
| API Validation | Pydantic |

---

## Project Structure

```text
enterprise-rag-pipeline/
│
├── api/
│   └── main.py
│
├── data/
│   └── sample_document.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   ├── chunking.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── reranker.py
│   ├── generation.py
│   └── pipeline.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## RAG Workflow

### 1. Document Ingestion

PDF and TXT documents are loaded and converted into structured text records.

Metadata such as the source document and PDF page number is preserved.

### 2. Text Processing

The extracted text is cleaned and divided into overlapping chunks.

Chunk overlap helps preserve contextual information across chunk boundaries.

### 3. Embedding Generation

Each document chunk is converted into a semantic vector using a Sentence Transformer model.

### 4. Vector Indexing

Embeddings, document chunks, and metadata are stored in ChromaDB.

### 5. Semantic Retrieval

When a user asks a question, the query is converted into an embedding and compared against indexed document embeddings.

The most semantically similar chunks are retrieved.

### 6. Reranking

Initial retrieval results are passed through a Cross-Encoder reranking model.

The reranker evaluates query-document relevance and prioritizes the strongest contextual evidence.

### 7. Context Construction

Top reranked chunks are assembled into a structured context containing source metadata.

### 8. Grounded Generation

The LLM receives the retrieved context and user question.

Prompt controls instruct the model to:

- Use only retrieved information
- Avoid unsupported claims
- Identify unavailable information
- Include source references

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-rag-pipeline.git
```

Move into the project:

```bash
cd enterprise-rag-pipeline
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Copy:

```text
.env.example
```

to:

```text
.env
```

Add your API key:

```text
OPENAI_API_KEY=your_api_key_here
```

Do not commit `.env` to GitHub.

---

## Run the API

Start FastAPI with:

```bash
uvicorn api.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Health Check

```http
GET /
```

Example response:

```json
{
  "application": "Enterprise RAG Pipeline",
  "status": "running"
}
```

---

### Ingest Document

```http
POST /ingest
```

Upload a PDF or TXT document.

Example response:

```json
{
  "filename": "enterprise_policy.pdf",
  "status": "indexed",
  "source_documents": 12,
  "chunks_created": 35,
  "chunks_indexed": 35
}
```

---

### Ask Question

```http
POST /ask
```

Request:

```json
{
  "question": "How does the platform reduce hallucinations?"
}
```

Example response:

```json
{
  "query": "How does the platform reduce hallucinations?",
  "answer": "The platform reduces hallucinations by grounding responses in retrieved enterprise documents and instructing the language model not to generate unsupported information [Source 1].",
  "sources": [
    {
      "source": "sample_document.txt",
      "page": null,
      "chunk_index": 4,
      "rerank_score": 7.4281
    }
  ]
}
```

---

## Retrieval Strategy

The application uses a two-stage retrieval strategy.

### Stage 1 — Vector Retrieval

Sentence embeddings identify semantically related document chunks.

```text
Query
  ↓
Query Embedding
  ↓
Vector Similarity
  ↓
Top-K Documents
```

### Stage 2 — Cross-Encoder Reranking

The query and each retrieved document are evaluated together by a Cross-Encoder.

```text
Top-K Retrieved Chunks
        ↓
Query + Document Pairs
        ↓
Cross-Encoder
        ↓
Relevance Scores
        ↓
Highest-Ranked Context
```

This approach can improve context quality compared with relying only on vector similarity.

---

## Hallucination Reduction

The project uses several grounding strategies:

- Context-based generation
- Retrieval before generation
- Cross-Encoder reranking
- Low-temperature generation
- Explicit grounding instructions
- Source attribution
- Refusal when supporting information is unavailable

---

## Future Enhancements

Potential improvements include:

- Hybrid BM25 + vector retrieval
- Reciprocal Rank Fusion
- Semantic chunking
- Query rewriting
- Multi-query retrieval
- Parent-child retrieval
- Metadata filtering
- User authentication
- Role-based document access
- Conversation memory
- Redis caching
- RAG evaluation
- Retrieval metrics
- LLM observability
- Docker deployment
- Kubernetes deployment
- AWS/Azure deployment
- CI/CD pipelines

---

## Use Cases

The architecture can support applications such as:

- Enterprise knowledge assistants
- Internal policy search
- Financial document intelligence
- Technical documentation assistants
- Customer-support knowledge systems
- Research-document search
- Compliance-document analysis
- Employee knowledge portals

---

## Key Concepts Demonstrated

This repository demonstrates practical knowledge of:

- Retrieval-Augmented Generation
- Large Language Models
- Embeddings
- Vector databases
- Semantic search
- Reranking
- Context construction
- Prompt engineering
- Grounded generation
- Document processing
- REST APIs
- Enterprise AI architecture

---

## Author

**Naga Harshitha Pothini**

AI / ML / Generative AI Engineer

Focus areas:

`Generative AI` • `Agentic AI` • `RAG` • `LLMs` • `Machine Learning` • `Python` • `Data Engineering`
