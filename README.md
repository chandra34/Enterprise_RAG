# Enterprise RAG Milvus Assistant

A modular, production-oriented Retrieval-Augmented Generation (RAG) system built with Python, FastAPI, Streamlit, Milvus (Standalone & Lite), Google GenAI, Groq, and PyMuPDF.

---

## 🏗️ Architecture

The system is designed with a decoupled architecture where ingestion, retrieval, vector storage, and generation are separate components. This allows parts of the system to be scaled or replaced independently.

### System Architecture Diagram

```mermaid
graph TD
    subgraph Frontend [Frontend (Streamlit)]
        UI[Streamlit App]
        Client[Backend API Client]
    end

    subgraph Backend [Backend (FastAPI)]
        Router[API Routes /api/v1]
        Ingest[Ingest Service]
        Pipeline[RAG Pipeline]
        Parser[PDF Parser (PyMuPDF)]
        Chunker[Text Chunker]
        EmbedService[Embedding Service]
        RetrieveService[Retrieval Service]
        LLM[LLM Service (Groq)]
    end

    subgraph Databases & APIs [Databases & External APIs]
        Milvus[(Milvus Vector DB)]
        Gemini[Google Gemini API]
        GroqAPI[Groq LLM API]
    end

    %% Upload Flow
    UI -->|Upload PDF| Router
    Router -->|Bytes & Filename| Ingest
    Ingest -->|Parse PDF| Parser
    Parser -->|Page Text| Chunker
    Chunker -->|Text Chunks| EmbedService
    EmbedService -->|Generate Embeddings| Gemini
    Gemini -->|Vectors (dim=1536)| EmbedService
    EmbedService -->|Return Normalized Vectors| Ingest
    Ingest -->|Insert Chunks & Vectors| Milvus

    %% Query Flow
    UI -->|Ask Question| Router
    Router -->|Query text| Pipeline
    Pipeline -->|Query text| RetrieveService
    RetrieveService -->|Embed Query| EmbedService
    RetrieveService -->|Vector Search| Milvus
    Milvus -->|Retrieve Source Chunks| RetrieveService
    RetrieveService -->|Return Chunks| Pipeline
    Pipeline -->|Assemble Prompt (Context + Question)| LLM
    LLM -->|Chat Completion Request| GroqAPI
    GroqAPI -->|Response Answer| LLM
    LLM -->|Return Answer| Pipeline
    Pipeline -->|Answer + Sources| Router
    Router -->|JSON Response| UI
```

### Module Descriptions
- `backend/pdf/parser.py`: Extracts clean text from PDF documents page-by-page using PyMuPDF.
- `backend/rag/chunking.py`: Splits the extracted text into overlapping chunks using token/character splitters to maintain semantic coherence.
- `backend/rag/embeddings.py`: Integrates with Google GenAI SDK (`gemini-embedding-2`) to generate normalized vector representations of chunks.
- `backend/vectordb/milvus_db.py`: Creates, configures, loads, and queries collections in the Milvus database.
- `backend/services/ingest_service.py`: Coordinates the end-to-end PDF parsing, chunking, embedding, and vector database insertion flow.
- `backend/rag/retrieval.py`: Retrieves similar context chunks from the vector database using cosine/IP similarity.
- `backend/rag/prompts.py`: Structures system and user templates for the generation context.
- `backend/rag/llm.py`: Connects to the Groq API (`llama-3.1-8b-instant`) for fast response generation.
- `backend/rag/pipeline.py`: Orchestrates retrieval, prompt construction, and final answer generation.
- `backend/api/routes.py`: Exposes REST endpoints for application health, document upload, and querying.
- `frontend/streamlit_app.py`: Provides an interactive web dashboard for uploading PDFs and chatting with the assistant.

---

## 🛠️ Technical Decisions

1. **FastAPI & Streamlit Separation**: Separating the frontend and backend ensures we can scale the FastAPI service independently (e.g., placing it behind a load balancer) and swap out the Streamlit UI with a custom React/Next.js interface in the future.
2. **Milvus Lite for Local Dev & Milvus Standalone for Production**: 
   - Local runs use **Milvus Lite** (running on a local file, e.g., `milvus_local.db`), which eliminates the need to run Docker containers during local testing and speeds up development.
   - The settings database URI can be easily changed in production to point to a full remote Milvus cluster (`milvus_host:port`) without editing any database logic.
3. **Google GenAI Embeddings (`gemini-embedding-2`)**: Standardized on Gemini API for high-quality, 1536-dimensional text embeddings, enabling accurate similarity matchings.
4. **Groq API (`llama-3.1-8b-instant`)**: Selected Groq for text generation because of its extremely high token throughput and low latency, making the retrieval-augmented chat loop feel instantaneous.
5. **PyMuPDF**: Lightweight and exceptionally fast parsing library compared to heavier alternatives like PyPDF or OCR-based tools, suitable for clean, text-based PDF documents.

---

## 🚀 How to Run the Application

### 1. Prerequisites
- Python 3.10 or 3.11 installed.
- Access to Google GenAI API Key (for embeddings).
- Access to Groq API Key (for LLM generation).

### 2. Environment Setup

Clone this repository and navigate to the project directory:
```bash
cd Enterprice_RAG
```

Create a virtual environment and activate it:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the root directory and configure the environment variables:
```env
# Application Settings
APP_NAME="RAG Milvus Assistant"
APP_ENV="development"
LOG_LEVEL="INFO"

# API Keys
GEMINI_API_KEY="your-google-gemini-api-key"
GROQ_API_KEY="your-groq-api-key"

# Embedding Config
EMBEDDING_MODEL_NAME="gemini-embedding-2"
MILVUS_DIMENSION=1536

# LLM Config
LLM_MODEL="llama-3.1-8b-instant"

# Milvus Config
# Use local file URI for Milvus Lite, or a host/port for remote Milvus
MILVUS_URI="./milvus_local.db"
MILVUS_COLLECTION_NAME="rag_documents"
```

*Note: If you have previously indexed documents with a different embedding dimension, delete `milvus_local.db` or change `MILVUS_COLLECTION_NAME` before starting fresh.*

### 4. Running the Backend (FastAPI)

Start the Uvicorn web server to run the backend API:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Health Check**: Verify the backend is up by visiting `http://localhost:8000/api/v1/health` in your browser.
- **Interactive Documentation**: Access the FastAPI Swagger docs at `http://localhost:8000/docs`.

### 5. Running the Frontend (Streamlit)

From the project root directory, run the Streamlit application in a separate terminal:
```bash
streamlit run frontend/streamlit_app.py --server.port 8501
```

- **Web UI**: Access the interface in your browser at `http://localhost:8501`.

---

## ⚠️ Known Limitations

1. **Milvus Lite Index Support**:
   - Milvus Lite only supports the `FLAT` index type (brute-force search) when using local `.db` files.
   - High-performance indices like `HNSW` are only active when connecting to a remote, fully featured Milvus server.
2. **Synchronous Threadpool Block during Ingestion**:
   - PDF ingestion is processed in FastAPI using `run_in_threadpool` off the main async loop. For massive PDFs or high concurrent usage, this can cause thread pool exhaustion. Production deployments should use an asynchronous message queue (e.g., Celery with Redis) to handle background document processing.
3. **Single-Tenant Architecture**:
   - The database collection is shared globally. All users upload files to and query from the same `rag_documents` collection. There is no user authorization or document tenancy partitioning.
4. **No Session Memory (Conversational History)**:
   - The RAG pipeline processes queries independently as single-turn interactions. It does not append previous questions/answers as context for subsequent queries.
5. **Security Configurations**:
   - CORS is configured with wildcards (`allow_origins=["*"]`) for easy local development, which must be restricted to explicit domains in a production environment.
   - There is no built-in API token validation or rate limiting.
6. **Local Upload Storage**:
   - Uploaded PDF bytes are temporarily stored on the server's local filesystem (`backend/uploads`) instead of a persistent object storage solution like Amazon S3 or Google Cloud Storage.
