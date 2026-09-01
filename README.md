# 🪄 Harry Potter RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the **Harry Potter books** using the content of the books as its external knowledge source.

The project combines **semantic search, vector databases, LLMs, and FastAPI** to build an end-to-end question-answering system.

## ✨ Features

* 📚 Question answering over the Harry Potter books
* 🔎 Semantic search using `multilingual-e5-large`
* 🗄️ Qdrant vector database for similarity search
* 🤖 Gemini for grounded answer generation
* 🚦 Groq-powered query routing
* 💬 Handles casual conversation separately
* 🚫 Rejects questions unrelated to Harry Potter
* 📖 Returns the book name, page number, and similarity score of retrieved sources
* ⚡ FastAPI backend with interactive Swagger documentation
* 🌐 CORS enabled for frontend integration

---

## 🏗️ Architecture

```text
                    User Question
                          │
                          ▼
                 ┌─────────────────┐
                 │  FastAPI /query │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   Groq Router   │
                 └────────┬────────┘
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
         retrieve      chitchat     off-topic
             │            │            │
             │            ▼            ▼
             │          Groq       Rejection
             │          response    response
             │
             ▼
      Query Embedding
      multilingual-e5-large
             │
             ▼
        Qdrant Search
             │
             ▼
       Top-K Chunks
             │
             ▼
     Retrieved Context
             │
             ▼
       Gemini LLM
             │
             ▼
      Grounded Answer
             │
             ▼
       Sources + Score
```

---

## 🧠 How the RAG System Works

### 1. Data Preparation

The Harry Potter books are processed from the provided PDF.

The preprocessing pipeline extracts and organizes the book content while preserving page information.

The resulting data contains metadata such as:

* Book name
* Page number
* Text content

The preprocessing and vector-database workflow is documented in the project notebook:

`notebook/rag_hp_pipeline.ipynb`

---

### 2. Embeddings

The project uses:

```text
intfloat/multilingual-e5-large
```

to convert text into vector representations.

For a user question, the query is encoded using the same embedding model that was used when creating the vector database.

The query is formatted as:

```text
query: <user question>
```

and embeddings are normalized before similarity search.

---

### 3. Vector Search

The embeddings are stored in **Qdrant**.

When the user asks a question, the system:

1. Creates an embedding for the question.
2. Searches the Qdrant collection.
3. Retrieves the top `K` most similar chunks.
4. Builds a context from the retrieved content.

Each retrieved result contains:

```text
Book
Page
Content
Similarity Score
```

---

## 🚦 Query Routing

Before performing retrieval, the system uses **Groq** as a lightweight query router.

Every user message is classified into one of three categories:

### `retrieve`

Questions that require information from the Harry Potter books.

Examples:

```text
Who are Harry Potter's parents?
What happened to Sirius Black?
Where is Hogwarts located?
```

These questions continue through the RAG pipeline.

### `chitchat`

Casual conversation such as:

```text
Hello!
Thanks!
Goodbye!
```

These requests are answered directly by the Groq-powered conversational model without querying the vector database.

### `off-topic`

Questions unrelated to the Harry Potter books.

For example:

```text
What is the capital of France?
Explain neural networks.
```

These requests are rejected with:

```text
I can only answer questions about the Harry Potter books.
```

This routing layer prevents unnecessary retrieval and keeps the chatbot focused on its intended knowledge domain.

---

## 🤖 Answer Generation

For retrieval-based questions, the retrieved book passages are provided to **Google Gemini** as context.

The model is instructed to:

* Answer only from the provided context
* Avoid outside knowledge
* Avoid inventing facts
* Clearly state when the answer cannot be determined from the retrieved context
* Keep responses concise and friendly

This helps ground the generated answer in the source material.

---

## 📖 Source Information

The API returns the sources used for each retrieved answer.

Example response structure:

```json
{
  "query": "Who are Harry Potter's parents?",
  "route": "retrieve",
  "answer": "...",
  "sources": [
    {
      "book_name": "...",
      "page_number": 1,
      "score": 0.89
    }
  ]
}
```

This makes it possible to inspect which book pages contributed to the generated response.

---

## 🛠️ Technologies

| Technology              | Purpose                            |
| ----------------------- | ---------------------------------- |
| Python                  | Core development                   |
| FastAPI                 | REST API                           |
| Uvicorn                 | API server                         |
| Sentence Transformers   | Text embeddings                    |
| `multilingual-e5-large` | Embedding model                    |
| Qdrant                  | Vector database                    |
| Gemini                  | RAG answer generation              |
| Groq                    | Query routing & chitchat           |
| LangChain               | LLM integration                    |
| PyMuPDF                 | PDF text extraction                |
| python-dotenv           | Environment configuration          |
| Jupyter                 | Data preparation & experimentation |

---

## 📁 Project Structure

```text
RAG-HarryPotter/
│
├── Data/
│   └── Harry Potter book data
│
├── notebook/
│   ├── rag_hp_pipeline.ipynb
│   └── output.md
│
├── UI HarryPotter Screenshots/
│   └── Application screenshots
│
├── rag_api.py
├── index.html
├── Requirements.txt
├── env.example
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/malakmohab/RAG-HarryPotter.git
cd RAG-HarryPotter
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r Requirements.txt
```

### 4. Configure environment variables

Create a `.env` file based on `env.example`.

The application requires configuration for:

```env
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=

EMBEDDING_MODEL=
TOP_K=

GEMINI_MODEL=
GEMINI_API_KEY=

GROQ_MODEL=
GROQ_API_KEY=
```

> **Important:** Never commit your `.env` file or API keys to GitHub.

---

## ▶️ Run the API

Start the FastAPI application with:

```bash
uvicorn rag_api:app --reload
```

The API will run locally at:

```text
http://127.0.0.1:8000
```

### Swagger Documentation

FastAPI automatically provides interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 🔌 API Endpoints

### `GET /`

Returns basic information about the API.

### `GET /health`

Health check endpoint.

Example:

```json
{
  "status": "ok"
}
```

### `POST /query`

Main RAG endpoint.

Request:

```json
{
  "query": "Who are Harry Potter's parents?"
}
```

Response:

```json
{
  "query": "Who are Harry Potter's parents?",
  "route": "retrieve",
  "answer": "...",
  "sources": [...]
}
```

---

## 🔄 Complete RAG Pipeline

```text
PDF
 │
 ▼
Text Extraction
 │
 ▼
Page Organization
 │
 ▼
Chunking
 │
 ▼
Sentence Embeddings
 │
 ▼
Qdrant Vector Database
 │
 └──────────────┐
                │
User Question   │
      │         │
      ▼         │
  Groq Router   │
      │         │
      ▼         │
  retrieve ─────┘
      │
      ▼
Query Embedding
      │
      ▼
Qdrant Similarity Search
      │
      ▼
Top-K Relevant Chunks
      │
      ▼
Context + Question
      │
      ▼
Gemini
      │
      ▼
Answer + Sources
```

---

## 💡 Why RAG Instead of Fine-Tuning?

This project uses RAG instead of fine-tuning the LLM on the Harry Potter books.

With RAG:

* The book content remains in an external knowledge base.
* The LLM does not need to be retrained when the knowledge source changes.
* Relevant information is retrieved dynamically for each question.
* The retrieved sources can be returned alongside the answer.
* The system can better control what information is provided to the LLM.

The LLM itself is **not trained on the Harry Potter books as part of this project**. The books are stored externally and retrieved when needed.

---

## 📸 Application

The repository includes screenshots of the Harry Potter chatbot interface in:

```text
UI HarryPotter Screenshots/
```

---

## 🎯 Project Goals

This project was built to demonstrate practical implementation of an end-to-end RAG system, including:

* Document preprocessing
* Text chunking
* Embedding generation
* Vector database creation
* Semantic retrieval
* Query routing
* LLM integration
* Context-grounded generation
* Source tracking
* REST API development

