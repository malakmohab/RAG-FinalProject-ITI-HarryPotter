
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


# ============================= Setup =============================
load_dotenv()

app = FastAPI(title="Harry Potter RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large",
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOP_K = int(os.getenv("TOP_K"))

model = SentenceTransformer(EMBEDDING_MODEL) # HERE WE NEED TO LOAD THE SAME EMBEDDING MODEL THAT WE USED TO CREATE THE VECTOR DATABASE, WITH THE SAME DIMENSIONALITY.

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

print("Qdrant URL loaded:", bool(QDRANT_URL))
print("Qdrant API key loaded:", bool(QDRANT_API_KEY))
print("Collection:", QDRANT_COLLECTION)
print("Embedding model:", EMBEDDING_MODEL)
print("TOP_K:", TOP_K)

# IF YOU WILL USE ANOTHER LLM FROM ANOTHER PROVIDER, USE THE CORRECT CLASS FROM LANGCHAIN AND PROVIDE THE REQUIRED PARAMETERS.
gemini_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0,
)

groq_llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
)


# =========================== Schemas ===========================

class QueryRequest(BaseModel):
    query: str


class Source(BaseModel):
    book_name: str
    page_number: int
    score: float


class QueryResponse(BaseModel):
    query: str
    route: str
    answer: str
    sources: list[Source]


# =========================== Endpoints ===========================

@app.get("/")
def root():
    return {"name": "Harry Potter RAG API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):



    ROUTER_SYSTEM_PROMPT = """You are a query router for a Harry Potter books chatbot.

Classify the user's message into exactly one of these categories:

- retrieve: The user is asking for information from the Harry Potter books,
  such as characters, events, places, relationships, facts, or story details.

- chitchat: The user is greeting, thanking, saying goodbye, or making
  casual conversation that does not require information from the books.

- off-topic: The user is asking about something unrelated to the Harry Potter books.

Return ONLY one word:
retrieve
chitchat
off-topic"""



    route = groq_llm.invoke([
        SystemMessage(content=(ROUTER_SYSTEM_PROMPT)),
        HumanMessage(content=request.query),
    ]).text.strip().lower()

    if route not in {"retrieve", "chitchat", "off-topic"}:
        route = "off-topic"

    if route == "chitchat":

        CHITCHAT_SYSTEM_PROMPT = """You are a friendly Harry Potter chatbot. Respond naturally and briefly to greetings, thanks, goodbyes, and casual conversation. Be warm and helpful, and keep the conversation related to Harry Potter when appropriate. Do not provide detailed book information or invent facts."""

        response = groq_llm.invoke([
            SystemMessage(content=CHITCHAT_SYSTEM_PROMPT),
            HumanMessage(content=request.query),
        ])

        return QueryResponse(
            query=request.query,
            route=route,
            answer=response.text,
            sources=[],
        )

    if route == "off-topic":
        return QueryResponse(
            query=request.query,
            route=route,
            answer="I can only answer questions about the Harry Potter books.",
            sources=[],
        )

    query_vector = model.encode(
        [f"query: {request.query}"],
        normalize_embeddings=True,
    )[0].tolist()

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    ).points

    context = "\n\n".join(
        f"Book: {result.payload['book_name']}\n"
        f"Page: {result.payload['page_number']}\n"
        f"Content: {result.payload['content']}"
        for result in results
    )


    RAG_SYSTEM_PROMPT = """You are a Harry Potter books chatbot. Answer the user's question using only the provided context from the Harry Potter books. If the answer cannot be found or determined from the provided context, say: "I do not know based on the provided context." Do not use outside knowledge. Do not make up or assume facts. Keep the answer clear, concise, and friendly."""

    response = gemini_llm.invoke([
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion:\n{request.query}"
        ),
    ])

    return QueryResponse(
        query=request.query,
        route=route,
        answer=response.text,
        sources=[
            Source(
                book_name=result.payload["book_name"],
                page_number=result.payload["page_number"],
                score=result.score,
            )
            for result in results
        ],
    )
