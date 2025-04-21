import os
import pathlib
import json
from typing import List, Dict

import fitz  # PyMuPDF
import chromadb
from chromadb import PersistentClient, Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types
from google.api_core import retry
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing_extensions import TypedDict 
# -----------------------------------------------------------------------------
# Runtime configuration --------------------------------------------------------
# -----------------------------------------------------------------------------
PDF_DIR = os.getenv("PDF_DIR", "data")            # Folder with regulation PDFs
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")  # Persistent vector store path
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")        # Gemini key (set in HF secrets)
if GOOGLE_API_KEY is None:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")

 
class F1RegulationResponse(TypedDict):
    query: str 
    response: str 
    relevant_sections: List[str] 
    clauses: List[str]

# -----------------------------------------------------------------------------
# Gemini client ----------------------------------------------------------------
# -----------------------------------------------------------------------------
client = genai.Client(api_key=GOOGLE_API_KEY)

# -----------------------------------------------------------------------------
# Embedding function -----------------------------------------------------------
# -----------------------------------------------------------------------------

is_retriable = lambda e: (
    isinstance(e, genai.errors.APIError) and e.code in {429, 503}
)


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Wrap Gemini text‑embedding‑004 for ChromaDB."""

    document_mode: bool = True  # True → document embeddings, False → query

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        task = "retrieval_document" if self.document_mode else "retrieval_query"
        resp = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(task_type=task),
        )
        return [e.values for e in resp.embeddings]


embed_fn = GeminiEmbeddingFunction()


def embed_query(text: str):
    """Return query embedding (switches embedding function to query mode)."""
    embed_fn.document_mode = False
    return embed_fn([text])

# -----------------------------------------------------------------------------
# PDF ingestion & ChromaDB build ----------------------------------------------
# -----------------------------------------------------------------------------

pathlib.Path(CHROMA_DIR).mkdir(exist_ok=True)
chroma_client = PersistentClient(path=CHROMA_DIR)
DB_NAME = "f1_regulations_db"

# Collection with embedding function attached
collection = chroma_client.get_or_create_collection(DB_NAME, embedding_function=embed_fn)

if collection.count() == 0:
    print("Building vector index from PDFs – first launch …")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    document_chunks: Dict[str, List[str]] = {}
    for pdf_name in os.listdir(PDF_DIR):
        if not pdf_name.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text("text") for page in doc)
        document_chunks[pdf_name] = splitter.split_text(text)

    # Flatten & batch‑insert
    docs = [c for chunks in document_chunks.values() for c in chunks]
    ids = [f"{name}_{i}" for name, chunks in document_chunks.items() for i, _ in enumerate(chunks)]

    MAX_BATCH = 100
    for i in range(0, len(docs), MAX_BATCH):
        collection.add(documents=docs[i:i + MAX_BATCH], ids=ids[i:i + MAX_BATCH])
        print(f"Inserted chunks {i} – {i + MAX_BATCH}")

    print("✅ Vector index ready.")
else:
    print(f"Vector index already present (chunks: {collection.count()}).")

# -----------------------------------------------------------------------------
# Core retrieval & generation --------------------------------------------------
# -----------------------------------------------------------------------------


def generate_response(query: str, context: str) -> F1RegulationResponse:
    prompt = f"""
You are an expert assistant in Formula 1 technical regulations. 
Your job is to read the provided Context and answer the User Question in a precise, structured JSON format according to the schema below. 
Only use information found in Context; do NOT invent or hallucinate any facts.

JSON schema (TypedDict):
{{
  \"query\": <the original question string>,
  \"response\": <concise, accurate answer string>,
  \"relevant_sections\": [<list of article numbers or page references>],
  \"clauses\": [<list of sub‑clauses or clause IDs>]
}}

Respond **only** with a single JSON object that matches this schema—no extra keys, no surrounding text.

Context:
{context}

User Question:
{query}

Begin.
""".strip()

    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=F1RegulationResponse,  # type: ignore[arg-type]
        ),
    )
    return json.loads(resp.text)


def retrieve_regulation(query: str) -> F1RegulationResponse:
    """Retrieve relevant chunks from ChromaDB and ask Gemini for a JSON answer."""
    query_emb = embed_query(query)
    results = collection.query(query_embeddings=query_emb, n_results=3)
    context = "\n".join(doc for docs in results["documents"] for doc in docs)
    return generate_response(query, context)

# -----------------------------------------------------------------------------
# Optional smoke test ----------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    demo = "What are the tire specifications for 2025?"
    print(json.dumps(retrieve_regulation(demo), indent=2))

# Public API export -----------------------------------------------------------
__all__ = ["retrieve_regulation"]
