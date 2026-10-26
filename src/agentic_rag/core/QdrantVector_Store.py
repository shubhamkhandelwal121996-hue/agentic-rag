import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from qdrant_client import QdrantClient,models
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_classic.storage import LocalFileStore
from langchain_classic.embeddings import CacheBackedEmbeddings


# ── Environment ─────────────────────────────────────────────────────────────

env_path = Path(__file__).with_name(".env")
load_dotenv(env_path)


# ── Config ───────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 1536  # text-embedding-3-small


# ── Embeddings and Local Cache ───────────────────────────────────────────────


# ── Embedding Cache ──────────────────────────────────────────────────────────

base_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# Get the folder containing vector_store.py
core_folder = Path(__file__).resolve().parent

# Create embedding_cache inside core
cache_path = core_folder / "embedding_cache"
cache_path.mkdir(parents=True, exist_ok=True)

# Initialize local file storage
embedding_file_store = LocalFileStore(str(cache_path))

embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    embedding_file_store,
    namespace=base_embeddings.model,
    query_embedding_cache=True,
    key_encoder="blake2b",
)

# ── Qdrant Client ────────────────────────────────────────────────────────────

qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=120,
)


# ── Collection ───────────────────────────────────────────────────────────────

def get_qdrant_collection_name(session_id: str) -> str:
    return f"PaperMind_{session_id.replace('-', '_')}"


def get_qdrant_vectorstore(session_id: str) -> QdrantVectorStore:
    collection_name = get_collection_name(session_id)

    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
        )

    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> None:
    get_vectorstore(session_id).add_documents(docs)


def list_papers(session_id: str) -> list[str]:
    collection_name = get_collection_name(session_id)

    if not qdrant_client.collection_exists(collection_name):
        return []

    seen: set[str] = set()
    titles: list[str] = []
    offset = None

    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )

        for point in points:
            title = (point.payload or {}).get(
                "metadata", {}
            ).get("title")

            if title and title not in seen:
                seen.add(title)
                titles.append(title)

        if offset is None:
            break

    return titles


def retrieve_documents(
    query: str,
    session_id: str,
    k: int = 4,
) -> list[Document]:

    vectorstore = get_vectorstore(session_id)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever.invoke(query)