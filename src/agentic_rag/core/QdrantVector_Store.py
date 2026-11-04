import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_classic.storage import LocalFileStore
from langchain_classic.embeddings import CacheBackedEmbeddings

# ── Environment ─────────────────────────────────────────────────────────────

env_path = Path(__file__).with_name(".env")
load_dotenv(env_path)

# ── Config ──────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 1536  # text-embedding-3-small

# ── Embeddings and Local Cache ──────────────────────────────────────────────

base_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Get the folder containing this Python file
core_folder = Path(__file__).resolve().parent

# Create the embedding cache directory
cache_path = core_folder / "embedding_cache"
cache_path.mkdir(parents=True, exist_ok=True)

# Initialize local file storage
embedding_file_store = LocalFileStore(str(cache_path))

# Cache document and query embeddings
embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    embedding_file_store,
    namespace=base_embeddings.model,
    query_embedding_cache=True,
    key_encoder="blake2b",
)

# ── Qdrant Client ───────────────────────────────────────────────────────────

qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=120,
)

# ── Collection ──────────────────────────────────────────────────────────────

def get_qdrant_collection_name(session_id: str) -> str:
    return f"PaperMind_{session_id.replace('-', '_')}"


def get_qdrant_vectorstore(session_id: str) -> QdrantVectorStore:

    collection_name = get_qdrant_collection_name(session_id)

    # Create collection only if it does not exist
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
        )

    # Required for exact-match metadata filtering in Qdrant Cloud.
    # Also handles collections created before metadata filtering was added.

    collection_info = qdrant_client.get_collection(collection_name)

    if "metadata.title" not in (collection_info.payload_schema or {}):
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.title",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )

# ── Public API ──────────────────────────────────────────────────────────────

def add_research_paper(
    docs: list[Document],
    session_id: str,
) -> None:

    # Store chunks, embeddings and their existing metadata
    get_qdrant_vectorstore(session_id).add_documents(docs)


def list_research_papers(session_id: str) -> list[str]:

    collection_name = get_qdrant_collection_name(session_id)

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


def delete_session(session_id: str) -> None:
    """Delete the Qdrant collection for a session."""

    collection_name = get_qdrant_collection_name(session_id)

    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)

# ── Retrieval with Optional Metadata Filtering ─────────────────────────────

def retrieve_documents(
    query: str,
    session_id: str,
    k: int = 4,
    title: str | None = None,
) -> list[Document]:

    vectorstore = get_qdrant_vectorstore(session_id)

    # Default: semantic similarity search across all papers
    search_kwargs = {"k": k}

    # Apply metadata filtering when a paper title is provided
    if title is not None:

        # Validate title against the papers stored in this session
        available_titles = list_research_papers(session_id)

        if title not in available_titles:
            raise ValueError(
                f"Research paper not found: {title}"
            )

        # Restrict similarity search to chunks from this paper
        search_kwargs["filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.title",
                    match=models.MatchValue(value=title),
                )
            ]
        )

    # Create retriever with optional metadata filter
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    # Retrieve top-k matching chunks
    return retriever.invoke(query)
