import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

load_dotenv()

# 1. Initialize embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# 2. Connect to Qdrant Cloud
client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

# 3. Create a vector collection
collection_name = "knowledge_base"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE,
        ),
    )

# 4. Initialize vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

# 5. Add documents
documents = [
    Document(
        page_content="RAG combines retrieval with generation.",
        metadata={"topic": "RAG"},
    ),
    Document(
        page_content="Qdrant stores and searches vector embeddings.",
        metadata={"topic": "Vector databases"},
    ),
]

vector_store.add_documents(documents)

# 6. Search for similar documents
query = "What is retrieval augmented generation?"

results = vector_store.similarity_search(
    query=query,
    k=2,
)

for doc in results:
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("---")