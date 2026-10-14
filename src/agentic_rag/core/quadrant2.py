import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient,models
from langchain_core.documents import Document
load_dotenv()

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
llm = ChatOpenAI(model = "gpt-5")

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

collection_name = "rag_base"
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config= models.VectorParams(
            size = 1536,
            distance=models.Distance.COSINE
    )
)

vector_store = QdrantVectorStore(
    client = client,
    collection_name=collection_name,
    embedding= embeddings
)

docs = [
    Document(
        page_content="Rag is retrieval augmented generation that enhances the llms by connecting them to external sources like databses,documents or api",
        metadata = {"Topic":"GenAI"}
    ),
    Document(
        page_content="Shubham khandelwal studied 2 Mtech from IIT madras and IIT kharagpur. " \
        "He scored 7.36 in IIT madras Mtech Industrial Mathematics and Scientific Computing and 7.39 in IIT Kharagpur Mtech Computer Science and Data Processing",
        metadata = {"Topic":"Shubham Khandelwal qualification Data"}
    )
]

vector_store.add_documents(docs)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)
query = "What is information of shubham khandelwal"
result = retriever.invoke(query)
context = result[0].page_content
prompt = (
    f"Answer from the given context first"
    f"context:{context}"
    f"Query:{query}"
)
response = llm.invoke(prompt)
print(response.content)