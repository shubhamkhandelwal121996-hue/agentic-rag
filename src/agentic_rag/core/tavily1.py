import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient(
    api_key=os.environ["TAVILY_API_KEY"]
)
query = "Is tansformer supersead RNN"
response = tavily_client.search(
    f"site:arxiv.org {query}",
max_results=3)

for result in response["results"]:
    print("Title:", result["title"])
    print("URL:", result["url"])
    print("Content:", result["content"])
    print()