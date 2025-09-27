from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


@tool
def search_tool(query: str) -> str:
    """Busca informacion en la web usando TavilySearch."""
    search = TavilySearch(max_results=5)
    results = search.run({"query": query})
    return results
