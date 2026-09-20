import json

from langchain_core.tools import tool


@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic."""
    slug = query.lower().replace(" ", "-")
    return json.dumps({
        "query": query,
        "results": [
            {
                "title": f"Getting Started with {query}",
                "url": f"https://example.com/{slug}",
                "snippet": "A comprehensive guide covering the fundamentals "
                "and best practices for getting started.",
            },
            {
                "title": f"{query} — Official Documentation",
                "url": f"https://docs.example.com/{slug}",
                "snippet": "Official reference documentation with detailed "
                "API specifications and usage examples.",
            },
            {
                "title": f"{query}: Best Practices & Tips",
                "url": f"https://blog.example.com/{slug}",
                "snippet": "Expert tips and industry best practices compiled "
                "from real-world production experience.",
            },
        ],
    })
