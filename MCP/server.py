from fastmcp import FastMCP

import arxiv
from tavily import TavilyClient

from pydantic import BaseModel

mcp = FastMCP(
    "Academic & Rsearch Tool Server"
)

class AcademicPapers(BaseModel):
    title: str
    authors: list
    published: str
    arxiv_id: str
    primary_category: str
    abstract: str
@mcp.tool()
def search_arxiv(query: str, max_results: int = 5):
    """
    This tool searches the arxiv database for academic literature, preprints and papers.

    Spanning computer science, physics, mathematics, quantitatives.

    This too is to be used when the user needs highly formal proofs,
    deep methodologies, or academic context.

    This tool takes multiple parameters:

    1. query: The topic the user is interested in searching / learning for.
    2. max_results: An INTEGER value, determining how many papers to find and return
    """
    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []

    for result in client.results(search):
        papers.append(AcademicPapers(
            title=result.title,
            authors=[author.name for author in result.authors],
            published=result.published.strftime("%Y-%m-%d"),
            arxiv_id=result.get_short_id(),
            primary_category=result.primary_category,
            abstract=result.summary.replace("\n", " ")
        ))

    return papers

if __name__ == "__main__":
    mcp.run()