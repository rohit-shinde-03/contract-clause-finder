from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pathlib import Path
import httpx
from chromadb import HttpClient
from typing import List

app = FastAPI(title="CCF Ingest & Search Service")

# —— Resolve project-root paths —— #
BASE_DIR = Path(__file__).resolve().parent.parent  # points to contract-clause-finder/
TEXT_DIR = BASE_DIR / "data" / "text"

# —— Tika config —— #
TIKA_URL = "http://localhost:9998/tika"

# —— Chroma client & collection —— #
chroma_client = HttpClient(host="localhost", port=8000)
collection = chroma_client.get_collection("contracts")


@app.post("/ingest")
async def ingest_contract(pdf: UploadFile = File(...)):
    # Validate content-type
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    # Read PDF bytes
    pdf_bytes = await pdf.read()

    # Call Tika for text extraction
    async with httpx.AsyncClient() as client:
        headers = {"Accept": "text/plain"}
        try:
            resp = await client.put(TIKA_URL, content=pdf_bytes, headers=headers, timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Tika error: {e}")

    text = resp.text

    # Ensure the output directory exists
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    # Write full text to disk
    txt_path = TEXT_DIR / f"{pdf.filename}.txt"
    txt_path.write_text(text, encoding="utf-8")

    # Return a short snippet and save path
    snippet = text[:500] + ("…" if len(text) > 500 else "")
    return {
        "filename": pdf.filename,
        "text_snippet": snippet,
        "saved_to": str(txt_path.relative_to(BASE_DIR))
    }


@app.get("/search")
async def search_clauses(
    q: str = Query(..., description="Search query"),
    n_results: int = Query(5, description="Number of results to return")
):
    """
    Search for relevant contract clauses matching the query.
    """
    # Query Chroma for top-n nearest chunks
    results = collection.query(
        query_texts=[q],
        n_results=n_results
    )

    hits: List[dict] = []
    for idx, doc in enumerate(results['documents'][0]):
        hits.append({
            "chunk_id": results['ids'][0][idx],
            "text": doc,
            "metadata": results['metadatas'][0][idx]
        })

    return {
        "query": q,
        "results": hits
    }
