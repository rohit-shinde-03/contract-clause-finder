from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pathlib import Path
import httpx
from chromadb import HttpClient
from typing import List
import subprocess, sys, re
from sentence_transformers import SentenceTransformer, models
import torch

app = FastAPI(title="CCF Ingest & Search Service")

# —— FP16 Embedder Setup —— #
transformer = models.Transformer(
    model_name_or_path="nlpaueb/legal-bert-small-uncased",
    model_args={"torch_dtype": torch.float16}
)
pooling   = models.Pooling(
    transformer.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
embedder  = SentenceTransformer(modules=[transformer, pooling])

# —— Paths & Config —— #
BASE_DIR    = Path(__file__).resolve().parent.parent
TEXT_DIR    = BASE_DIR / "data" / "text"
ML_SCRIPT   = BASE_DIR / "ml" / "embed_contract.py"
TIKA_URL    = "http://localhost:9998/tika"
CHROMA_COLL = "contracts_bert"

# —— Shared Chroma Client & Collection —— #
_chroma_client = HttpClient(host="localhost", port=8000)
_collection    = _chroma_client.get_or_create_collection(
    name=CHROMA_COLL,
    metadata={"description": "FP16 Legal-BERT embeddings"}
)


@app.post("/ingest")
async def ingest_contract(pdf: UploadFile = File(...)):
    # 1) Validate
    if pdf.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files accepted")

    # 2) Extract text via Tika
    data = await pdf.read()
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.put(
                TIKA_URL,
                content=data,
                headers={"Accept": "text/plain"},
                timeout=30.0
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Tika error: {e}")
    text = resp.text

    # 3) Write to disk
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = TEXT_DIR / f"{pdf.filename}.txt"
    txt_path.write_text(text, encoding="utf-8")

    # 4) Spawn embedding in background
    subprocess.Popen([
        sys.executable,
        str(ML_SCRIPT),
        str(txt_path)
    ])

    return {"filename": pdf.filename, "embedding_started": True}


@app.get("/search")
async def search_clauses(
    q: str = Query(..., description="Search query"),
    n_results: int = Query(5, description="Number of results")
):
    try:
        # 1) Embed the query
        query_emb = embedder.encode([q], convert_to_numpy=True).tolist()

        # 2) Semantic top-N search; include documents & metadatas
        sem = _collection.query(
            query_embeddings=query_emb,
            n_results=n_results,
            include=["documents", "metadatas"]   # <-- no "ids" here
        )
        docs = sem["documents"][0]
        ids  = sem["ids"][0]      # ids come back by default
        metas= sem["metadatas"][0]

        # 3) Literal pre-filter among the top-N
        pattern = re.compile(rf"\b{re.escape(q)}\b", re.IGNORECASE)
        hits = [
            {"chunk_id": cid, "text": doc, "metadata": meta}
            for doc, cid, meta in zip(docs, ids, metas)
            if pattern.search(doc)
        ]

        # 4) Fallback: if no literal hits, return semantic hits
        if not hits:
            hits = [
                {"chunk_id": cid, "text": doc, "metadata": meta}
                for doc, cid, meta in zip(docs, ids, metas)
            ]

        return {"query": q, "results": hits}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

