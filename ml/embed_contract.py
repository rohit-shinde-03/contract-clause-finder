#!/usr/bin/env python3
import sys
import uuid
from pathlib import Path
from typing import List

from chromadb import HttpClient
from sentence_transformers import SentenceTransformer, models
import tiktoken
import torch

# —— Configuration —— #
BASE_DIR           = Path(__file__).resolve().parent.parent
TEXT_DIR           = BASE_DIR / "data" / "text"
CHROMA_COLLECTION  = "contracts_bert"
EMBED_MODEL_NAME   = "nlpaueb/legal-bert-small-uncased"
CHUNK_SIZE_TOKENS  = 500
CHUNK_OVERLAP_TOKENS = 50

def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def chunk_text(text: str) -> List[str]:
    enc    = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE_TOKENS, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        start += CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS
    return chunks

def load_fp16_embedder(model_name: str) -> SentenceTransformer:
    # FP16 transformer backbone
    transformer = models.Transformer(
        model_name_or_path=model_name,
        model_args={"torch_dtype": torch.float16}
    )
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    return SentenceTransformer(modules=[transformer, pooling])

def embed_and_store(file_id: str, chunks: List[str]):
    model = load_fp16_embedder(EMBED_MODEL_NAME)
    embs  = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    client     = HttpClient(host="localhost", port=8000)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"description": "FP16 Legal-BERT embeddings"}
    )

    ids       = [f"{file_id}-{i}" for i in range(len(chunks))]
    metadatas = [{"file_id": file_id, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(
        embeddings=embs.tolist(),
        ids=ids,
        metadatas=metadatas,
        documents=chunks
    )
    print(f"[+] Stored {len(chunks)} chunks for file {file_id}")

def main():
    if len(sys.argv) != 2:
        print("Usage: ml/embed_contract.py path/to/text.txt")
        sys.exit(1)
    txt_path = sys.argv[1]
    file_id  = uuid.uuid4().hex
    text     = load_text(txt_path)
    chunks   = chunk_text(text)
    embed_and_store(file_id, chunks)

if __name__ == "__main__":
    main()
