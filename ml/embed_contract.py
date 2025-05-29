import os
import sys
import uuid
from typing import List

import chromadb
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer, models
import tiktoken


# —— Configuration —— #
TIKA_TEXT_DIR = "../data/text/"       # where full-text .txt files live
CHROMA_COLLECTION = "contracts"
EMBED_MODEL_NAME = "nlpaueb/legal-bert-small-uncased"
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks by token count."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        start += CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS
    return chunks


def embed_and_store(file_id: str, chunks: List[str]):
    # Build a custom SentenceTransformer pipeline:
    word_embedding_model = models.Transformer(
        model_name_or_path="nlpaueb/legal-bert-small-uncased",
        max_seq_length=512
    )
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Init Chroma client (HTTP API on localhost:8000)

    client = HttpClient(
        host="localhost",
        port=8000
    )

    collection = client.get_or_create_collection(
    name="contracts_bert",
    metadata={"description": "Legal-BERT embeddings of contracts"}
    )

    # Generate embeddings in batches
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    # Prepare metadata & IDs
    ids = [f"{file_id}-{i}" for i in range(len(chunks))]
    metadata = [{"file_id": file_id, "chunk_index": i} for i in range(len(chunks))]

    # Upsert to Chroma
    collection.upsert(
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadata,
        documents=chunks
    )
    print(f"[+] Stored {len(chunks)} chunks for file {file_id}")


def main(txt_path: str):
    # Generate a unique ID for this document
    file_id = uuid.uuid4().hex

    text = load_text(txt_path)
    chunks = chunk_text(text)
    embed_and_store(file_id, chunks)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python embed_contract.py path/to/contract.txt")
        sys.exit(1)
    main(sys.argv[1])
