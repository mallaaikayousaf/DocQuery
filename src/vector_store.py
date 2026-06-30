import uuid
from typing import List, Optional, Sequence

import chromadb


def create_chroma_client(persist_directory: str = "chroma_db"):

    client = chromadb.PersistentClient(path=persist_directory)
    return client


def create_collection(
    client,
    collection_name: str = "pdf_chunks"
):

    collection = client.get_or_create_collection(collection_name)
    print(f"Collection '{collection_name}' ready (get_or_create)")
    return collection


def store_embeddings(
    collection,
    documents: List[str],
    embeddings: Sequence[Sequence[float]],
    metadatas: Optional[Sequence[dict]] = None,
    ids: Optional[List[str]] = None
):

    if ids is None:
        ids = [f"chunk_{uuid.uuid4()}" for _ in range(len(documents))]

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    print(f"Stored {len(documents)} chunks in collection")


def retrieve_chunks(
    collection,
    query_embedding: List[float],
    top_k: int = 3
) -> List[str]:

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results['documents'][0] if results['documents'] else []

    return documents


def get_collection_stats(collection) -> dict:

    count = collection.count()

    stats = {
        'count': count,
        'name': collection.name,
        'metadata': collection.metadata
    }

    return stats