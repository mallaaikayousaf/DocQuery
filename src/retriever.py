from typing import List, Tuple

from src.embeddings import generate_embeddings
from src.vector_store import retrieve_chunks


class Retriever:

    def __init__(self, model, collection):

        self.model = model
        self.collection = collection

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:

        query_embeddings = generate_embeddings([query], self.model)
        query_embedding = query_embeddings[0]

        chunks = retrieve_chunks(
            self.collection,
            query_embedding.tolist(),
            top_k=top_k
        )

        return chunks

    def retrieve_with_scores(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:

        query_embeddings = generate_embeddings([query], self.model)
        query_embedding = query_embeddings[0]

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['documents', 'distances']
        )

        chunks = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []

        scores = [1 - d for d in distances]  # Simple conversion

        return list(zip(chunks, scores))