import time
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):

    try:
        print(f"Loading embedding model: {model_name}...")
        start_time = time.time()

        model = SentenceTransformer(model_name)

        embedding_dim = model.get_sentence_embedding_dimension()
        load_time = time.time() - start_time

        print(f"Model loaded in {load_time:.2f} seconds")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Max sequence length: {model.max_seq_length}")

        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have sentence-transformers installed: "
              "pip install sentence-transformers")
        raise


def generate_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:

    if not texts:
        print("Warning: Empty text list provided")
        return np.array([])

    if not model:
        raise ValueError("Model must be provided")

    print(f"Generating embeddings for {len(texts)} texts...")
    start_time = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True  # Set to True for cosine similarity
    )

    elapsed_time = time.time() - start_time

    print(f"Generated {len(embeddings)} embeddings")
    print(f"Time: {elapsed_time:.2f} seconds")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Average time per text: {elapsed_time/len(texts):.4f} seconds")

    return embeddings


def get_embedding_dimension(model: SentenceTransformer) -> int:
    if not model:
        raise ValueError("Model must be provided")

    dim = model.get_sentence_embedding_dimension()
    if dim is None:
        raise ValueError("Could not determine embedding dimension for this model")
    return dim
