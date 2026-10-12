"""Thin Qdrant wrapper storing MRL-truncated vectors.

Truncating to 256d instead of the native 768d cuts storage ~3x with a small
recall cost — see PROPOSAL.md for the tradeoff.
"""

from typing import Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class VectorStore:
    def __init__(self, url: str, collection: str, dim: int = 256):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.dim = dim
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids: Sequence[int], vectors: np.ndarray, payloads: Sequence[dict] | None = None):
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim}-dim vectors, got {vectors.shape[1]}")
        points = [
            PointStruct(id=ids[i], vector=vectors[i].tolist(), payload=(payloads[i] if payloads else {}))
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        return self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=top_k,
        ).points
