from typing import Dict, List

from src.config import TOP_K
from src.vector_store import VectorStore


class Retriever:

    def __init__(
        self,
        vector_store: VectorStore
    ):

        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K
    ) -> List[Dict]:

        if not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        results = (
            self.vector_store.search(
                query=query,
                top_k=top_k
            )
        )

        return results
