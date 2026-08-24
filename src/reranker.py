from typing import Dict, List

from sentence_transformers import (
    CrossEncoder
)

from src.config import (
    RERANKER_MODEL,
    RERANK_TOP_K
)


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            RERANKER_MODEL
        )

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = RERANK_TOP_K
    ) -> List[Dict]:

        if not documents:
            return []

        pairs = [
            [
                query,
                document["text"]
            ]
            for document
            in documents
        ]

        scores = self.model.predict(
            pairs
        )

        for document, score in zip(
            documents,
            scores
        ):

            document[
                "rerank_score"
            ] = float(score)

        ranked_documents = sorted(
            documents,
            key=lambda document:
                document[
                    "rerank_score"
                ],
            reverse=True
        )

        return ranked_documents[
            :top_k
        ]
