from typing import Dict, List

import chromadb

from sentence_transformers import (
    SentenceTransformer
)

from src.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL
)


class VectorStore:

    def __init__(self):

        self.embedding_model = (
            SentenceTransformer(
                EMBEDDING_MODEL
            )
        )

        self.client = (
            chromadb.PersistentClient(
                path=CHROMA_PATH
            )
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=COLLECTION_NAME
            )
        )

    def create_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:

        embeddings = (
            self.embedding_model.encode(
                texts,
                normalize_embeddings=True
            )
        )

        return embeddings.tolist()

    def add_documents(
        self,
        documents: List[Dict]
    ) -> int:

        if not documents:
            return 0

        texts = [
            document["text"]
            for document in documents
        ]

        ids = [
            document["id"]
            for document in documents
        ]

        metadatas = [
            document["metadata"]
            for document in documents
        ]

        embeddings = (
            self.create_embeddings(
                texts
            )
        )

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

        return len(documents)

    def search(
        self,
        query: str,
        top_k: int
    ) -> List[Dict]:

        query_embedding = (
            self.create_embeddings(
                [query]
            )[0]
        )

        results = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k
        )

        retrieved_documents = []

        if not results["documents"]:
            return retrieved_documents

        documents = results[
            "documents"
        ][0]

        metadatas = results[
            "metadatas"
        ][0]

        distances = results[
            "distances"
        ][0]

        ids = results[
            "ids"
        ][0]

        for (
            document_id,
            text,
            metadata,
            distance
        ) in zip(
            ids,
            documents,
            metadatas,
            distances
        ):

            retrieved_documents.append(
                {
                    "id": document_id,
                    "text": text,
                    "metadata": metadata,
                    "distance": distance
                }
            )

        return retrieved_documents
