from pathlib import Path
from typing import Dict

from src.chunking import (
    chunk_documents
)

from src.generation import Generator

from src.ingestion import (
    load_document
)

from src.reranker import Reranker

from src.retrieval import Retriever

from src.vector_store import VectorStore


class EnterpriseRAGPipeline:

    def __init__(self):

        self.vector_store = (
            VectorStore()
        )

        self.retriever = Retriever(
            self.vector_store
        )

        self.reranker = (
            Reranker()
        )

        self.generator = (
            Generator()
        )

    def ingest(
        self,
        file_path: str
    ) -> Dict:

        if not Path(
            file_path
        ).exists():

            raise FileNotFoundError(
                f"File not found: "
                f"{file_path}"
            )

        documents = load_document(
            file_path
        )

        chunks = chunk_documents(
            documents
        )

        indexed_count = (
            self.vector_store
            .add_documents(
                chunks
            )
        )

        return {
            "source_documents":
                len(documents),

            "chunks_created":
                len(chunks),

            "chunks_indexed":
                indexed_count
        }

    def ask(
        self,
        query: str
    ) -> Dict:

        retrieved_documents = (
            self.retriever.retrieve(
                query
            )
        )

        reranked_documents = (
            self.reranker.rerank(
                query=query,
                documents=
                    retrieved_documents
            )
        )

        answer = (
            self.generator.generate(
                query=query,
                documents=
                    reranked_documents
            )
        )

        sources = []

        for document in (
            reranked_documents
        ):

            metadata = document[
                "metadata"
            ]

            sources.append(
                {
                    "source":
                        metadata.get(
                            "source"
                        ),

                    "page":
                        metadata.get(
                            "page"
                        ),

                    "chunk_index":
                        metadata.get(
                            "chunk_index"
                        ),

                    "rerank_score":
                        round(
                            document.get(
                                "rerank_score",
                                0
                            ),
                            4
                        )
                }
            )

        return {
            "query": query,
            "answer": answer,
            "sources": sources
        }
