from typing import Dict, List

from openai import OpenAI

from src.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL
)


class Generator:

    def __init__(self):

        if not OPENAI_API_KEY:

            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

    def build_context(
        self,
        documents: List[Dict]
    ) -> str:

        context_parts = []

        for index, document in enumerate(
            documents,
            start=1
        ):

            metadata = document[
                "metadata"
            ]

            source = metadata.get(
                "source",
                "unknown"
            )

            page = metadata.get(
                "page"
            )

            source_label = source

            if page:
                source_label += (
                    f", page {page}"
                )

            context_parts.append(
                f"""
[Source {index}: {source_label}]
{document["text"]}
"""
            )

        return "\n".join(
            context_parts
        )

    def generate(
        self,
        query: str,
        documents: List[Dict]
    ) -> str:

        if not documents:

            return (
                "I could not find enough "
                "relevant information in "
                "the indexed documents."
            )

        context = self.build_context(
            documents
        )

        system_prompt = """
You are an enterprise knowledge assistant.

Answer questions only using the supplied
document context.

Rules:
1. Do not invent information.
2. If the context does not contain the answer,
   clearly state that the information is not
   available.
3. Prefer concise factual answers.
4. Cite supporting sources using
   [Source 1], [Source 2], etc.
5. Do not claim information that is not
   supported by the retrieved context.
"""

        user_prompt = f"""
DOCUMENT CONTEXT:

{context}

QUESTION:

{query}

Provide a grounded answer with source citations.
"""

        response = (
            self.client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
        )
