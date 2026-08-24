import os
import tempfile

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile
)

from pydantic import BaseModel

from src.pipeline import (
    EnterpriseRAGPipeline
)


app = FastAPI(
    title="Enterprise RAG Pipeline",
    description=(
        "Production-style Retrieval "
        "Augmented Generation API"
    ),
    version="1.0.0"
)


pipeline = (
    EnterpriseRAGPipeline()
)


class QuestionRequest(
    BaseModel
):

    question: str


@app.get("/")
def home():

    return {
        "application":
            "Enterprise RAG Pipeline",

        "status":
            "running"
    }


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...)
):

    extension = os.path.splitext(
        file.filename
    )[1]

    if extension.lower() not in {
        ".txt",
        ".pdf"
    }:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only TXT and PDF "
                "files are supported."
            )
        )

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            content = (
                await file.read()
            )

            temp_file.write(
                content
            )

            temp_path = (
                temp_file.name
            )

        result = pipeline.ingest(
            temp_path
        )

        return {
            "filename":
                file.filename,

            "status":
                "indexed",

            **result
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        if (
            "temp_path" in locals()
            and os.path.exists(
                temp_path
            )
        ):

            os.remove(
                temp_path
            )


@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail=(
                "Question cannot "
                "be empty."
            )
        )

    try:

        return pipeline.ask(
            request.question
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
