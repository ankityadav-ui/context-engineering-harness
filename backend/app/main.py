# Import FastAPI components used to create the API and handle file uploads.
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

# Import CORS middleware so the React frontend can communicate with FastAPI.
from fastapi.middleware.cors import CORSMiddleware

# Import application request and response schemas.
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentUploadResponse,
    DocumentSearchRequest,
    RAGRequest,
)

# Import the normal LLM generation function and its custom error.
from app.llm import (
    generate_response,
    LLMError,
)

# Import document saving and processing functions.
from app.rag.document_service import (
    save_document,
    process_document,
)

# Import embedding generation for semantic document search.
from app.rag.embedding_service import generate_embedding

# Import Pinecone similarity search.
from app.rag.pinecone_service import search_similar_content

from app.context_engine.engine import (
    assemble_context,
    format_context,
    resolve_image_path,
)
from app.llm import (
    generate_response,
    generate_multimodal_response,
    LLMError,
)

# Create the FastAPI application.
app = FastAPI(
    title="Context Engineering Harness",
    description="Backend API for the Context Engineering Harness.",
    version="1.0.0",
)


# Allow requests from the React development server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Basic endpoint used to verify that the backend is running.
@app.get("/")
def root():
    return {
        "message": "Context Engineering Harness API is running."
    }


# Endpoint used by the React frontend for normal chat.
@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    try:
        # Send the user's message directly to the language model.
        response = generate_response(
            request.message
        )

    except LLMError as error:
        # Return a service-unavailable error when the LLM fails.
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    # Return the generated response.
    return ChatResponse(
        response=response
    )


# Endpoint used to upload and process a PDF document.
@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
):

    try:
        # Save the uploaded PDF and generate its document ID.
        (
            document_id,
            original_filename,
            document_path,
        ) = save_document(file)

        # Process the PDF through extraction, chunking,
        # embeddings, and Pinecone indexing.
        (
            content_units,
            ocr_pages,
            table_units,
            image_units,
        ) = process_document(
            document_path=document_path,
            document_id=document_id,
        )

    except ValueError as error:
        # Return a bad-request error for invalid PDF uploads.
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    # Count text content units separately.
    text_unit_count = sum(
        1
        for unit in content_units
        if unit["content_type"] == "text"
    )

    # Return processing statistics to the frontend.
    return DocumentUploadResponse(
        document_id=document_id,
        filename=original_filename,
        status="processed",
        text_units=text_unit_count,
        ocr_pages=len(ocr_pages),
        tables=len(table_units),
        images=len(image_units),
    )


# Development endpoint used to inspect extracted multimodal content.
@app.post(
    "/documents/extract-preview",
)
async def extract_preview(
    file: UploadFile = File(...),
):

    try:
        # Save the uploaded PDF.
        (
            document_id,
            original_filename,
            document_path,
        ) = save_document(file)

        # Process the document into multimodal content units.
        (
            content_units,
            ocr_pages,
            table_units,
            image_units,
        ) = process_document(
            document_path=document_path,
            document_id=document_id,
        )

    except ValueError as error:
        # Return a bad-request error for invalid files.
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    # Return the complete extraction result for development.
    return {
        "document_id": document_id,
        "filename": original_filename,
        "content_units": content_units,
        "ocr_pages": ocr_pages,
        "tables": table_units,
        "images": image_units,
    }


# Endpoint used to perform semantic search over indexed documents.
@app.post(
    "/documents/search",
)
def search_documents(
    request: DocumentSearchRequest,
):

    try:
        # Convert the user's search query into an embedding.
        query_embedding = generate_embedding(
            request.query
        )

        # Search Pinecone for semantically similar content.
        results = search_similar_content(
            query_embedding=query_embedding,
            top_k=request.top_k,
            document_id=request.document_id,
        )

    except ValueError as error:
        # Return a bad-request error for invalid search input.
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    # Return the retrieved results.
    return {
        "query": request.query,
        "results": results,
    }


@app.post(
    "/documents/ask",
)
def ask_document(
    request: RAGRequest,
):
    """
    Answer a document question using retrieved
    text, tables, and actual images.
    """
    try:
        context = assemble_context(
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id,
        )

        formatted_context = format_context(
            context
        )

        image_paths = []

        for result in context["rag"]:
            if result["content_type"] != "image":
                continue

            image_path = resolve_image_path(
                result["content_ref"]
            )

            image_paths.append(image_path)

        prompt = f"""
You are answering a question using retrieved
document context.

Use the provided document context and images
to answer the user's question.

If the answer cannot be found in the provided
context or images, clearly say that the information
was not found.

Do not invent facts.

Retrieved Context:
{formatted_context}

User Question:
{request.query}
"""

        if image_paths:
            answer = generate_multimodal_response(
                message=prompt,
                image_paths=image_paths,
            )
        else:
            answer = generate_response(
                prompt
            )

    except LLMError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    return {
        "query": request.query,
        "answer": answer,
        "context": context,
        "images_sent_to_llm": len(image_paths),
    }