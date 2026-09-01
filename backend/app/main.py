import os
import shutil
from .chat import generate_rag_answer, generate_normal_answer
from .llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from .memory import (
    create_memory,
    list_memories,
    get_memory,
    update_memory,
    delete_memory,
    build_memory_context,
    store_extracted_memories,
)
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session

from .chunker import chunk_text
from .config import MEMORY_DEFAULT_USER_ID, MEMORY_MAX_HISTORY_MESSAGES
from .database import engine
from .document_parser import extract_text
from .models import Case, ChatMessage, ChatSession, Document, DocumentChunk, EvalQuery, EvalResult, Memory
from .schemas import (
    CaseCreate,
    CaseResponse,
    ChatHistoryResponse,
    ChatMessageCreate,
    ChatMessageResponse,
    ChatMessageWithSourcesResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    EvalAggregate,
    EvalQueryCreate,
    EvalQueryDetail,
    EvalQueryResponse,
    EvalRunResponse,
    EvalResultResponse,
    MemoryCreate,
    MemoryResponse,
    MemoryUpdate,
    SourceChunk,
)
from .vector_store import add_chunks, delete_chunks_by_document, search_chunks


app = FastAPI()


# ============================================================
# DATABASE
# ============================================================

def get_db():
    with Session(engine) as session:
        yield session


# ============================================================
# CORS
# ============================================================

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


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Backend is working!"
    }


@app.get("/db-test")
def database_test():
    try:
        with engine.connect():
            return {
                "status": "Database connected!"
            }

    except Exception as e:
        return {
            "status": "Database connection failed",
            "error": str(e)
        }


# ============================================================
# CASES
# ============================================================

@app.post(
    "/cases",
    response_model=CaseResponse
)
def create_case(
    case: CaseCreate,
    db: Session = Depends(get_db)
):
    new_case = Case(
        name=case.name,
        description=case.description
    )

    db.add(new_case)
    db.commit()
    db.refresh(new_case)

    return new_case


@app.get(
    "/cases",
    response_model=list[CaseResponse]
)
def get_cases(
    db: Session = Depends(get_db)
):
    cases = (
        db.query(Case)
        .order_by(Case.created_at.desc())
        .all()
    )

    return cases


@app.get(
    "/cases/{case_id}",
    response_model=CaseResponse
)
def get_case(
    case_id: int,
    db: Session = Depends(get_db)
):
    case = (
        db.query(Case)
        .filter(Case.id == case_id)
        .first()
    )

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    return case


@app.delete("/cases/{case_id}")
def delete_case(
    case_id: int,
    db: Session = Depends(get_db)
):
    case = (
        db.query(Case)
        .filter(Case.id == case_id)
        .first()
    )

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    db.delete(case)
    db.commit()

    return {
        "message": "Case deleted successfully"
    }


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

@app.post("/cases/{case_id}/documents")
def upload_document(
    case_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Check whether case exists
    # --------------------------------------------------------

    case = (
        db.query(Case)
        .filter(Case.id == case_id)
        .first()
    )

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    # --------------------------------------------------------
    # Create upload directory
    # --------------------------------------------------------

    upload_dir = os.path.join(
        "uploads",
        f"case_{case_id}"
    )

    os.makedirs(
        upload_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create file path
    # --------------------------------------------------------

    file_path = os.path.join(
        upload_dir,
        file.filename
    )

    # --------------------------------------------------------
    # Save uploaded file
    # --------------------------------------------------------

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    try:
        extracted_text = extract_text(
            file_path
        )

    except ValueError as error:

        if os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        if os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {error}"
        )

    # --------------------------------------------------------
    # Create document database record
    # --------------------------------------------------------

    document = Document(
        case_id=case_id,
        filename=file.filename,
        file_path=file_path,
        text_content=extracted_text,
    )

    db.add(document)

    db.commit()

    db.refresh(document)

    # --------------------------------------------------------
    # Split extracted text into chunks
    # --------------------------------------------------------

    chunks = chunk_text(
        extracted_text
    )

    # --------------------------------------------------------
    # Create PostgreSQL chunk records
    # --------------------------------------------------------

    chunk_records = []

    for index, chunk in enumerate(chunks):

        document_chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=index,
            text=chunk,
        )

        db.add(document_chunk)

        chunk_records.append(
            document_chunk
        )

    db.commit()

    # --------------------------------------------------------
    # Refresh chunks to get generated IDs
    # --------------------------------------------------------

    for chunk_record in chunk_records:
        db.refresh(chunk_record)

    # --------------------------------------------------------
    # Prepare chunks for ChromaDB
    # --------------------------------------------------------

    vector_chunks = [
        {
            "id": chunk.id,
            "text": chunk.text,
            "case_id": case_id,
            "document_id": document.id,
            "chunk_index": chunk.chunk_index,
        }
        for chunk in chunk_records
    ]

    # --------------------------------------------------------
    # Add chunks to ChromaDB
    # --------------------------------------------------------

    try:
        add_chunks(
            vector_chunks
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Vector database processing failed: {error}"
        )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "message": "Document uploaded, extracted, chunked and embedded successfully",

        "document": {
            "id": document.id,
            "case_id": document.case_id,
            "filename": document.filename,
            "file_path": document.file_path,
        },

        "text": extracted_text,

        "text_length": len(extracted_text),

        "chunk_count": len(chunks),
    }


# ============================================================
# GET DOCUMENTS FOR A CASE
# ============================================================

@app.get("/cases/{case_id}/documents")
def get_case_documents(
    case_id: int,
    db: Session = Depends(get_db),
):
    # Check case
    case = (
        db.query(Case)
        .filter(Case.id == case_id)
        .first()
    )

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    documents = (
        db.query(Document)
        .filter(Document.case_id == case_id)
        .order_by(Document.created_at.desc())
        .all()
    )

    return documents


# ============================================================
# GET SINGLE DOCUMENT
# ============================================================

@app.get(
    "/cases/{case_id}/documents/{document_id}"
)
def get_document(
    case_id: int,
    document_id: int,
    db: Session = Depends(get_db),
):
    document = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.case_id == case_id,
        )
        .first()
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    return {
        "id": document.id,
        "case_id": document.case_id,
        "filename": document.filename,
        "file_path": document.file_path,
        "text_content": document.text_content,
        "created_at": document.created_at,
    }


# ============================================================
# GET DOCUMENT CHUNKS
# ============================================================

@app.get(
    "/cases/{case_id}/documents/{document_id}/chunks"
)
def get_document_chunks(
    case_id: int,
    document_id: int,
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Check document
    # --------------------------------------------------------

    document = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.case_id == case_id,
        )
        .first()
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    # --------------------------------------------------------
    # Get chunks
    # --------------------------------------------------------

    chunks = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_id == document_id
        )
        .order_by(
            DocumentChunk.chunk_index
        )
        .all()
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "document_id": document_id,
        "filename": document.filename,
        "chunk_count": len(chunks),

        "chunks": [
            {
                "id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "created_at": chunk.created_at,
            }
            for chunk in chunks
        ],
    }

# ============================================================
# DELETE DOCUMENT
# ============================================================

@app.delete("/cases/{case_id}/documents/{document_id}")
def delete_document(
    case_id: int,
    document_id: int,
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Check document exists
    # --------------------------------------------------------

    document = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.case_id == case_id,
        )
        .first()
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    # --------------------------------------------------------
    # Remove vectors from ChromaDB
    # --------------------------------------------------------

    try:
        delete_chunks_by_document(
            document_id=document_id
        )
    except Exception:
        pass  # Best effort - vectors may not exist

    # --------------------------------------------------------
    # Remove uploaded file from disk
    # --------------------------------------------------------

    if document.file_path and os.path.exists(document.file_path):
        os.remove(document.file_path)

    # --------------------------------------------------------
    # Delete document (cascade deletes chunks)
    # --------------------------------------------------------

    db.delete(document)
    db.commit()

    return {
        "message": "Document deleted successfully"
    }


# ============================================================
# EVALUATION
# ============================================================


@app.post(
    "/cases/{case_id}/eval-queries",
    response_model=EvalQueryResponse,
)
def create_eval_query(
    case_id: int,
    data: EvalQueryCreate,
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Check whether case exists
    # --------------------------------------------------------

    case = (
        db.query(Case)
        .filter(Case.id == case_id)
        .first()
    )

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not data.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    # Note: empty expected_document_ids is allowed for negative/unanswerable queries

    # --------------------------------------------------------
    # Create eval query
    # --------------------------------------------------------

    import json

    eval_query = EvalQuery(
        case_id=case_id,
        query=data.query,
        description=data.description,
        expected_document_ids=json.dumps(data.expected_document_ids),
        expected_chunk_indices=(
            json.dumps(data.expected_chunk_indices)
            if data.expected_chunk_indices
            else None
        ),
    )

    db.add(eval_query)
    db.commit()
    db.refresh(eval_query)

    return EvalQueryResponse(
        id=eval_query.id,
        case_id=eval_query.case_id,
        query=eval_query.query,
        description=eval_query.description,
        expected_document_ids=data.expected_document_ids,
        expected_chunk_indices=data.expected_chunk_indices,
        created_at=eval_query.created_at,
    )


@app.get(
    "/cases/{case_id}/eval-queries",
    response_model=list[EvalQueryDetail],
)
def get_eval_queries(
    case_id: int,
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Check whether case exists
    # --------------------------------------------------------

    case = (
        db.query(Case)
        .filter(Case.id == case_id)
        .first()
    )

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    # --------------------------------------------------------
    # Get queries with last result
    # --------------------------------------------------------

    import json

    queries = (
        db.query(EvalQuery)
        .filter(EvalQuery.case_id == case_id)
        .order_by(EvalQuery.created_at.desc())
        .all()
    )

    result = []
    for q in queries:
        last_result = (
            db.query(EvalResult)
            .filter(EvalResult.eval_query_id == q.id)
            .order_by(EvalResult.created_at.desc())
            .first()
        )

        last_result_resp = None
        if last_result:
            last_result_resp = EvalResultResponse(
                id=last_result.id,
                eval_query_id=last_result.eval_query_id,
                top_k=last_result.top_k,
                retrieved_doc_ids=json.loads(last_result.retrieved_doc_ids),
                retrieved_distances=json.loads(last_result.retrieved_distances),
                hit_at_k=last_result.hit_at_k,
                precision_at_k=last_result.precision_at_k,
                recall_at_k=last_result.recall_at_k,
                reciprocal_rank=last_result.reciprocal_rank,
                passed=last_result.passed,
                created_at=last_result.created_at,
            )

        result.append(
            EvalQueryDetail(
                id=q.id,
                case_id=q.case_id,
                query=q.query,
                description=q.description,
                expected_document_ids=json.loads(q.expected_document_ids),
                expected_chunk_indices=(
                    json.loads(q.expected_chunk_indices)
                    if q.expected_chunk_indices
                    else None
                ),
                created_at=q.created_at,
                last_result=last_result_resp,
            )
        )

    return result


@app.delete("/eval-queries/{query_id}")
def delete_eval_query(
    query_id: int,
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Check whether query exists
    # --------------------------------------------------------

    eval_query = (
        db.query(EvalQuery)
        .filter(EvalQuery.id == query_id)
        .first()
    )

    if eval_query is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation query not found"
        )

    # --------------------------------------------------------
    # Delete (cascade deletes results)
    # --------------------------------------------------------

    db.delete(eval_query)
    db.commit()

    return {
        "message": "Evaluation query deleted successfully"
    }


@app.post(
    "/cases/{case_id}/evaluation/run",
    response_model=EvalRunResponse,
)
def run_evaluation(
    case_id: int,
    top_k: int = 5,
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Check whether case exists
    # --------------------------------------------------------

    case = (
        db.query(Case)
        .filter(Case.id == case_id)
        .first()
    )

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    # --------------------------------------------------------
    # Get all evaluation queries for this case
    # --------------------------------------------------------

    queries = (
        db.query(EvalQuery)
        .filter(EvalQuery.case_id == case_id)
        .all()
    )

    if not queries:
        raise HTTPException(
            status_code=400,
            detail="No evaluation queries found for this case"
        )

    # --------------------------------------------------------
    # Run evaluation for each query
    # --------------------------------------------------------

    from .evaluation import evaluate_single_query, calculate_aggregate

    import json

    query_results = []

    for eval_query in queries:
        try:
            result = evaluate_single_query(
                eval_query=eval_query,
                top_k=top_k,
                db=db,
            )

            # --------------------------------------------------------
            # Save result to database
    # --------------------------------------------------------

            db_result = EvalResult(
                eval_query_id=eval_query.id,
                top_k=top_k,
                retrieved_doc_ids=json.dumps(result["retrieved_doc_ids"]),
                retrieved_distances=json.dumps(result["retrieved_distances"]),
                hit_at_k=result["hit_at_k"],
                precision_at_k=result["precision_at_k"],
                recall_at_k=result["recall_at_k"],
                reciprocal_rank=result["reciprocal_rank"],
                passed=result["passed"],
            )

            db.add(db_result)

            query_results.append({
                "query_id": eval_query.id,
                "query": eval_query.query,
                "description": eval_query.description,
                "expected_document_ids": result["expected_document_ids"],
                "retrieved_doc_ids": result["retrieved_doc_ids"],
                "retrieved_distances": result["retrieved_distances"],
                "retrieved_chunks": result["retrieved_chunks"],
                "hit_at_k": result["hit_at_k"],
                "precision_at_k": result["precision_at_k"],
                "recall_at_k": result["recall_at_k"],
                "reciprocal_rank": result["reciprocal_rank"],
                "passed": result["passed"],
                "avg_distance": result["avg_distance"],
            })

        except Exception as error:
            query_results.append({
                "query_id": eval_query.id,
                "query": eval_query.query,
                "description": eval_query.description,
                "error": str(error),
                "passed": False,
            })

    # --------------------------------------------------------
    # Save all results
    # --------------------------------------------------------

    db.commit()

    # --------------------------------------------------------
    # Calculate aggregate metrics
    # --------------------------------------------------------

    successful_results = [
        r for r in query_results
        if "error" not in r
    ]

    aggregate = calculate_aggregate(successful_results)

    return EvalRunResponse(
        case_id=case_id,
        top_k=top_k,
        aggregate=aggregate,
        query_results=query_results,
    )


@app.post(
    "/cases/{case_id}/eval-queries/seed",
)
def seed_eval_queries(
    case_id: int,
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Check whether case exists
    # --------------------------------------------------------

    case = (
        db.query(Case)
        .filter(Case.id == case_id)
        .first()
    )

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    # --------------------------------------------------------
    # Get documents for this case
    # --------------------------------------------------------

    import json

    documents = (
        db.query(Document)
        .filter(Document.case_id == case_id)
        .all()
    )

    if not documents:
        raise HTTPException(
            status_code=400,
            detail="No documents found. Upload documents first."
        )

    # --------------------------------------------------------
    # Check if queries already exist
    # --------------------------------------------------------

    existing = (
        db.query(EvalQuery)
        .filter(EvalQuery.case_id == case_id)
        .count()
    )

    if existing > 0:
        raise HTTPException(
            status_code=400,
            detail=f"{existing} evaluation queries already exist. Delete them first."
        )

    # --------------------------------------------------------
    # Build seed queries based on actual documents
    # --------------------------------------------------------

    # Get document filenames for context
    doc_map = {d.id: d.filename for d in documents}
    first_doc_id = documents[0].id
    first_doc_name = documents[0].filename

    seed_queries = [
        {
            "query": "What topics are covered in Module 1?",
            "description": "Query about Module 1 content - should find relevant chunks from the teaching plan",
            "expected_document_ids": [first_doc_id],
        },
        {
            "query": "What is the course schedule and grading policy?",
            "description": "Query about course logistics - should find schedule/grading sections",
            "expected_document_ids": [first_doc_id],
        },
        {
            "query": "What are the prerequisites for this course?",
            "description": "Query about prerequisites - should find relevant section if present",
            "expected_document_ids": [first_doc_id],
        },
    ]

    # --------------------------------------------------------
    # Create evaluation queries
    # --------------------------------------------------------

    created = []

    for sq in seed_queries:
        eval_query = EvalQuery(
            case_id=case_id,
            query=sq["query"],
            description=sq["description"],
            expected_document_ids=json.dumps(sq["expected_document_ids"]),
        )
        db.add(eval_query)
        created.append(sq["query"])

    db.commit()

    return {
        "message": f"Seeded {len(created)} evaluation queries",
        "queries": created,
        "documents": doc_map,
    }


# ============================================================
# RAG RETRIEVAL DEBUG
# ============================================================

@app.get("/cases/{case_id}/search/debug")
def debug_search(
    case_id: int,
    query: str,
    top_k: int = 5,
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Check whether case exists
    # --------------------------------------------------------

    case = (
        db.query(Case)
        .filter(Case.id == case_id)
        .first()
    )

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty"
        )

    if top_k < 1 or top_k > 20:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20"
        )

    # --------------------------------------------------------
    # Run the full context engineering pipeline
    # --------------------------------------------------------

    from .rag_context import build_rag_context

    try:
        result = build_rag_context(
            query=query,
            case_id=case_id,
            top_k=top_k,
            db=db,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {error}"
        )

    # --------------------------------------------------------
    # Build accepted/rejected from raw retrieved chunks
    # --------------------------------------------------------

    from .config import RAG_DISTANCE_THRESHOLD

    raw_chunks = result.get("retrieved_chunks", [])
    accepted = [
        c for c in raw_chunks
        if c["distance"] <= RAG_DISTANCE_THRESHOLD
    ]
    rejected = [
        c for c in raw_chunks
        if c["distance"] > RAG_DISTANCE_THRESHOLD
    ]

    # --------------------------------------------------------
    # Resolve filenames for all chunks
    # --------------------------------------------------------

    all_doc_ids = list(set(
        c.get("document_id")
        for c in raw_chunks
        if c.get("document_id")
    ))

    filename_map = {}
    if all_doc_ids:
        docs = (
            db.query(Document)
            .filter(Document.id.in_(all_doc_ids))
            .all()
        )
        filename_map = {d.id: d.filename for d in docs}

    for chunk in raw_chunks:
        chunk["filename"] = filename_map.get(
            chunk.get("document_id"), "unknown"
        )
    for chunk in accepted:
        chunk["filename"] = filename_map.get(
            chunk.get("document_id"), "unknown"
        )
    for chunk in rejected:
        chunk["filename"] = filename_map.get(
            chunk.get("document_id"), "unknown"
        )

    # --------------------------------------------------------
    # Build response
    # --------------------------------------------------------

    metadata = result.get("metadata", {})

    return {
        "query": query,
        "case_id": case_id,
        "requested_top_k": top_k,
        "statistics": metadata,
        "retrieved_chunks": raw_chunks,
        "accepted_chunks": accepted,
        "rejected_chunks": rejected,
        "final_context_chunks": result.get("chunks", []),
        "deduplicated_chunks": result.get("deduplicated_chunks", []),
        "context": result.get("context", ""),
    }



# ============================================================
# SEMANTIC SEARCH
# ============================================================

@app.get("/cases/{case_id}/search")
def semantic_search(
    case_id: int,
    query: str,
    top_k: int = 5,
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Check whether case exists
    # --------------------------------------------------------

    case = (
        db.query(Case)
        .filter(Case.id == case_id)
        .first()
    )

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty"
        )

    # --------------------------------------------------------
    # Validate top_k
    # --------------------------------------------------------

    if top_k < 1 or top_k > 20:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20"
        )

    # --------------------------------------------------------
    # Search ChromaDB
    # --------------------------------------------------------

    try:
        results = search_chunks(
            query=query,
            case_id=case_id,
            top_k=top_k,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {error}"
        )

    # --------------------------------------------------------
    # Format results
    # --------------------------------------------------------

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    search_results = []

    for index, document_text in enumerate(documents):

        metadata = metadatas[index]

        search_results.append(
            {
                "text": document_text,
                "case_id": metadata.get("case_id"),
                "document_id": metadata.get("document_id"),
                "chunk_index": metadata.get("chunk_index"),
                "distance": distances[index],
            }
        )

    # --------------------------------------------------------
    # Resolve filenames (batch query)
    # --------------------------------------------------------

    if search_results:
        document_ids = list(
            set(r["document_id"] for r in search_results if r.get("document_id"))
        )
        if document_ids:
            docs = (
                db.query(Document)
                .filter(Document.id.in_(document_ids))
                .all()
            )
            filename_map = {d.id: d.filename for d in docs}
            for r in search_results:
                r["filename"] = filename_map.get(
                    r.get("document_id"), "unknown"
                )

    return {
        "query": query,
        "case_id": case_id,
        "result_count": len(search_results),
        "results": search_results,
    }

# ============================================================
# RAG CHAT (Legacy - backward compatible)
# ============================================================

@app.get("/cases/{case_id}/chat")
def chat(
    case_id: int,
    query: str,
    top_k: int = 5,
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Check whether case exists
    # --------------------------------------------------------

    case = (
        db.query(Case)
        .filter(Case.id == case_id)
        .first()
    )

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    # --------------------------------------------------------
    # Validate top_k
    # --------------------------------------------------------

    if top_k < 1 or top_k > 20:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20"
        )

    # --------------------------------------------------------
    # Generate RAG answer
    # --------------------------------------------------------

    try:
        result = generate_rag_answer(
            query=query,
            case_id=case_id,
            top_k=top_k,
            db=db,
        )

    except LLMAuthenticationError as error:
        raise HTTPException(
            status_code=502,
            detail=f"LLM authentication failed: {error}"
        )
    except LLMRateLimitError as error:
        raise HTTPException(
            status_code=429,
            detail=f"LLM rate limit exceeded: {error}"
        )
    except LLMTimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail=f"LLM request timed out: {error}"
        )
    except LLMConfigurationError as error:
        raise HTTPException(
            status_code=500,
            detail=f"LLM configuration error: {error}"
        )
    except LLMProviderError as error:
        raise HTTPException(
            status_code=502,
            detail=f"LLM provider error: {error}"
        )

    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return result


# ============================================================
# CHAT SESSIONS
# ============================================================

@app.post(
    "/cases/{case_id}/chats",
    response_model=ChatSessionResponse,
)
def create_chat_session(
    case_id: int,
    session_data: ChatSessionCreate,
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Check whether case exists
    # --------------------------------------------------------

    case = (
        db.query(Case)
        .filter(Case.id == case_id)
        .first()
    )

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    # --------------------------------------------------------
    # Create chat session
    # --------------------------------------------------------

    new_session = ChatSession(
        case_id=case_id,
        title=session_data.title,
        chat_mode=session_data.chat_mode,
    )

    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    return new_session


@app.get(
    "/cases/{case_id}/chats",
    response_model=list[ChatSessionResponse],
)
def get_chat_sessions(
    case_id: int,
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Check whether case exists
    # --------------------------------------------------------

    case = (
        db.query(Case)
        .filter(Case.id == case_id)
        .first()
    )

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    # --------------------------------------------------------
    # Get sessions ordered by most recently updated
    # --------------------------------------------------------

    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.case_id == case_id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )

    return sessions


@app.get(
    "/chats/{chat_id}",
    response_model=ChatHistoryResponse,
)
def get_chat_history(
    chat_id: int,
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Check whether chat session exists
    # --------------------------------------------------------

    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == chat_id)
        .first()
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found"
        )

    # --------------------------------------------------------
    # Get messages in chronological order
    # --------------------------------------------------------

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == chat_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    return ChatHistoryResponse(
        id=session.id,
        case_id=session.case_id,
        title=session.title,
        chat_mode=session.chat_mode,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=messages,
    )


@app.post(
    "/chats/{chat_id}/messages",
    response_model=ChatMessageWithSourcesResponse,
)
def send_chat_message(
    chat_id: int,
    message_data: ChatMessageCreate,
    top_k: int = 3,
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Check whether chat session exists
    # --------------------------------------------------------

    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == chat_id)
        .first()
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found"
        )

    # --------------------------------------------------------
    # Validate message content
    # --------------------------------------------------------

    if not message_data.content.strip():
        raise HTTPException(
            status_code=400,
            detail="Message content cannot be empty"
        )

    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    user_message = ChatMessage(
        session_id=chat_id,
        role="user",
        content=message_data.content,
    )

    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    # --------------------------------------------------------
    # SHORT-TERM MEMORY: Load recent conversation history
    # --------------------------------------------------------

    history_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == chat_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    # Exclude the just-saved user message from history;
    # it will be added separately by the chat functions.
    conversation_history = []
    for msg in history_messages[:-1]:  # all except the last (current)
        conversation_history.append({
            "role": msg.role,
            "content": msg.content,
        })

    # Limit to configurable window
    if len(conversation_history) > MEMORY_MAX_HISTORY_MESSAGES:
        conversation_history = conversation_history[-MEMORY_MAX_HISTORY_MESSAGES:]

    # --------------------------------------------------------
    # LONG-TERM MEMORY: Retrieve relevant memories
    # --------------------------------------------------------

    memory_context = build_memory_context(
        query=message_data.content,
        user_id=MEMORY_DEFAULT_USER_ID,
        case_id=session.case_id,
        db=db,
    )

    # --------------------------------------------------------
    # Generate answer based on chat mode
    # --------------------------------------------------------

    try:
        if session.chat_mode == "normal":
            rag_result = generate_normal_answer(
                query=message_data.content,
                conversation_history=conversation_history,
                memory_context=memory_context,
            )
        else:
            rag_result = generate_rag_answer(
                query=message_data.content,
                case_id=session.case_id,
                top_k=top_k,
                db=db,
                conversation_history=conversation_history,
                memory_context=memory_context,
            )

    except LLMAuthenticationError as error:
        raise HTTPException(
            status_code=502,
            detail=f"LLM authentication failed: {error}"
        )
    except LLMRateLimitError as error:
        raise HTTPException(
            status_code=429,
            detail=f"LLM rate limit exceeded: {error}"
        )
    except LLMTimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail=f"LLM request timed out: {error}"
        )
    except LLMConfigurationError as error:
        raise HTTPException(
            status_code=500,
            detail=f"LLM configuration error: {error}"
        )
    except LLMProviderError as error:
        raise HTTPException(
            status_code=502,
            detail=f"LLM provider error: {error}"
        )

    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    assistant_message = ChatMessage(
        session_id=chat_id,
        role="assistant",
        content=rag_result["answer"],
    )

    db.add(assistant_message)

    # --------------------------------------------------------
    # Update session timestamp
    # --------------------------------------------------------

    from datetime import datetime
    session.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(assistant_message)

    # --------------------------------------------------------
    # LONG-TERM MEMORY EXTRACTION: Analyze conversation for durable info
    # --------------------------------------------------------

    try:
        store_extracted_memories(
            user_message=message_data.content,
            assistant_response=rag_result["answer"],
            user_id=MEMORY_DEFAULT_USER_ID,
            case_id=session.case_id,
            db=db,
        )
    except Exception:
        pass  # Extraction failure must not break the chat response

    # --------------------------------------------------------
    # Build response with sources
    # --------------------------------------------------------

    sources = []
    for chunk in rag_result.get("chunks", []):
        sources.append(
            SourceChunk(
                document_id=chunk.get("document_id"),
                filename=chunk.get("filename", "unknown"),
                chunk_index=chunk.get("chunk_index"),
                text=chunk.get("text"),
                distance=chunk.get("distance"),
            )
        )

    return ChatMessageWithSourcesResponse(
        id=assistant_message.id,
        session_id=assistant_message.session_id,
        role=assistant_message.role,
        content=assistant_message.content,
        created_at=assistant_message.created_at,
        sources=sources,
        chunk_count=rag_result.get("chunk_count", 0),
    )


@app.delete("/chats/{chat_id}")
def delete_chat_session(
    chat_id: int,
    db: Session = Depends(get_db),
):
    # --------------------------------------------------------
    # Check whether chat session exists
    # --------------------------------------------------------

    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == chat_id)
        .first()
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found"
        )

    # --------------------------------------------------------
    # Delete session (cascade deletes messages)
    # --------------------------------------------------------

    db.delete(session)
    db.commit()

    return {
        "message": "Chat session deleted successfully"
    }


# ============================================================
# LONG-TERM MEMORY
# ============================================================


@app.post(
    "/memories",
    response_model=MemoryResponse,
)
def create_memory_endpoint(
    data: MemoryCreate,
    user_id: str = Query(default=MEMORY_DEFAULT_USER_ID),
    db: Session = Depends(get_db),
):
    memory = create_memory(
        content=data.content,
        memory_type=data.memory_type,
        user_id=user_id,
        case_id=data.case_id,
        importance=data.importance,
        db=db,
    )

    return memory


@app.get(
    "/memories",
    response_model=list[MemoryResponse],
)
def list_memories_endpoint(
    user_id: str = Query(default=MEMORY_DEFAULT_USER_ID),
    case_id: int | None = Query(default=None),
    memory_type: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    memories = list_memories(
        user_id=user_id,
        case_id=case_id,
        memory_type=memory_type,
        db=db,
    )

    return memories


@app.get(
    "/memories/{memory_id}",
    response_model=MemoryResponse,
)
def get_memory_endpoint(
    memory_id: int,
    user_id: str = Query(default=MEMORY_DEFAULT_USER_ID),
    db: Session = Depends(get_db),
):
    memory = get_memory(memory_id, user_id, db)

    if memory is None:
        raise HTTPException(
            status_code=404,
            detail="Memory not found"
        )

    return memory


@app.put(
    "/memories/{memory_id}",
    response_model=MemoryResponse,
)
def update_memory_endpoint(
    memory_id: int,
    data: MemoryUpdate,
    user_id: str = Query(default=MEMORY_DEFAULT_USER_ID),
    db: Session = Depends(get_db),
):
    memory = update_memory(
        memory_id=memory_id,
        content=data.content,
        memory_type=data.memory_type,
        importance=data.importance,
        user_id=user_id,
        db=db,
    )

    if memory is None:
        raise HTTPException(
            status_code=404,
            detail="Memory not found"
        )

    return memory


@app.delete("/memories/{memory_id}")
def delete_memory_endpoint(
    memory_id: int,
    user_id: str = Query(default=MEMORY_DEFAULT_USER_ID),
    db: Session = Depends(get_db),
):
    deleted = delete_memory(
        memory_id=memory_id,
        user_id=user_id,
        db=db,
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Memory not found"
        )

    return {
        "message": "Memory deleted successfully"
    }