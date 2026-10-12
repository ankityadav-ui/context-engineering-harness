from pathlib import Path

from app.rag.rag_service import retrieve_context


BASE_DIR = Path(__file__).resolve().parents[2]


def resolve_image_path(content_ref: str) -> str:
    """
    Convert the stored image reference into an absolute
    filesystem path that Ollama can access.
    """
    if not content_ref:
        raise ValueError("Image content reference is empty.")

    path = Path(content_ref)

    if path.is_absolute():
        resolved_path = path
    else:
        resolved_path = BASE_DIR / "storage" / path

    resolved_path = resolved_path.resolve()

    print("CONTENT REF:", content_ref)
    print("RESOLVED IMAGE PATH:", resolved_path)
    print("IMAGE EXISTS:", resolved_path.exists())

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Image file not found: {resolved_path}"
        )

    return str(resolved_path)

def assemble_context(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> dict:
    """
    Retrieve RAG results and place them into the
    Context Engine structure.
    """
    rag_results = retrieve_context(
        query=query,
        top_k=top_k,
        document_id=document_id,
    )

    context = {
        "conversation": None,
        "memory": [],
        "rag": rag_results,
        "knowledge_graph": [],
    }

    return context


def format_context(
    context: dict,
) -> str:
    """
    Convert retrieved multimodal content into
    text that can be included in the prompt.
    """
    sections = []

    for result in context["rag"]:
        content_type = result["content_type"]
        page_number = result["page_number"]

        if content_type == "text":
            sections.append(
                f"[TEXT - Page {page_number}]\n"
                f"{result['content']}"
            )

        elif content_type == "table":
            sections.append(
                f"[TABLE - Page {page_number}]\n"
                f"{result['content']}"
            )

        elif content_type == "image":
            sections.append(
                f"[IMAGE - Page {page_number}]\n"
                f"Image reference: "
                f"{result['content_ref']}"
            )

    return "\n\n".join(sections)