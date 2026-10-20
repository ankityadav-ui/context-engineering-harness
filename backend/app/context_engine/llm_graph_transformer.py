"""LangChain LLMGraphTransformer integration for knowledge-graph ingestion.

Gemini is the single model provider; the transformer uses the project's
shared model configuration and credentials.

Responsibilities kept in this module:

* build a transformer from the project's shared Gemini model;
* constrain extraction to the project's node and relationship schema;
* normalise LLM output into the graph's stable ``TYPE:slug`` entity ids;
* reject implausible and conflicting entity types;
* require verbatim, source-grounded evidence for relationships;
* validate/deduplicate nodes and relationships *before* any Neo4j write;
* attach page and source-unit references so provenance survives the write;
* audit stored nodes read-only (no mutation).

Nothing here writes to Neo4j.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from langchain_core.documents import Document

from app.context_engine.entity_validation import (
    entity_comparison_key,
    is_plausible_entity,
    is_usable_entity_name,
)

logger = logging.getLogger("app.graph_extraction")

# --------------------------------------------------
# Schema
# --------------------------------------------------

# Node types the graph accepts, in project (stored) form.
ALLOWED_ENTITY_TYPES = (
    "PERSON",
    "ORGANIZATION",
    "TECHNOLOGY",
    "PRODUCT",
    "LOCATION",
    "DOCUMENT",
)

# Relationship types the graph accepts (mirrors graph_service allowlist).
ALLOWED_RELATIONSHIP_TYPES = (
    "RELATES_TO",
    "CONCERNS",
    "PRESCRIBED",
    "MENTIONS",
    "PART_OF",
)

# LLM node label -> stored entity_type.
ENTITY_TYPE_ALIASES = {
    "person": "PERSON",
    "people": "PERSON",
    "human": "PERSON",
    "organization": "ORGANIZATION",
    "organisation": "ORGANIZATION",
    "org": "ORGANIZATION",
    "company": "ORGANIZATION",
    "business": "ORGANIZATION",
    "institution": "ORGANIZATION",
    "technology": "TECHNOLOGY",
    "tech": "TECHNOLOGY",
    "framework": "TECHNOLOGY",
    "software": "TECHNOLOGY",
    "library": "TECHNOLOGY",
    "database": "TECHNOLOGY",
    "language": "TECHNOLOGY",
    "product": "PRODUCT",
    "service": "PRODUCT",
    "location": "LOCATION",
    "place": "LOCATION",
    "gpe": "LOCATION",
    "loc": "LOCATION",
    "city": "LOCATION",
    "country": "LOCATION",
    "document": "DOCUMENT",
    "file": "DOCUMENT",
}

# Terms that must never be stored as a person or a place. An LLM (or spaCy)
# labelling these as PERSON/GPE is a model error, not evidence.
TECHNICAL_TERMS = frozenset(
    {
        "ai",
        "artificial intelligence",
        "ml",
        "machine learning",
        "llm",
        "large language model",
        "rag",
        "retrieval augmented generation",
        "sql",
        "fastapi",
        "neo4j",
        "postgresql",
        "postgres",
        "pinecone",
        "python",
        "flutter",
        "langchain",
        "docker",
        "kubernetes",
    }
)

# Terms that are legitimately technologies/products when the LLM says so.
TECHNOLOGY_TERMS = TECHNICAL_TERMS

EXTRACTION_INSTRUCTIONS = (
    "Extraction rules for this knowledge graph:\n"
    "- Allowed node types are exactly: " + ", ".join(ALLOWED_ENTITY_TYPES) + ".\n"
    "- Allowed relationship types are exactly: "
    + ", ".join(ALLOWED_RELATIONSHIP_TYPES)
    + ".\n"
    "- Never label a technology, framework, database, programming language or "
    "product as Person or Location. AI, ML, LLM, RAG, SQL, FastAPI, Neo4j, "
    "PostgreSQL, Pinecone, Python and Flutter are Technology (or Product), "
    "never Person or Location.\n"
    "- Use the surrounding document to disambiguate. A capitalized token is not "
    "evidence of a person.\n"
    "- If the evidence is insufficient for a type, omit the node instead of "
    "guessing a type.\n"
    "- For every relationship, provide an `evidence` property containing an "
    "exact, contiguous quote from this source unit. The quote must contain "
    "both endpoint names and language that supports the relationship type. "
    "Do not infer a relationship from names merely co-occurring.\n"
    "- Use the full identifier for an entity, and the same identifier for every "
    "mention of it."
)

_ANONYMIZED_ALIAS_PATTERN = re.compile(
    r"^(?:person|organization|place)\s+[A-Z]$",
    flags=re.IGNORECASE,
)

_RELATIONSHIP_EVIDENCE_PATTERNS = {
    "RELATES_TO": re.compile(
        r"\b(?:relat(?:e|es|ed|ing)|associat(?:e|es|ed|ion)|"
        r"connect(?:s|ed|ing)|linked|influenc(?:e|es|ed)|"
        r"uses?|utiliz(?:e|es|ed|ing)|owns?|founded|created|develop(?:s|ed)|"
        r"works? with|interact(?:s|ed|ing)|caus(?:e|es|ed)|"
        r"led to|produced)\b",
        flags=re.IGNORECASE,
    ),
    "CONCERNS": re.compile(
        r"\b(?:concern(?:s|ed|ing)?|about|regard(?:s|ing)?|"
        r"focus(?:es|ed)? on|address(?:es|ed)?|pertain(?:s|ed|ing)? to|"
        r"describ(?:e|es|ed))\b",
        flags=re.IGNORECASE,
    ),
    "PRESCRIBED": re.compile(
        r"\b(?:prescrib(?:e|es|ed|ing)|recommend(?:s|ed|ing)|"
        r"administer(?:s|ed|ing)|ordered|medicat(?:e|es|ed|ion))\b",
        flags=re.IGNORECASE,
    ),
    "MENTIONS": re.compile(
        r"\b(?:mention(?:s|ed|ing)?|cit(?:e|es|ed|ing)|"
        r"refer(?:s|red|ring)? to|named)\b",
        flags=re.IGNORECASE,
    ),
    "PART_OF": re.compile(
        r"\b(?:part of|belongs? to|member of|component of|within|"
        r"subset of|division of|subsidiary of)\b",
        flags=re.IGNORECASE,
    ),
}


# --------------------------------------------------
# Results
# --------------------------------------------------


@dataclass
class ExtractedNode:
    """A validated node ready to be written to Neo4j."""

    entity_id: str
    name: str
    entity_type: str
    page_numbers: list[int] = field(default_factory=list)
    document_id: str | None = None
    source_refs: list[str] = field(default_factory=list)


@dataclass
class ExtractedRelationship:
    """A validated relationship ready to be written to Neo4j."""

    source_id: str
    target_id: str
    relationship_type: str
    page_numbers: list[int] = field(default_factory=list)
    document_id: str | None = None
    source_refs: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Outcome of one document extraction."""

    nodes: list[ExtractedNode] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)
    backend: str = "llm"
    provider: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "entity_id": node.entity_id,
                    "name": node.name,
                    "entity_type": node.entity_type,
                    "page_numbers": list(node.page_numbers),
                    "document_id": node.document_id,
                    "source_refs": list(node.source_refs),
                }
                for node in self.nodes
            ],
            "relationships": [
                {
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "relationship_type": rel.relationship_type,
                    "page_numbers": list(rel.page_numbers),
                    "document_id": rel.document_id,
                    "source_refs": list(rel.source_refs),
                }
                for rel in self.relationships
            ],
            "dropped": list(self.dropped),
            "backend": self.backend,
            "provider": self.provider,
        }


class GraphExtractionError(RuntimeError):
    """Raised when Gemini cannot produce a valid graph."""


# --------------------------------------------------
# Entity identity helpers
# --------------------------------------------------


def canonical_entity_type(raw_type: str | None) -> str | None:
    """Map an LLM/spaCy label onto the stored schema, or None if unsupported."""

    if not raw_type:
        return None

    folded = str(raw_type).strip().casefold()

    return ENTITY_TYPE_ALIASES.get(folded)


def normalize_name(raw_name: str | None) -> str:
    """Collapse whitespace in an extracted name."""

    return " ".join(str(raw_name or "").split()).strip()


def is_technical_term(name: str) -> bool:
    """True when the name is a technical term that must not read as a human/place."""

    folded = normalize_name(name).casefold()

    return folded in TECHNICAL_TERMS


def build_entity_id(entity_type: str, name: str) -> str:
    """Deterministic ``type:slug`` id, matching graph_service conventions."""

    normalized = entity_comparison_key(name)
    slug = "".join(
        character
        if character.isalnum() or character in " -"
        else "_"
        for character in normalized
    ).strip("_")[:120]
    slug = slug.replace(" ", "_")

    return f"{entity_type.lower()}:{slug}"


# --------------------------------------------------
# LLM / transformer construction
# --------------------------------------------------


def build_transformer(llm=None, *, provider: str = "gemini"):
    """Create an ``LLMGraphTransformer`` constrained to the project schema."""

    if provider != "gemini":
        from app.llm import LLMError

        raise LLMError("Graph extraction supports Gemini only.")

    from langchain_experimental.graph_transformers import LLMGraphTransformer

    if llm is None:
        from app.llm import build_graph_extraction_llm

        llm = build_graph_extraction_llm(provider)

    return LLMGraphTransformer(
        llm=llm,
        allowed_nodes=list(ALLOWED_ENTITY_TYPES),
        allowed_relationships=list(ALLOWED_RELATIONSHIP_TYPES),
        relationship_properties=["evidence"],
        additional_instructions=EXTRACTION_INSTRUCTIONS,
        # Non-strict: the LLM may answer with a slightly different label
        # spelling; canonical_entity_type() below is the real gate.
        strict_mode=False,
    )


def llm_extraction_enabled() -> bool:
    """Whether ingestion should use the LLM extractor.

    Defaults to on. ``GRAPH_EXTRACTION_BACKEND=spacy`` disables it explicitly
    (the offline/legacy path), which is never entered silently.
    """

    configured = os.getenv("GRAPH_EXTRACTION_BACKEND", "llm").strip().casefold()

    return configured != "spacy"


# --------------------------------------------------
# Validation (runs before any Neo4j write)
# --------------------------------------------------


def _raw_node_fields(node: Any) -> tuple[str, str, str]:
    """Read (id, type, name) from a LangChain Node-like object or dict."""

    if isinstance(node, dict):
        return (
            str(node.get("id") or ""),
            str(node.get("type") or ""),
            str(node.get("name") or node.get("id") or ""),
        )

    node_id = str(getattr(node, "id", "") or "")
    node_type = str(getattr(node, "type", "") or "")

    return node_id, node_type, node_id


def _raw_relationship_fields(rel: Any) -> tuple[str, str, str, str]:
    """Read endpoints, type and evidence from a relationship-like object."""

    if isinstance(rel, dict):
        properties = rel.get("properties") or {}
        return (
            str(rel.get("source_id") or rel.get("source") or ""),
            str(rel.get("target_id") or rel.get("target") or ""),
            str(rel.get("type") or rel.get("relationship_type") or ""),
            str(
                rel.get("evidence")
                or properties.get("evidence")
                or ""
            ),
        )

    source = getattr(rel, "source", None)
    target = getattr(rel, "target", None)
    properties = getattr(rel, "properties", None) or {}

    return (
        str(getattr(source, "id", source) or ""),
        str(getattr(target, "id", target) or ""),
        str(getattr(rel, "type", "") or ""),
        str(properties.get("evidence") or ""),
    )


def _relationship_evidence_is_supported(
    relationship_type: str,
    evidence: str,
    source: str,
    target: str,
    source_text: str,
) -> bool:
    """Require a verbatim source quote, both endpoints, and a type cue."""

    quote = evidence.strip()

    if not quote or quote not in source_text:
        return False

    quote_key = " ".join(quote.casefold().split())

    def contains_endpoint(name: str) -> bool:
        endpoint = " ".join(normalize_name(name).casefold().split())
        return bool(
            endpoint
            and re.search(
                rf"(?<!\w){re.escape(endpoint)}(?!\w)",
                quote_key,
            )
        )

    return (
        contains_endpoint(source)
        and contains_endpoint(target)
        and bool(
            _RELATIONSHIP_EVIDENCE_PATTERNS[relationship_type].search(
                quote
            )
        )
    )


def validate_and_clean_graph(
    raw_nodes: Iterable[Any],
    raw_relationships: Iterable[Any],
    page_numbers: Sequence[int] | None = None,
    source_text: str | None = None,
) -> tuple[list[ExtractedNode], list[ExtractedRelationship], list[dict[str, str]]]:
    """Validate, normalise and deduplicate one extraction result.

    Drops unsupported types, technical terms typed as people/places, empty
    names, self-loops and duplicate nodes/relationships. Every drop is
    recorded with a reason so callers can log it instead of hiding it.
    """

    pages = sorted({int(page) for page in (page_numbers or []) if page})
    source_key = entity_comparison_key(source_text) if source_text else None
    dropped: list[dict[str, str]] = []

    # Phase 1: accept one unambiguous type per name in this source unit.
    # Conflicting labels are omitted rather than guessed into one identity.
    resolved_types: dict[str, str] = {}
    spellings: dict[str, str] = {}
    ambiguous_names: set[str] = set()

    for raw_node in raw_nodes:
        node_id, raw_type, raw_name = _raw_node_fields(raw_node)
        name = normalize_name(raw_name) or normalize_name(node_id)
        entity_type = canonical_entity_type(raw_type)

        if not name:
            dropped.append({"name": node_id, "reason": "empty_name"})
            continue

        if (
            not is_usable_entity_name(
                name,
                allow_hyphenated_model_tokens=entity_type
                in {"PRODUCT", "TECHNOLOGY", "DOCUMENT"},
            )
            or _ANONYMIZED_ALIAS_PATTERN.fullmatch(name)
        ):
            dropped.append(
                {"name": name, "reason": "malformed_or_anonymized_name"}
            )
            continue

        if source_key is not None:
            name_key = entity_comparison_key(name)
            if not name_key or not re.search(
                rf"(?<!\w){re.escape(name_key)}(?!\w)",
                source_key,
            ):
                dropped.append(
                    {"name": name, "reason": "unsupported_source_evidence"}
                )
                continue

        if entity_type is None:
            dropped.append(
                {
                    "name": name,
                    "reason": f"unsupported_type:{raw_type or 'none'}",
                }
            )
            continue

        # A model error: a technology labelled as a person or a place.
        if is_technical_term(name) and entity_type in {"PERSON", "LOCATION"}:
            dropped.append(
                {
                    "name": name,
                    "reason": f"technical_term_as_{entity_type.lower()}",
                }
            )
            continue

        plausibility_type = {
            "PERSON": "PERSON",
            "ORGANIZATION": "ORG",
            "LOCATION": "GPE",
        }.get(entity_type)
        if (
            plausibility_type is not None
            and not is_plausible_entity(name, plausibility_type)
        ):
            dropped.append(
                {
                    "name": name,
                    "reason": f"implausible_type:{entity_type.lower()}",
                }
            )
            continue

        key = name.casefold()

        if key in ambiguous_names:
            dropped.append({"name": name, "reason": "ambiguous_type"})
            continue

        spellings.setdefault(key, name)

        current = resolved_types.get(key)

        if current is None:
            resolved_types[key] = entity_type
            continue

        if current != entity_type:
            ambiguous_names.add(key)
            resolved_types.pop(key, None)
            spellings.pop(key, None)
            dropped.append(
                {
                    "name": name,
                    "reason": "ambiguous_type:" + "/".join(
                        sorted({current, entity_type})
                    ),
                }
            )
        else:
            dropped.append({"name": name, "reason": "duplicate_node"})

    # Phase 2: build the deduplicated node set.
    nodes: dict[str, ExtractedNode] = {}

    for key, entity_type in resolved_types.items():
        name = spellings[key]
        entity_id = build_entity_id(entity_type, name)
        nodes[entity_id] = ExtractedNode(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            page_numbers=list(pages),
        )

    by_name = {
        normalize_name(node.name).casefold(): node.entity_id
        for node in nodes.values()
    }

    relationships: dict[tuple[str, str, str], ExtractedRelationship] = {}

    for raw_rel in raw_relationships:
        (
            source_raw,
            target_raw,
            raw_type,
            evidence,
        ) = _raw_relationship_fields(raw_rel)

        relationship_type = (raw_type or "").strip().upper().replace(" ", "_")

        if relationship_type not in ALLOWED_RELATIONSHIP_TYPES:
            dropped.append(
                {
                    "name": f"{source_raw}->{target_raw}",
                    "reason": f"unsupported_relationship:{raw_type or 'none'}",
                }
            )
            continue

        if (
            source_text is not None
            and not _relationship_evidence_is_supported(
                relationship_type,
                evidence,
                source_raw,
                target_raw,
                source_text,
            )
        ):
            dropped.append(
                {
                    "name": f"{source_raw}->{target_raw}",
                    "reason": "unsupported_relationship_evidence",
                }
            )
            continue

        source_id = by_name.get(normalize_name(source_raw).casefold())
        target_id = by_name.get(normalize_name(target_raw).casefold())

        if source_id is None or target_id is None:
            dropped.append(
                {
                    "name": f"{source_raw}->{target_raw}",
                    "reason": "relationship_missing_node",
                }
            )
            continue

        if source_id == target_id:
            dropped.append(
                {
                    "name": f"{source_raw}->{target_raw}",
                    "reason": "self_loop",
                }
            )
            continue

        key = (source_id, target_id, relationship_type)

        if key in relationships:
            dropped.append(
                {
                    "name": f"{source_raw}->{target_raw}",
                    "reason": "duplicate_relationship",
                }
            )
            continue

        relationships[key] = ExtractedRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            page_numbers=list(pages),
        )

    return list(nodes.values()), list(relationships.values()), dropped


# --------------------------------------------------
# Extraction entry points
# --------------------------------------------------


def _convert_documents(transformer, documents: Sequence[Document]) -> list[Any]:
    """Run Gemini graph extraction; the caller handles retry and sanitizing."""
    return transformer.convert_to_graph_documents(list(documents))


def _translate_graph_error(error: Exception) -> GraphExtractionError:
    """Return a useful graph error without exposing provider response text."""

    if isinstance(error, GraphExtractionError):
        return error
    from app.llm import LLMError, _translate_error

    if isinstance(error, LLMError):
        if error.__cause__ is not None:
            return _translate_graph_error(error.__cause__)
        return GraphExtractionError(str(error))
    if isinstance(error, (NotImplementedError, TimeoutError)):
        if isinstance(error, TimeoutError):
            return GraphExtractionError(
                "Gemini graph extraction request timed out."
            )
        return GraphExtractionError(
            "Gemini graph extraction does not support the required "
            "structured output operation."
        )
    if isinstance(error, ImportError):
        return GraphExtractionError(
            "Gemini graph extraction structured-output dependencies are "
            "unavailable."
        )

    message = str(error).casefold()
    if type(error).__name__ == "OutputParserException" or any(
        marker in message
        for marker in (
            "failed to parse structured output",
            "invalid json output",
            "could not parse",
        )
    ):
        return GraphExtractionError(
            "Gemini returned malformed structured graph output."
        )
    if isinstance(error, AttributeError) and any(
        marker in message
        for marker in ("with_structured_output", "structured output")
    ):
        return GraphExtractionError(
            "Gemini graph extraction does not support the required "
            "structured output operation."
        )

    translated = _translate_error(error, "Graph extraction")
    return GraphExtractionError(str(translated))


def extract_graph(
    content: str,
    *,
    document_id: str,
    transformer=None,
    pages: Sequence[dict[str, Any]] | None = None,
) -> ExtractionResult:
    """Extract a validated graph from one document.

    ``pages`` may carry page number, source-unit ID, and content fields. When
    given, extraction runs per content unit so provenance is preserved.
    Otherwise the whole text is extracted once without page-level provenance.
    """

    if not content or not content.strip():
        raise GraphExtractionError("Document content is empty; nothing to extract.")

    if transformer is None:
        try:
            active_transformer = build_transformer(provider="gemini")
        except Exception as error:
            raise _translate_graph_error(error) from error
    else:
        active_transformer = transformer

    per_page = [
        page
        for page in (pages or [])
        if str(page.get("content") or "").strip()
    ]

    if not per_page:
        per_page = [{"page_number": None, "content": content}]

    documents = [
        Document(
            page_content=str(page["content"])[:20000],
            metadata={
                "document_id": document_id,
                "page_number": page.get("page_number"),
                "source_unit_id": page.get("source_unit_id"),
            },
        )
        for page in per_page
    ]

    try:
        from app.llm import _retry_gemini

        graph_documents = _retry_gemini(
            lambda: _convert_documents(active_transformer, documents),
            "Graph extraction",
        )
    except Exception as error:
        raise _translate_graph_error(error) from error

    if len(graph_documents) != len(documents):
        raise GraphExtractionError(
            "The model returned a different number of graph documents than inputs."
        )

    merged_nodes: dict[str, ExtractedNode] = {}
    accumulated_relationships: list[Any] = []
    dropped: list[dict[str, str]] = []

    for document, graph_document in zip(documents, graph_documents):
        page_number = document.metadata.get("page_number")
        pages_here = [page_number] if page_number else []

        nodes, relationships, dropped_here = validate_and_clean_graph(
            getattr(graph_document, "nodes", []) or [],
            getattr(graph_document, "relationships", []) or [],
            page_numbers=pages_here,
            source_text=document.page_content,
        )

        dropped.extend(dropped_here)
        page_ref = (
            (
                f"{document_id}#page={page_number}"
                f"#unit={document.metadata['source_unit_id']}"
            )
            if page_number is not None and document.metadata.get("source_unit_id")
            else (
                f"{document_id}#page={page_number}"
                if page_number is not None
                else document_id
            )
        )

        for node in nodes:
            node.document_id = document_id
            node.source_refs = [page_ref]
            existing = merged_nodes.get(node.entity_id)

            if existing is None:
                merged_nodes[node.entity_id] = node
                continue

            for page in node.page_numbers:
                if page not in existing.page_numbers:
                    existing.page_numbers.append(page)
            if page_ref not in existing.source_refs:
                existing.source_refs.append(page_ref)

        for relationship in relationships:
            relationship.document_id = document_id
            relationship.source_refs = [page_ref]
            accumulated_relationships.append(
                (
                    relationship,
                    pages_here,
                )
            )

    # Re-validate relationships across page boundaries so cross-page
    # duplicates collapse and provenance is unioned.
    final_relationships: dict[tuple[str, str, str], ExtractedRelationship] = {}

    for relationship, pages_here in accumulated_relationships:
        key = (
            relationship.source_id,
            relationship.target_id,
            relationship.relationship_type,
        )

        existing = final_relationships.get(key)

        if existing is None:
            relationship.page_numbers = sorted({*relationship.page_numbers, *pages_here})
            final_relationships[key] = relationship
            continue

        for page in pages_here:
            if page not in existing.page_numbers:
                existing.page_numbers.append(page)
        for source_ref in relationship.source_refs:
            if source_ref not in existing.source_refs:
                existing.source_refs.append(source_ref)

        dropped.append(
            {
                "name": f"{relationship.source_id}->{relationship.target_id}",
                "reason": "duplicate_relationship",
            }
        )

    types_by_name: dict[str, set[str]] = {}
    for node in merged_nodes.values():
        types_by_name.setdefault(
            entity_comparison_key(node.name),
            set(),
        ).add(node.entity_type)

    ambiguous_keys = {
        name_key
        for name_key, entity_types in types_by_name.items()
        if len(entity_types) > 1
    }
    ambiguous_ids = {
        node.entity_id
        for node in merged_nodes.values()
        if entity_comparison_key(node.name) in ambiguous_keys
    }
    if ambiguous_ids:
        for node in merged_nodes.values():
            if node.entity_id in ambiguous_ids:
                dropped.append(
                    {"name": node.name, "reason": "ambiguous_type_across_units"}
                )
        merged_nodes = {
            entity_id: node
            for entity_id, node in merged_nodes.items()
            if entity_id not in ambiguous_ids
        }
        final_relationships = {
            key: relationship
            for key, relationship in final_relationships.items()
            if relationship.source_id not in ambiguous_ids
            and relationship.target_id not in ambiguous_ids
        }

    for node in merged_nodes.values():
        node.page_numbers = sorted(set(node.page_numbers))

    for relationship in final_relationships.values():
        relationship.page_numbers = sorted(set(relationship.page_numbers))

    logger.info(
        "LLM graph extraction for %s: %d nodes, %d relationships, %d dropped",
        document_id,
        len(merged_nodes),
        len(final_relationships),
        len(dropped),
    )

    return ExtractionResult(
        nodes=list(merged_nodes.values()),
        relationships=list(final_relationships.values()),
        dropped=dropped,
        backend="llm",
        provider="gemini",
    )


def extract_graph_legacy_spacy(content: str, *, document_id: str) -> ExtractionResult:
    """The original spaCy extraction, kept as an explicit opt-in path.

    Selected only with ``GRAPH_EXTRACTION_BACKEND=spacy`` or an explicit
    ``extractor="spacy"`` argument; never entered silently.
    """

    from app.context_engine import graph_service

    return graph_service._spacy_extraction(content, document_id=document_id)


# --------------------------------------------------
# Read-only audit
# --------------------------------------------------


def audit_suspicious_nodes(graph_service_module=None) -> list[dict[str, Any]]:
    """Report stored nodes whose type is outside the allowed schema.

    Strictly read-only: it issues MATCH/RETURN queries only and never writes.
    """

    module = graph_service_module

    if module is None:
        from app.context_engine import graph_service as module  # type: ignore

    findings: list[dict[str, Any]] = []

    driver = module.get_driver()

    type_query = """
    MATCH (entity:Entity)
    WHERE entity.entity_type IS NOT NULL
    RETURN entity.entity_type AS entity_type, count(entity) AS nodes
    ORDER BY nodes DESC
    """

    with driver.session(database=module.NEO4J_DATABASE) as session:
        type_rows = [dict(record) for record in session.run(type_query)]

    for row in type_rows:
        entity_type = row.get("entity_type")

        if entity_type in ALLOWED_ENTITY_TYPES:
            continue

        corrected = canonical_entity_type(entity_type)

        sample_query = """
        MATCH (entity:Entity {entity_type: $entity_type})
        RETURN entity.entity_id AS entity_id, entity.name AS name
        LIMIT 5
        """

        with driver.session(database=module.NEO4J_DATABASE) as session:
            samples = [
                dict(record)
                for record in session.run(sample_query, entity_type=entity_type)
            ]

        findings.append(
            {
                "entity_type": entity_type,
                "node_count": row.get("nodes", 0),
                "is_allowed": False,
                "proposed_correction": corrected,
                # Evidence only; no mutation is performed or scheduled.
                "evidence": {"sample_nodes": samples},
                "confidence": 0.9 if corrected else 0.4,
            }
        )

    return findings
