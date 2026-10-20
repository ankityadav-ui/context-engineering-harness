"""Entity extraction using the shared Gemini model for knowledge graph ingestion.

This module provides entity extraction from document content using the
project's configured Gemini LLM. It uses a structured prompt with
strict rules to ensure correct entity type classification and prevent
misclassification of technical terms.

Entity type schema: PERSON, ORG, GPE, LOC, TECH, PRODUCT, DOC
"""

from __future__ import annotations

from typing import Any


# Technical terms and concepts that should never be classified as people,
# organizations, or locations. These are common in documentation, code,
# and technical writing.
TECHNICAL_TERMS = frozenset({
    "ai",
    "artificial intelligence",
    "rag",
    "retrieval augmented generation",
    "sql",
    "structured query language",
    "fastapi",
    "flask",
    "django",
    "express",
    "neo4j",
    "postgresql",
    "mysql",
    "sqlite",
    "python",
    "javascript",
    "typescript",
    "java",
    "c++",
    "c#",
    "flutter",
    "react",
    "angular",
    "vue",
    "docker",
    "kubernetes",
    "aws",
    "gcp",
    "azure",
    "api",
    "rest",
    "graphql",
    "grpc",
    "microservice",
    "monolith",
    "legacy",
    "deprecated",
})

# Entity type schema supported by the knowledge graph.
# These correspond to the Neo4j entity labels used in the project.
ENTITY_TYPES = frozenset({"PERSON", "ORG", "GPE", "LOC", "TECH", "PRODUCT", "DOC"})


def validate_entity_type(entity_type: str) -> str | None:
    """Validate and normalize an entity type.

    Returns the normalized type if valid, or None if the type is not supported.
    """
    type_upper = entity_type.upper().strip()
    if type_upper in ENTITY_TYPES:
        return type_upper
    return None


def is_technical_term(name: str) -> bool:
    """Check if a name is a technical term that should not be classified as an entity type.

    Args:
        name: The entity name to check.

    Returns:
        True if the name is a technical term, False otherwise.
    """
    name_lower = name.lower().strip()
    # Check exact match
    if name_lower in TECHNICAL_TERMS:
        return True
    # Check if any technical term is a substring
    for term in TECHNICAL_TERMS:
        if term in name_lower:
            return True
    return False


def post_process_entities(raw_output: str) -> list[dict]:
    """Post-process raw LLM output to extract validated entities.

    Returns a list of {"name": str, "entity_type": str} dicts.

    The function:
    1. Tries to find a JSON list in the output
    2. If no JSON found, tries to parse line by line with name:type pattern
    3. Validates each entity's type against the controlled schema
    4. Filters out technical terms
    5. Deduplicates by normalized name
    6. Limits to 20 entities max
    """
    import json
    import re

    entities: list[dict] = []

    # Try to find JSON list in the output
    json_match = re.search(r"\[.*\]", raw_output, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "name" in item and "entity_type" in item:
                        validated_type = validate_entity_type(item["entity_type"])
                        if validated_type and not is_technical_term(str(item["name"])):
                            entities.append(
                                {
                                    "name": str(item["name"]),
                                    "entity_type": validated_type,
                                }
                            )
        except (json.JSONDecodeError, ValueError):
            pass

    # If no JSON found, try to parse line by line
    if not entities:
        lines = raw_output.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Try to find name:type pattern
            pattern = r"([^:]+):\s*([A-Z]+)"
            match = re.match(pattern, line)
            if match:
                name = match.group(1).strip()
                etype = match.group(2).strip()
                validated_type = validate_entity_type(etype)
                if validated_type and not is_technical_term(name):
                    entities.append({"name": name, "entity_type": validated_type})

    # Deduplicate by normalized name
    seen: set[str] = set()
    deduped: list[dict] = []
    for entity in entities:
        norm_name = entity["name"].lower().replace("_", "").replace("-", "")
        if norm_name not in seen:
            seen.add(norm_name)
            deduped.append(entity)

    # Limit to max_entities
    return deduped[:20]


def extract_entities_from_content(
    content: str, max_entities: int = 20, llm_client: Any | None = None
) -> list[dict]:
    """Extract entities from document content using the Gemini LLM.

    Uses a structured prompt with strict rules to extract entities in a
    controlled schema. Technical terms are prevented from being misclassified
    as people, organizations, or locations.

    The prompt emphasizes:
    1. Only extract entities of these types: PERSON, ORG, GPE, LOC, TECH, PRODUCT, DOC
    2. Technical terms must NOT be classified as PERSON, ORG, GPE, or LOC
    3. If evidence is insufficient to determine a type, DO NOT assign a type
    4. Each entity must have a clear, descriptive name
    5. Return entities as a JSON list with "name" and "entity_type" fields only

    Args:
        content: The document content to extract entities from
        max_entities: Maximum number of entities to extract
        llm_client: Optional pre-configured LangChain chat model (for testing)

    Returns:
        List of {"name": str, "entity_type": str} dicts, limited to max_entities
    """
    from app.llm import generate_response

    prompt = build_entity_extraction_prompt(content, max_entities)
    if llm_client is None:
        raw_output = generate_response(prompt=prompt)
    else:
        response = llm_client.invoke(prompt)
        raw_output = getattr(response, "content", "")
        if not isinstance(raw_output, str):
            raise TypeError("The chat model returned non-text entity output.")

    entities = post_process_entities(raw_output)
    return entities[:max(0, max_entities)]


def build_entity_extraction_prompt(content: str, max_entities: int = 20) -> str:
    """Build a structured prompt for Gemini entity extraction.

    Returns a prompt that Gemini can follow to extract entities
    in a controlled schema. The prompt emphasizes correct classification
    of technical terms.

    The prompt uses a strict format and emphasizes:
    1. Only extract entities of these types: PERSON, ORG, GPE, LOC, TECH, PRODUCT, DOC
    2. Technical terms must NOT be classified as PERSON, ORG, GPE, or LOC
    3. If evidence is insufficient to determine a type, DO NOT assign a type
    4. Each entity must have a clear, descriptive name
    5. Return entities as a JSON list with "name" and "entity_type" fields only

    Args:
        content: The document content to extract entities from
        max_entities: Maximum number of entities to extract

    Returns:
        A structured prompt string for the LLM
    """
    # Take a sample of the content (first 3000 chars to keep prompts reasonable)
    sample = content[:3000] if content else ""

    # Truncate technical terms list to most relevant ones
    terms_list = ", ".join(sorted(list(TECHNICAL_TERMS)))[:500]

    prompt = (
        "You are an entity extraction expert. Extract entities from the following document content.\n\n"
        "STRICT RULES:\n"
        " 1. Only extract entities of these types: " + str(ENTITY_TYPES) + "\n"
        " 2. Technical terms like " + terms_list + " must NOT be classified as PERSON, ORG, GPE, or LOC\n"
        " 3. If evidence is insufficient to determine a type, DO NOT assign a type (drop the entity)\n"
        " 4. Extract no more than " + str(max_entities) + " entities\n"
        " 5. Each entity must have a clear, descriptive name\n"
        " 6. Return entities as a JSON list with \"name\" and \"entity_type\" fields only\n\n"
        "Document content:\n\n"
        + sample
        + "\n"
    )

    return prompt
