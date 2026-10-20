"""Graph-specific entity validation.

This module is deliberately separate from ``app.masking.ner_service``.
Masking must keep recall high (a missed entity is a privacy leak), so the
shared NER service is left untouched. The knowledge graph needs the
opposite trade-off: it should store fewer, better-typed nodes.

Two independent checks are applied here:

1. **Name quality** (:func:`is_usable_entity_name`) - is this span a
   plausible entity name at all? Rejects masking placeholders, markup
   remnants, numbers merged into names, OCR debris, and merged headings.
2. **Type plausibility** (:func:`is_plausible_entity`) - is the label
   spaCy assigned compatible with the name? Uses generic linguistic
   evidence (name morphology, particles, concept vocabulary, legal and
   organizational designators), not a list of document-specific
   exceptions.

:func:`resolve_entity_types` adds deterministic document-level evidence for
the explicit legacy spaCy path. The LLM graph path uses the shared name and
type-plausibility checks but drops names with conflicting types rather than
choosing a label without enough context.

None of these rules inspect or modify masking output.
"""

import re

# Entity labels the graph schema supports.
ENTITY_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "LAW",
    "EVENT",
}

# Label priority used when one name is labelled inconsistently inside a
# document. Places win over organizations, organizations over people.
# This only merges contradictory labels for the *same* name; it never
# changes the type of a name that was labelled consistently.
TYPE_PRIORITY = {
    "GPE": 5,
    "LOC": 5,
    "ORG": 4,
    "PERSON": 3,
    "LAW": 2,
    "EVENT": 1,
}

# Generic English concept vocabulary. These words name ideas, technologies,
# and disciplines rather than people, organizations, or places, so they are
# implausible inside a person name and cannot be a stand-alone entity name.
# This is domain vocabulary, not a list of document-specific mentions.
CONCEPT_TOKENS = frozenset({
    "analysis", "analytics", "architecture", "augmentation", "biology",
    "chemistry", "cloud", "computing", "data", "design", "development",
    "economics", "education", "engineering", "finance", "graph",
    "healthcare", "intelligence", "law", "learning", "management",
    "marketing", "mathematics", "model", "network", "operations",
    "physics", "privacy", "processing", "retrieval", "research",
    "science", "security", "statistics", "system", "testing", "topology",
    # Generic document/technical artifact words (PDF, NER, OCR, ...).
    # They describe the medium itself, not an entity, so a stand-alone
    # mention of one is never a person, organization, or place. This is
    # a bounded, generic category of document-structure vocabulary -
    # not a growing list of document-specific exceptions.
    "api", "csv", "faq", "html", "http", "https", "json",
    "ner", "ocr", "otp", "pdf", "pdfs", "png", "url", "urls",
    "uri", "urn", "uuid", "xml",
})

# Productive nominalizing suffixes: a two-plus-token PERSON label ending in
# one of these is almost always a misclassified concept ("Nachine
# Learning", "Data Topology"). Real surnames with these endings are rare,
# and one-token names are exempt so short surnames survive.
NON_PERSON_SUFFIXES = (
    "ology",
    "tion",
    "sion",
    "ics",
    "ism",
    "ment",
    "ness",
    "ance",
    "ence",
    "ship",
    "hood",
)

# Words that may legitimately appear inside a person name.
NAME_PARTICLES = frozenset({
    "al", "bin", "da", "de", "del", "der", "di", "dos", "du", "ibn",
    "la", "le", "st.", "ten", "ter", "van", "von", "y",
})

# Honorifics and generational suffixes that are valid name tokens.
NAME_AFFIXES = frozenset({
    "dr.", "dr", "esq.", "ii", "iii", "iv", "jr.", "jr", "md", "mr.",
    "mr", "mrs.", "mrs", "ms.", "ms", "phd", "prof.", "prof", "sr.",
    "sr",
})

# Tokens that indicate an organization rather than a person.
ORG_DESIGNATORS = frozenset({
    "agency", "association", "bank", "bureau", "co", "co.", "college",
    "committee", "commission", "company", "corporation", "corp", "corp.",
    "council", "court", "department", "foundation", "gmbh", "group",
    "hospital", "inc", "inc.", "institute", "institution", "laboratory",
    "labs", "llc", "llp", "ltd", "ltd.", "ministry", "office",
    "organization", "organisation", "partners", "plc", "school",
    "services", "solutions", "technologies", "university",
})

# Synthetic-data markers that should not become graph organizations.
SYNTHETIC_MARKERS = frozenset({
    "example", "fictional", "sample", "synthetic",
})

# A LAW entity must end in a legal term to be stored.
LEGAL_TERMS = ("act", "code", "court", "regulation", "section", "statute")

_PLACEHOLDER_PATTERN = re.compile(
    r"^(?:PERSON|ORG|GPE|LOC|LAW|EVENT)_\d+",
    flags=re.IGNORECASE,
)

_STANDALONE_LABELS = frozenset(
    label.casefold() for label in ENTITY_LABELS
)

_ALLOWED_NAME_CHARACTERS = re.compile(
    r"[\w\s.'’&(),-]+",
    flags=re.UNICODE,
)
_HYPHENATED_MODEL_TOKEN = re.compile(
    r"[A-Za-z][A-Za-z0-9]*-\d+[A-Za-z]?"
)

# Characters left behind by masking or OCR that never belong in a name.
_REJECTED_CHARACTERS = "[]<>\\®�"

MAX_NAME_TOKENS = 6


def normalize_entity_name(name: str) -> str:
    """Collapse whitespace in an entity span."""

    return " ".join(str(name or "").split()).strip()


# Punctuation that never distinguishes two spellings of one entity:
# separators, terminal punctuation, and wrapper brackets/quotes.
# Apostrophes are folded too ("O'Brien" == "OBrien"), which is safe
# because both spellings refer to the same name.
_COMPARISON_PUNCTUATION = re.compile(r"[.,;:!?\"'\'`()\[\]{}]")


def entity_comparison_key(name: str) -> str:
    """Deterministic identity key used to merge entity spellings.

    NFKC-normalize, casefold, fold harmless punctuation to spaces, and
    collapse whitespace. Hyphens are preserved so genuinely different
    names ("Smith-Jones" vs "Smith Jones") are not merged. Used both by
    graph ingestion (stable node ids) and by the cleanup report.
    """

    try:
        import unicodedata

        cleaned = unicodedata.normalize("NFKC", str(name or ""))
    except (TypeError, ValueError):
        cleaned = str(name or "")

    cleaned = cleaned.casefold()
    cleaned = _COMPARISON_PUNCTUATION.sub(" ", cleaned)
    cleaned = " ".join(cleaned.split())

    return cleaned


def entity_name_key(name: str) -> str:
    """Return the case-insensitive identity used to merge entity names."""

    return entity_comparison_key(name)


def is_usable_entity_name(
    name: str,
    *,
    allow_hyphenated_model_tokens: bool = False,
) -> bool:
    """Return True when the span is a plausible entity name.

    This is name-quality validation only; it says nothing about whether
    the assigned entity type makes sense.
    """

    cleaned = normalize_entity_name(name)

    if len(cleaned) < 2:
        return False

    # Masking placeholders such as PERSON_15 or ORG_4.
    if _PLACEHOLDER_PATTERN.match(cleaned):
        return False

    # Stand-alone type labels leaked from masking output.
    if cleaned.casefold() in _STANDALONE_LABELS:
        return False

    # Markup remnants, backslashes, replacement symbols.
    if any(character in cleaned for character in _REJECTED_CHARACTERS):
        return False

    # Ordinary names and common punctuation only.
    if not _ALLOWED_NAME_CHARACTERS.fullmatch(cleaned):
        return False

    if not any(character.isalpha() for character in cleaned):
        return False

    tokens = cleaned.split()

    if len(tokens) > MAX_NAME_TOKENS:
        return False

    # A page or list number merged into the name.
    if tokens[0][0].isdigit():
        return False

    # Too many structural separators: a merged heading, not a name.
    if sum(cleaned.count(character) for character in "/:;|") > 2:
        return False

    for token in tokens:
        # Connectors inside organization names are not words.
        if token in {"&", "+"}:
            continue

        token_name = token.strip(".,;:!?()")
        has_digit = any(character.isdigit() for character in token_name)
        has_letter = any(character.isalpha() for character in token)

        # Tokens mixing digits and letters, e.g. "0nd".
        if has_digit and has_letter:
            if not (
                allow_hyphenated_model_tokens
                and _HYPHENATED_MODEL_TOKEN.fullmatch(token_name)
            ):
                return False

        if not has_letter:
            return False

    return True


def looks_like_person_name(name: str) -> bool:
    """Return True when the name has a plausible personal-name shape."""

    cleaned = normalize_entity_name(name)
    tokens = [
        token.strip(".,:;()")
        for token in cleaned.split()
    ]
    tokens = [token for token in tokens if token]

    if not tokens or len(tokens) > 4:
        return False

    # Concepts and disciplines are not people.
    if any(token.casefold() in CONCEPT_TOKENS for token in tokens):
        return False

    # Organizational designators are not people.
    if any(token.casefold() in ORG_DESIGNATORS for token in tokens):
        return False

    for index, token in enumerate(tokens):
        folded = token.casefold()

        if folded in NAME_PARTICLES or folded in NAME_AFFIXES:
            continue

        if token[0].isupper():
            continue

        # A lowercase token is only acceptable as a particle (above).
        return False

    if len(tokens) == 1:
        # A single all-caps token is an initialism/acronym shape (PDF,
        # NER, API), not a personal name. Genuine one-word person names
        # are written with at least one lowercase letter; OCR'd names
        # in ALL CAPS headings are dropped rather than stored with a
        # wrong PERSON type.
        if tokens[0].isupper():
            return False

        # Otherwise a single token is a person only when it is a
        # capitalized word, which the loop above already guaranteed.
        return True

    last = tokens[-1].casefold()

    if last in NAME_AFFIXES:
        return True

    if last.endswith(NON_PERSON_SUFFIXES):
        return False

    # Present participles / gerunds: "Learning", "Engineering".
    if last.endswith("ing") and len(last) >= 6:
        return False

    return True


def is_plausible_entity(name: str, entity_type: str) -> bool:
    """Return True when ``name`` is compatible with ``entity_type``.

    Conservative by design: when the evidence is insufficient the entity
    is dropped from the graph rather than stored with a wrong type.
    """

    cleaned = normalize_entity_name(name)

    if entity_type not in ENTITY_LABELS:
        return False

    if not is_usable_entity_name(cleaned):
        return False

    key = entity_name_key(cleaned)
    tokens = [token.casefold().strip(".,:;()") for token in cleaned.split()]

    if entity_type == "PERSON":
        return looks_like_person_name(cleaned)

    if entity_type == "ORG":
        if any(token in SYNTHETIC_MARKERS for token in tokens):
            return False

        # A single concept word is not an organization ("Topology").
        if key in CONCEPT_TOKENS:
            return False

        # Multi-word concept phrases without an organizational designator
        # ("Machine Learning") are not organizations either.
        if (
            any(token in CONCEPT_TOKENS for token in tokens)
            and not any(token in ORG_DESIGNATORS for token in tokens)
        ):
            return False

        return True

    if entity_type in {"GPE", "LOC"}:
        return key not in CONCEPT_TOKENS

    if entity_type == "LAW":
        return any(key.endswith(term) for term in LEGAL_TERMS)

    # EVENT: name quality is all the evidence available.
    return True


def resolve_entity_types(
    candidates: list[tuple[str, str]],
) -> dict[str, dict]:
    """Resolve one entity type per name across a whole document.

    ``candidates`` is an ordered list of ``(name, label)`` pairs. The
    return value maps ``entity_name_key`` to ``{"name", "entity_type",
    "observed_types"}``, keeping the first spelling seen and the highest
    priority label observed.
    """

    resolved: dict[str, dict] = {}

    for name, label in candidates:
        cleaned = normalize_entity_name(name)

        if label not in ENTITY_LABELS:
            continue

        key = entity_name_key(cleaned)

        if not key:
            continue

        entry = resolved.get(key)

        if entry is None:
            resolved[key] = {
                "name": cleaned,
                "entity_type": label,
                "observed_types": {label},
            }
            continue

        entry["observed_types"].add(label)

        if TYPE_PRIORITY.get(label, 0) > TYPE_PRIORITY.get(
            entry["entity_type"],
            0,
        ):
            entry["entity_type"] = label

    # Deterministic output order: first appearance in the document.
    return resolved
