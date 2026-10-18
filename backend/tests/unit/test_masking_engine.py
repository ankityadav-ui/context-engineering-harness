
from app.masking.masking_engine import mask_document_text


def test_mask_document_text_masks_sensitive_data():
    text = (
        "Rahul Sharma can be contacted at rahul@example.com. "
        "Rahul Sharma's phone number is 9876543210."
    )

    result = mask_document_text(
        text=text,
        document_id="test-document-001",
    )

    assert "Rahul Sharma" not in result.masked_text
    assert "rahul@example.com" not in result.masked_text
    assert "9876543210" not in result.masked_text
    assert "[EMAIL_1]" in result.masked_text
    assert "[PHONE_1]" in result.masked_text
    assert result.safety_passed is False

    person_placeholders = {
        entity.entity_type
        for entity in result.entities
        if entity.entity_type == "PERSON"
    }
    assert "PERSON" in person_placeholders


def test_ner_never_merges_a_heading_with_the_next_line():
    """A line break must end an entity span.

    spaCy reads PDF lines as one long sentence, so it used to label
    "Education\nExample University" as one ORG. Masking that span deleted
    the heading and left the resume as disconnected placeholders.
    """

    text = (
        "Alex Rivera\n"
        "Education\n"
        "Example University, B.Tech in Computer Science\n"
        "Skills\n"
        "Python, SQL, FastAPI\n"
        "Project\n"
        "Document search application using RAG."
    )

    result = mask_document_text(text=text, document_id="resume-1")

    # Headings stay readable.
    assert "Education" in result.masked_text
    assert "Skills" in result.masked_text
    assert "Project" in result.masked_text

    # The name and organization are still masked.
    assert "Alex Rivera" not in result.masked_text
    assert "Example University" not in result.masked_text
    assert "[PERSON_1]" in result.masked_text
    assert "[ORG_1]" in result.masked_text

    # Every replacement span still matches the original text.
    for entity in result.entities:
        assert text[entity.start:entity.end] == entity.text


def test_document_header_name_is_masked_when_ner_misses_it(monkeypatch):
    """Real resumes leak the name when spaCy does not know it.

    ``en_core_web_sm`` returns no entity at all for several real names
    (this is exactly what happened to a production resume), so the
    masked upload used to index and send the full name. NER is stubbed
    out here so the test pins the header rule itself.
    """

    import app.masking.masking_engine as masking_engine

    monkeypatch.setattr(
        masking_engine, "detect_entities", lambda _text: []
    )

    text = (
        "Priya Nair\n"
        "priya.nair@example.com\n"
        "Education\n"
        "Example University\n"
        "Skills\n"
        "Python, FastAPI"
    )

    result = mask_document_text(
        text=text,
        document_id="resume-2",
        include_header_name=True,
    )

    assert "Priya Nair" not in result.masked_text
    assert "priya.nair@example.com" not in result.masked_text
    assert "[PERSON_1]" in result.masked_text
    # Structure still survives.
    assert "Education" in result.masked_text
    assert "Skills" in result.masked_text


def test_header_rule_does_not_mask_a_section_heading(monkeypatch):
    import app.masking.masking_engine as masking_engine

    monkeypatch.setattr(
        masking_engine, "detect_entities", lambda _text: []
    )

    text = "Work Experience\nSenior Backend Engineer"

    result = mask_document_text(
        text=text,
        document_id="resume-3",
        include_header_name=True,
    )

    assert "Work Experience" in result.masked_text


def test_header_rule_is_opt_in(monkeypatch):
    # POST /masking/text and one-off calls keep the raw behaviour: the
    # layout rule only runs for the first text unit of a masked upload.
    import app.masking.masking_engine as masking_engine

    monkeypatch.setattr(
        masking_engine, "detect_entities", lambda _text: []
    )

    text = "Priya Nair\nEducation"

    result = mask_document_text(text=text, document_id="resume-4")

    assert "Priya Nair" in result.masked_text
