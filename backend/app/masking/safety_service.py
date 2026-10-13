
# Import regex support for detecting obvious missed patterns.
import re


# Define patterns that should not remain visible in masked text.
SAFETY_PATTERNS = {
    # Detect email addresses that may have escaped masking.
    "EMAIL": re.compile(
        r"\b[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),

    # Detect common 10-digit Indian mobile numbers.
    "PHONE": re.compile(
        r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"
    ),

    # Detect PAN-like identifiers.
    "PAN": re.compile(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
    ),
}


def validate_masked_text(masked_text: str) -> dict:
    # Treat empty input as unsafe.
    if not masked_text or not masked_text.strip():
        return {
            "passed": False,
            "reasons": ["Masked text is empty."],
        }

    # Collect categories still detected in the output.
    detected_categories = []

    # Scan the masked text with every safety pattern.
    for category, pattern in SAFETY_PATTERNS.items():
        if pattern.search(masked_text):
            detected_categories.append(category)

    # Pass only if none of the configured patterns remain.
    return {
        "passed": not detected_categories,
        "reasons": [
            f"Possible unmasked {category} detected."
            for category in detected_categories
        ],
    }
