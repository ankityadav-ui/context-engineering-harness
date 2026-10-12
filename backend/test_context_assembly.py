
# Import the formatter that we want to test.
from app.context_engine.engine import format_context


# Create sample data without calling PostgreSQL, Pinecone, or Neo4j.
def main():
    # Simulate the combined result returned by the Context Engine.
    sample_context = {
        "query": "What does the document say about the heart?",
        "conversation": {
            "summary": "The user previously discussed heart anatomy.",
            "recent_messages": [
                {
                    "role": "user",
                    "content": "Explain the heart diagram.",
                },
                {
                    "role": "assistant",
                    "content": "The heart pumps blood.",
                },
            ],
        },
        "memory": [
            {
                "memory_type": "preference",
                "content": "The user prefers simple explanations.",
            }
        ],
        "rag": [
            {
                "content_type": "text",
                "content": "The heart circulates blood.",
                "document_id": "test-doc-001",
                "page_number": 1,
            },
            {
                "content_type": "table",
                "content": "Chamber | Function: Left ventricle | Pumps blood",
                "document_id": "test-doc-001",
                "page_number": 2,
            },
        ],
        "knowledge_graph": [
            {
                "source": "Heart",
                "relationship": "PART_OF",
                "target": "Circulatory System",
            }
        ],
        "source_errors": [],
    }

    # Format the sample into the single context string for the LLM.
    formatted = format_context(sample_context)

    # Verify that every context source appears in the output.
    expected_sections = [
        "CURRENT QUESTION:",
        "CONVERSATION SUMMARY:",
        "RECENT CONVERSATION:",
        "RELEVANT LONG-TERM MEMORIES:",
        "RELEVANT DOCUMENT EVIDENCE:",
        "KNOWLEDGE GRAPH RELATIONSHIPS:",
    ]

    # Stop the test if any expected section is missing.
    for section in expected_sections:
        assert section in formatted, (
            f"Missing expected section: {section}"
        )

    # Print the formatted context so we can inspect the result.
    print(formatted)

    # Confirm that all context sections were included.
    print("\nPASS: Context formatting test succeeded.")


# Run the test when this file is executed directly.
if __name__ == "__main__":
    main()
