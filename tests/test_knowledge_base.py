"""
Tests for Dynamic Knowledge Base.

Tests:
- Empty database
- Adding knowledge
- Searching knowledge
- Duplicate prevention
- Persistence
- Chunking
- Statistics
- Dynamic updates
- Clearing the database
"""

import pytest

from modules.knowledge_base import DynamicKnowledgeBase


@pytest.fixture
def kb(tmp_path):
    """
    Create a temporary knowledge database for each test.

    This prevents tests from modifying the real
    vector_db/knowledge.json file.
    """

    test_file = tmp_path / "test_knowledge.json"

    database = DynamicKnowledgeBase(
        storage_file=test_file
    )

    return database


def test_empty_database(kb):
    stats = kb.get_stats()

    assert stats["documents"] == 0
    assert stats["chunks"] == 0
    assert stats["sources"] == 0


def test_add_document(kb):
    result = kb.add_document(
        text=(
            "Customers can request a refund "
            "within 30 days of purchase."
        ),
        source="Company Policy",
        title="Refund Policy",
    )

    assert result["success"] is True
    assert result["chunks_added"] >= 1


def test_document_saved(kb):
    kb.add_document(
        text="Technical support is available 24 hours a day.",
        source="Support Guide",
        title="Technical Support",
    )

    assert kb.storage_file.exists()


def test_search_document(kb):
    kb.add_document(
        text=(
            "Customers can request a refund "
            "within 30 days of purchase."
        ),
        source="Company Policy",
        title="Refund Policy",
    )

    results = kb.search(
        "How many days do I have to request a refund?"
    )

    assert len(results) > 0
    assert results[0]["title"] == "Refund Policy"


def test_search_result_structure(kb):
    kb.add_document(
        text="Premium customers receive priority customer support.",
        source="Customer Guide",
        title="Premium Support",
    )

    results = kb.search(
        "Do premium customers get priority support?"
    )

    assert len(results) > 0

    expected_keys = {
        "id",
        "title",
        "source",
        "content",
        "score",
        "added_at",
    }

    assert expected_keys.issubset(
        results[0].keys()
    )


def test_duplicate_document(kb):
    text = (
        "Customers can cancel their subscription "
        "at any time."
    )

    first = kb.add_document(
        text=text,
        source="Subscription Policy",
        title="Cancellation",
    )

    second = kb.add_document(
        text=text,
        source="Subscription Policy",
        title="Cancellation",
    )

    assert first["success"] is True
    assert second["success"] is False
    assert second["chunks_added"] == 0


def test_empty_document(kb):
    result = kb.add_document(
        text="",
        source="Test",
        title="Empty",
    )

    assert result["success"] is False
    assert result["chunks_added"] == 0


def test_whitespace_document(kb):
    result = kb.add_document(
        text="      ",
        source="Test",
        title="Whitespace",
    )

    assert result["success"] is False


def test_chunk_text_small_document():
    text = "This is a small document."

    chunks = DynamicKnowledgeBase.chunk_text(
        text,
        chunk_size=100,
    )

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_large_document():
    words = [
        f"word{i}"
        for i in range(300)
    ]

    text = " ".join(words)

    chunks = DynamicKnowledgeBase.chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1


def test_chunk_overlap():
    words = [
        f"word{i}"
        for i in range(200)
    ]

    text = " ".join(words)

    chunks = DynamicKnowledgeBase.chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    first_chunk = chunks[0].split()
    second_chunk = chunks[1].split()

    # Last 20 words of first chunk should
    # match first 20 words of second chunk.

    assert first_chunk[-20:] == second_chunk[:20]


def test_stats_after_add(kb):
    kb.add_document(
        text="Refund requests must be submitted within 30 days.",
        source="Company Policy",
        title="Refund Policy",
    )

    stats = kb.get_stats()

    assert stats["documents"] == 1
    assert stats["chunks"] >= 1
    assert stats["sources"] == 1
    assert stats["last_update"] is not None


def test_multiple_documents(kb):
    kb.add_document(
        text="Refunds are available within 30 days.",
        source="Refund Policy",
        title="Refunds",
    )

    kb.add_document(
        text="Technical support is available every day.",
        source="Support Guide",
        title="Support",
    )

    stats = kb.get_stats()

    assert stats["documents"] == 2
    assert stats["sources"] == 2


def test_persistence(tmp_path):
    storage = tmp_path / "persistent.json"

    first_database = DynamicKnowledgeBase(
        storage_file=storage
    )

    first_database.add_document(
        text=(
            "The premium plan includes "
            "priority technical support."
        ),
        source="Plans",
        title="Premium Plan",
    )

    # Create another instance using the same file.

    second_database = DynamicKnowledgeBase(
        storage_file=storage
    )

    stats = second_database.get_stats()

    assert stats["documents"] == 1

    results = second_database.search(
        "Does premium include priority support?"
    )

    assert len(results) > 0


def test_dynamic_update(kb):
    """
    Demonstrates the main Task 3 requirement.

    Search before update -> no information.

    Add new information.

    Search after update -> information available.
    """

    query = "What is the delivery time for premium orders?"

    before_update = kb.search(query)

    assert before_update == []

    kb.add_document(
        text=(
            "Premium orders are delivered "
            "within two business days."
        ),
        source="Delivery Policy",
        title="Premium Delivery",
    )

    after_update = kb.search(query)

    assert len(after_update) > 0

    assert (
        after_update[0]["title"]
        == "Premium Delivery"
    )


def test_irrelevant_search(kb):
    kb.add_document(
        text="Refunds are available within 30 days.",
        source="Policy",
        title="Refund Policy",
    )

    results = kb.search(
        "quantum astrophysics black holes",
        minimum_score=0.05,
    )

    assert results == []


def test_clear_database(kb):
    kb.add_document(
        text="This is temporary information.",
        source="Test",
        title="Temporary",
    )

    assert kb.get_stats()["documents"] == 1

    kb.clear()

    stats = kb.get_stats()

    assert stats["documents"] == 0
    assert stats["chunks"] == 0

    results = kb.search(
        "temporary information"
    )

    assert results == []