"""
Tests for the Medical Q&A Module.

Tests:
- Dataset loading
- Medical entity recognition
- Question retrieval
- Answer generation
- Similarity scores
- Empty questions
"""

import pytest

from modules.medical_qa import (
    MedicalQABot,
    extract_medical_entities,
    normalize_text,
)


@pytest.fixture(scope="module")
def medical_bot():
    """
    Create one MedicalQABot instance
    for all tests.
    """
    return MedicalQABot()


def test_normalize_text():
    text = "   What   is   Diabetes?   "

    result = normalize_text(text)

    assert result == "what is diabetes?"


def test_disease_entity():
    entities = extract_medical_entities(
        "What are the symptoms of diabetes?"
    )

    entity_names = [
        entity["text"]
        for entity in entities
    ]

    assert "diabetes" in entity_names


def test_symptom_entity():
    entities = extract_medical_entities(
        "I have fever and cough."
    )

    entity_names = [
        entity["text"]
        for entity in entities
    ]

    assert "fever" in entity_names
    assert "cough" in entity_names


def test_treatment_entity():
    entities = extract_medical_entities(
        "Is insulin used for diabetes?"
    )

    entity_names = [
        entity["text"]
        for entity in entities
    ]

    assert "insulin" in entity_names
    assert "diabetes" in entity_names


def test_dataset_loaded(medical_bot):
    assert medical_bot.data is not None
    assert len(medical_bot.data) > 0


def test_required_columns(medical_bot):
    required = {
        "question",
        "answer",
        "source",
        "category",
    }

    assert required.issubset(
        medical_bot.data.columns
    )


def test_search_returns_results(medical_bot):
    results = medical_bot.search(
        "What are symptoms of diabetes?"
    )

    assert isinstance(results, list)
    assert len(results) > 0


def test_search_result_structure(medical_bot):
    results = medical_bot.search(
        "What is diabetes?"
    )

    result = results[0]

    expected_keys = {
        "question",
        "answer",
        "source",
        "category",
        "similarity",
    }

    assert expected_keys.issubset(
        result.keys()
    )


def test_similarity_range(medical_bot):
    results = medical_bot.search(
        "What is asthma?"
    )

    for result in results:
        assert 0.0 <= result["similarity"] <= 1.0


def test_answer_question(medical_bot):
    result = medical_bot.answer_question(
        "What are the symptoms of diabetes?"
    )

    assert isinstance(result, dict)

    assert "answer" in result
    assert result["answer"]


def test_answer_contains_entities(medical_bot):
    result = medical_bot.answer_question(
        "What are symptoms of diabetes?"
    )

    assert "entities" in result

    entity_names = [
        entity["text"]
        for entity in result["entities"]
    ]

    assert "diabetes" in entity_names


def test_empty_question(medical_bot):
    results = medical_bot.search("")

    assert results == []


def test_whitespace_question(medical_bot):
    results = medical_bot.search("     ")

    assert results == []


def test_top_k(medical_bot):
    results = medical_bot.search(
        "diabetes",
        top_k=2,
    )

    assert len(results) <= 2