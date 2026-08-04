"""
Tests for Research Expert Module.

Tests:
- Dataset loading
- Required columns
- Text cleaning
- Research paper search
- Result structure
- Relevance scores
- Paper retrieval
- Context generation
- Summarization
- Empty queries
"""

from pathlib import Path

import pandas as pd
import pytest

from modules.research_expert import ResearchExpert


@pytest.fixture
def sample_dataset(tmp_path):
    """
    Create a small temporary arXiv-style dataset.

    This means tests do not depend on the large
    Kaggle arXiv dataset.
    """

    file_path = (
        tmp_path
        / "test_arxiv.csv"
    )

    data = [
        {
            "id": "1001.0001",
            "title": (
                "Machine Learning for "
                "Malware Detection"
            ),
            "authors": "Alice Researcher",
            "abstract": (
                "This paper studies machine learning "
                "methods for detecting malicious "
                "software. Classification algorithms "
                "are evaluated for malware detection."
            ),
            "categories": "cs.CR cs.LG",
        },
        {
            "id": "1001.0002",
            "title": (
                "Network Intrusion Detection "
                "Using Deep Learning"
            ),
            "authors": "Bob Scientist",
            "abstract": (
                "This research presents a deep learning "
                "approach for detecting network "
                "intrusions and identifying malicious "
                "network traffic."
            ),
            "categories": "cs.CR cs.NI",
        },
        {
            "id": "1001.0003",
            "title": (
                "Phishing Website Detection"
            ),
            "authors": "Carol Analyst",
            "abstract": (
                "The study investigates phishing "
                "website detection using URL features "
                "and machine learning techniques."
            ),
            "categories": "cs.CR",
        },
        {
            "id": "1001.0004",
            "title": (
                "Privacy Preserving Authentication"
            ),
            "authors": "David Security",
            "abstract": (
                "This paper proposes an authentication "
                "system designed to improve privacy "
                "while maintaining secure access "
                "control."
            ),
            "categories": "cs.CR",
        },
    ]

    dataframe = pd.DataFrame(
        data
    )

    dataframe.to_csv(
        file_path,
        index=False,
        encoding="utf-8",
    )

    return file_path


@pytest.fixture
def expert(sample_dataset):
    """Create ResearchExpert using test data."""

    return ResearchExpert(
        dataset_path=sample_dataset
    )


def test_dataset_loaded(expert):

    assert expert.data is not None

    assert len(expert.data) == 4


def test_required_columns(expert):

    required = {
        "id",
        "title",
        "authors",
        "abstract",
        "categories",
    }

    assert required.issubset(
        expert.data.columns
    )


def test_clean_text():

    text = (
        "  Malware   detection\n"
        "using machine learning  "
    )

    result = (
        ResearchExpert.clean_text(
            text
        )
    )

    assert result == (
        "Malware detection "
        "using machine learning"
    )


def test_search_returns_results(expert):

    results = expert.search(
        "malware detection machine learning"
    )

    assert isinstance(
        results,
        list,
    )

    assert len(results) > 0


def test_malware_search(expert):

    results = expert.search(
        "malware detection"
    )

    assert len(results) > 0

    assert (
        "malware"
        in results[0]["title"].lower()
        or
        "malware"
        in results[0]["abstract"].lower()
    )


def test_intrusion_search(expert):

    results = expert.search(
        "network intrusion detection"
    )

    assert len(results) > 0

    assert (
        "intrusion"
        in results[0]["title"].lower()
        or
        "intrusion"
        in results[0]["abstract"].lower()
    )


def test_search_result_structure(expert):

    results = expert.search(
        "phishing detection"
    )

    assert len(results) > 0

    result = results[0]

    expected = {
        "id",
        "title",
        "authors",
        "abstract",
        "categories",
        "score",
    }

    assert expected.issubset(
        result.keys()
    )


def test_score_range(expert):

    results = expert.search(
        "authentication privacy"
    )

    assert len(results) > 0

    for result in results:

        assert (
            0.0
            <= result["score"]
            <= 1.0
        )


def test_top_k(expert):

    results = expert.search(
        "security detection",
        top_k=2,
    )

    assert len(results) <= 2


def test_empty_query(expert):

    results = expert.search("")

    assert results == []


def test_whitespace_query(expert):

    results = expert.search(
        "      "
    )

    assert results == []


def test_get_paper(expert):

    paper = expert.get_paper(
        "1001.0001"
    )

    assert paper is not None

    assert (
        paper["id"]
        == "1001.0001"
    )

    assert (
        "Malware"
        in paper["title"]
    )


def test_unknown_paper(expert):

    paper = expert.get_paper(
        "9999.9999"
    )

    assert paper is None


def test_create_context(expert):

    context = expert.create_context(
        "malware machine learning",
        top_k=2,
    )

    assert isinstance(
        context,
        str,
    )

    assert context

    assert "Title:" in context

    assert "Abstract:" in context


def test_simple_summary(expert):

    result = expert.simple_summary(
        "1001.0001"
    )

    assert result is not None

    assert (
        result["id"]
        == "1001.0001"
    )

    assert result["summary"]


def test_unknown_summary(expert):

    result = expert.simple_summary(
        "9999.9999"
    )

    assert result is None