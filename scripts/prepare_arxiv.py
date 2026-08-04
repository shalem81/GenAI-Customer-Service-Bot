"""
arXiv Dataset Preparation
GenAI Customer Service Bot

Reads the Kaggle arXiv metadata JSON file and creates a
smaller Computer Science / Cybersecurity dataset.

Input:
    data/arxiv-metadata-oai-snapshot.json

Output:
    data/arxiv_cybersecurity.csv
"""

from pathlib import Path
import json
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "arxiv-metadata-oai-snapshot.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "arxiv_cybersecurity.csv"
)


# Maximum number of papers used by the internship project.
MAX_PAPERS = 10000


# arXiv category prefixes/areas useful for a cybersecurity-focused
# computer-science research assistant.
TARGET_CATEGORIES = {
    "cs.CR",  # Cryptography and Security
    "cs.NI",  # Networking and Internet Architecture
    "cs.LG",  # Machine Learning
    "cs.AI",  # Artificial Intelligence
}


SECURITY_KEYWORDS = {
    "security",
    "cybersecurity",
    "cyber security",
    "malware",
    "ransomware",
    "phishing",
    "intrusion",
    "vulnerability",
    "vulnerabilities",
    "attack",
    "attacks",
    "adversarial",
    "authentication",
    "authorization",
    "cryptography",
    "encryption",
    "privacy",
    "firewall",
    "botnet",
    "zero-day",
    "zero day",
    "threat detection",
    "network security",
    "access control",
    "digital forensics",
}


def clean_text(text):
    """Normalize whitespace in metadata text."""

    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()


def parse_authors(authors):
    """
    Convert arXiv authors field into readable text.

    Depending on the dataset version, authors may already
    be a string or may contain structured author data.
    """

    if not authors:
        return "Unknown"

    if isinstance(authors, str):
        return clean_text(authors)

    if isinstance(authors, list):

        names = []

        for author in authors:

            if isinstance(author, str):
                names.append(
                    clean_text(author)
                )

            elif isinstance(author, dict):

                first = clean_text(
                    author.get(
                        "first",
                        ""
                    )
                )

                middle = clean_text(
                    author.get(
                        "middle",
                        ""
                    )
                )

                last = clean_text(
                    author.get(
                        "last",
                        ""
                    )
                )

                name = " ".join(
                    value
                    for value in [
                        first,
                        middle,
                        last,
                    ]
                    if value
                )

                if name:
                    names.append(name)

        if names:
            return ", ".join(names)

    return clean_text(authors)


def has_target_category(categories):
    """Check whether the paper belongs to a target category."""

    categories = clean_text(categories)

    category_list = set(
        categories.split()
    )

    return bool(
        category_list
        & TARGET_CATEGORIES
    )


def contains_security_keyword(title, abstract):
    """Check title/abstract for cybersecurity terminology."""

    combined = (
        f"{title} {abstract}"
    ).lower()

    return any(
        keyword in combined
        for keyword in SECURITY_KEYWORDS
    )


def is_relevant_paper(record):
    """
    Keep papers relevant to the chosen domain.

    cs.CR papers are accepted directly.

    Papers from related CS categories are accepted when
    their title/abstract contains a security keyword.
    """

    categories = clean_text(
        record.get(
            "categories",
            ""
        )
    )

    title = clean_text(
        record.get(
            "title",
            ""
        )
    )

    abstract = clean_text(
        record.get(
            "abstract",
            ""
        )
    )

    category_list = set(
        categories.split()
    )

    if "cs.CR" in category_list:
        return True

    if has_target_category(categories):
        return contains_security_keyword(
            title,
            abstract,
        )

    return False


def prepare_arxiv_dataset():
    """Read arXiv metadata line-by-line and create the subset."""

    if not INPUT_FILE.exists():

        print("\nDataset file not found:")
        print(INPUT_FILE)

        print(
            "\nDownload the Kaggle arXiv dataset and place "
            "arxiv-metadata-oai-snapshot.json inside the data folder."
        )

        return

    rows = []

    processed = 0
    skipped_invalid = 0

    print()
    print("=" * 60)
    print("ARXIV CYBERSECURITY DATASET PREPARATION")
    print("=" * 60)

    print("\nInput:")
    print(INPUT_FILE)

    print(
        f"\nTarget maximum papers: {MAX_PAPERS}"
    )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            processed += 1

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                skipped_invalid += 1
                continue

            if not is_relevant_paper(record):
                continue

            paper_id = clean_text(
                record.get(
                    "id",
                    ""
                )
            )

            title = clean_text(
                record.get(
                    "title",
                    ""
                )
            )

            abstract = clean_text(
                record.get(
                    "abstract",
                    ""
                )
            )

            authors = parse_authors(
                record.get(
                    "authors",
                    ""
                )
            )

            categories = clean_text(
                record.get(
                    "categories",
                    ""
                )
            )

            if (
                not paper_id
                or not title
                or not abstract
            ):
                continue

            rows.append(
                {
                    "id": paper_id,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "categories": categories,
                }
            )

            if len(rows) >= MAX_PAPERS:
                break

            if processed % 100000 == 0:

                print(
                    f"Processed {processed:,} records | "
                    f"Selected {len(rows):,}"
                )

    if not rows:

        print(
            "\nNo matching papers were found."
        )

        return

    dataframe = pd.DataFrame(
        rows
    )

    dataframe = dataframe.drop_duplicates(
        subset=["id"]
    )

    dataframe = dataframe.dropna(
        subset=[
            "title",
            "abstract",
        ]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(
        "Records processed:",
        f"{processed:,}",
    )

    print(
        "Invalid records skipped:",
        skipped_invalid,
    )

    print(
        "Papers selected:",
        f"{len(dataframe):,}",
    )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("\nColumns:")
    print(
        list(dataframe.columns)
    )

    print("\nSample papers:")

    print(
        dataframe[
            [
                "id",
                "title",
                "categories",
            ]
        ].head()
    )


if __name__ == "__main__":
    prepare_arxiv_dataset()