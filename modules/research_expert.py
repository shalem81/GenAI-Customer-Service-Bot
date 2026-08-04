"""
Research Expert Module
GenAI Customer Service Bot

Task 4:
Domain Expert Chatbot using arXiv research papers.

Features:
- Load processed arXiv papers
- Search research papers
- TF-IDF retrieval
- Retrieve relevant abstracts
- Paper information
- Context generation for an LLM
"""

from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_DATASET = (
    BASE_DIR
    / "data"
    / "arxiv_cybersecurity.csv"
)


class ResearchExpert:
    """Research-paper retrieval engine."""

    def __init__(self, dataset_path=None):

        self.dataset_path = Path(
            dataset_path or DEFAULT_DATASET
        )

        self.data = None
        self.vectorizer = None
        self.paper_matrix = None

        self.load_dataset()

    @staticmethod
    def clean_text(text):
        """Normalize text."""

        if not isinstance(text, str):
            return ""

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    def load_dataset(self):
        """Load processed arXiv dataset."""

        if not self.dataset_path.exists():

            raise FileNotFoundError(
                "\nResearch dataset not found:\n"
                f"{self.dataset_path}\n\n"
                "Create data/arxiv_cybersecurity.csv "
                "before starting the Research Expert."
            )

        self.data = pd.read_csv(
            self.dataset_path
        )

        required_columns = {
            "id",
            "title",
            "authors",
            "abstract",
            "categories",
        }

        missing = (
            required_columns
            - set(self.data.columns)
        )

        if missing:

            raise ValueError(
                "arXiv dataset is missing columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

        self.data = self.data.dropna(
            subset=[
                "title",
                "abstract",
            ]
        ).copy()

        self.data["title"] = (
            self.data["title"]
            .astype(str)
            .apply(self.clean_text)
        )

        self.data["abstract"] = (
            self.data["abstract"]
            .astype(str)
            .apply(self.clean_text)
        )

        self.data["authors"] = (
            self.data["authors"]
            .fillna("Unknown")
            .astype(str)
            .apply(self.clean_text)
        )

        self.data["categories"] = (
            self.data["categories"]
            .fillna("")
            .astype(str)
            .apply(self.clean_text)
        )

        # Combine title + abstract for retrieval.
        self.data["search_text"] = (
            self.data["title"]
            + " "
            + self.data["abstract"]
        )

        self.build_index()

    def build_index(self):
        """Create TF-IDF research-paper index."""

        if self.data is None or self.data.empty:

            self.vectorizer = None
            self.paper_matrix = None

            return

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
        )

        self.paper_matrix = (
            self.vectorizer.fit_transform(
                self.data["search_text"]
            )
        )

    def search(
        self,
        query,
        top_k=5,
        minimum_score=0.01,
    ):
        """Search relevant research papers."""

        query = self.clean_text(
            query
        )

        if not query:

            return []

        if (
            self.vectorizer is None
            or self.paper_matrix is None
        ):

            return []

        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        scores = cosine_similarity(
            query_vector,
            self.paper_matrix,
        ).flatten()

        top_indices = (
            scores.argsort()[::-1][:top_k]
        )

        results = []

        for index in top_indices:

            score = float(
                scores[index]
            )

            if score < minimum_score:
                continue

            row = self.data.iloc[
                index
            ]

            results.append(
                {
                    "id": str(
                        row["id"]
                    ),
                    "title": row[
                        "title"
                    ],
                    "authors": row[
                        "authors"
                    ],
                    "abstract": row[
                        "abstract"
                    ],
                    "categories": row[
                        "categories"
                    ],
                    "score": round(
                        score,
                        4,
                    ),
                }
            )

        return results

    def get_paper(self, paper_id):
        """Retrieve one paper by arXiv ID."""

        paper_id = str(
            paper_id
        ).strip()

        matches = self.data[
            self.data["id"]
            .astype(str)
            == paper_id
        ]

        if matches.empty:
            return None

        row = matches.iloc[0]

        return {
            "id": str(
                row["id"]
            ),
            "title": row[
                "title"
            ],
            "authors": row[
                "authors"
            ],
            "abstract": row[
                "abstract"
            ],
            "categories": row[
                "categories"
            ],
        }

    def create_context(
        self,
        query,
        top_k=3,
    ):
        """
        Create research context that can later
        be passed to Gemini/another LLM.
        """

        papers = self.search(
            query=query,
            top_k=top_k,
        )

        if not papers:

            return ""

        context_parts = []

        for number, paper in enumerate(
            papers,
            start=1,
        ):

            context = (
                f"Paper {number}\n"
                f"Title: {paper['title']}\n"
                f"Authors: {paper['authors']}\n"
                f"Categories: {paper['categories']}\n"
                f"Abstract: {paper['abstract']}"
            )

            context_parts.append(
                context
            )

        return "\n\n".join(
            context_parts
        )

    def simple_summary(
        self,
        paper_id,
        max_sentences=3,
    ):
        """
        Create a basic extractive summary.

        This is temporary.

        Later Gemini/open-source LLM will
        generate better explanations.
        """

        paper = self.get_paper(
            paper_id
        )

        if not paper:

            return None

        abstract = paper[
            "abstract"
        ]

        sentences = re.split(
            r"(?<=[.!?])\s+",
            abstract,
        )

        selected = sentences[
            :max_sentences
        ]

        summary = " ".join(
            selected
        )

        return {
            "id": paper["id"],
            "title": paper["title"],
            "summary": summary,
        }


if __name__ == "__main__":

    try:

        expert = ResearchExpert()

        print()
        print("=" * 60)
        print("ARXIV CYBERSECURITY RESEARCH EXPERT")
        print("=" * 60)

        print(
            "\nPapers loaded:",
            len(expert.data),
        )

        query = input(
            "\nEnter research topic: "
        )

        results = expert.search(
            query,
            top_k=5,
        )

        if not results:

            print(
                "\nNo relevant papers found."
            )

        else:

            print()
            print("Relevant Papers")

            for number, paper in enumerate(
                results,
                start=1,
            ):

                print()
                print("-" * 60)

                print(
                    f"{number}. {paper['title']}"
                )

                print(
                    "arXiv ID:",
                    paper["id"],
                )

                print(
                    "Authors:",
                    paper["authors"],
                )

                print(
                    "Categories:",
                    paper["categories"],
                )

                print(
                    "Relevance:",
                    paper["score"],
                )

                print(
                    "\nAbstract:"
                )

                print(
                    paper["abstract"][:800]
                )

    except Exception as error:

        print()
        print(
            "Research Expert could not start."
        )

        print(error)