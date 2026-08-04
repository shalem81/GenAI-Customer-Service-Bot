"""
Dynamic Knowledge Base Module
GenAI Customer Service Bot

Features:
- Add text documents dynamically
- Split documents into searchable chunks
- Persistent JSON storage
- TF-IDF retrieval
- Cosine similarity search
- Duplicate detection
- Source replacement
- Dynamic knowledge updates
- Knowledge-base statistics
"""

from pathlib import Path
from datetime import datetime
import hashlib
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DB_DIR = BASE_DIR / "vector_db"

KNOWLEDGE_FILE = (
    VECTOR_DB_DIR
    / "knowledge.json"
)


# =========================================================
# DYNAMIC KNOWLEDGE BASE
# =========================================================

class DynamicKnowledgeBase:

    def __init__(self, storage_file=None):

        self.storage_file = Path(
            storage_file or KNOWLEDGE_FILE
        )

        self.storage_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.documents = []

        self.vectorizer = None

        self.document_matrix = None

        self.load()

    # =====================================================
    # STORAGE
    # =====================================================

    def load(self):
        """
        Load existing knowledge from JSON storage.
        """

        if not self.storage_file.exists():

            self.documents = []

            self._build_index()

            return

        try:

            with open(
                self.storage_file,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            if isinstance(data, list):

                self.documents = data

            else:

                self.documents = []

        except (
            json.JSONDecodeError,
            OSError,
        ):

            self.documents = []

        self._build_index()

    def save(self):
        """
        Save the current knowledge base to disk.
        """

        with open(
            self.storage_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.documents,
                file,
                indent=4,
                ensure_ascii=False,
            )

    # =====================================================
    # TEXT PROCESSING
    # =====================================================

    @staticmethod
    def clean_text(text):
        """
        Normalize whitespace.
        """

        if not text:

            return ""

        return re.sub(
            r"\s+",
            " ",
            str(text),
        ).strip()

    @staticmethod
    def generate_hash(text):
        """
        Generate SHA-256 hash for a document.
        """

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def chunk_text(
        text,
        chunk_size=120,
        overlap=20,
    ):
        """
        Split text into overlapping word chunks.

        Example:

        chunk_size = 120
        overlap = 20

        Chunk 1:
            words 0-119

        Chunk 2:
            words 100-219
        """

        text = DynamicKnowledgeBase.clean_text(
            text
        )

        if not text:

            return []

        words = text.split()

        if len(words) <= chunk_size:

            return [text]

        chunks = []

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk = " ".join(
                words[start:end]
            )

            if chunk:

                chunks.append(
                    chunk
                )

            if end >= len(words):

                break

            start = end - overlap

        return chunks

    # =====================================================
    # SEARCH INDEX
    # =====================================================

    def _build_index(self):
        

        if not self.documents:

            self.vectorizer = None

            self.document_matrix = None

            return

        texts = [
            document.get(
                "content",
                "",
            )
            for document in self.documents
            if document.get(
                "content"
            )
        ]

        if not texts:

            self.vectorizer = None

            self.document_matrix = None

            return

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
        )

        self.document_matrix = (
            self.vectorizer.fit_transform(
                texts
            )
        )

    # =====================================================
    # ADD DOCUMENT
    # =====================================================

    def add_document(
        self,
        text,
        source="Manual",
        title="Untitled",
    ):
        """
        Add a new document.

        Exact duplicate documents are ignored.
        """

        text = self.clean_text(
            text
        )

        if not text:

            return {
                "success": False,
                "message": "Document is empty.",
                "chunks_added": 0,
            }

        document_hash = self.generate_hash(
            text
        )

        existing_hashes = {
            document.get(
                "document_hash"
            )
            for document in self.documents
        }

        if document_hash in existing_hashes:

            return {
                "success": False,
                "message": (
                    "This document already exists "
                    "in the knowledge base."
                ),
                "chunks_added": 0,
            }

        chunks = self.chunk_text(
            text
        )

        timestamp = (
            datetime.now().isoformat(
                timespec="seconds"
            )
        )

        for index, chunk in enumerate(
            chunks
        ):

            record = {
                "id": (
                    f"{document_hash[:12]}_"
                    f"{index}"
                ),
                "title": title,
                "source": source,
                "content": chunk,
                "chunk_index": index,
                "document_hash": document_hash,
                "added_at": timestamp,
            }

            self.documents.append(
                record
            )

        self.save()

        self._build_index()

        return {
            "success": True,
            "message": (
                "Knowledge base updated "
                "successfully."
            ),
            "chunks_added": len(
                chunks
            ),
            "document_hash": document_hash,
        }

    # =====================================================
    # REMOVE SOURCE
    # =====================================================

    def remove_source(
        self,
        source,
    ):
        """
        Remove every chunk belonging to a source.

        Returns:
            Number of chunks removed.
        """

        if not source:

            return 0

        original_count = len(
            self.documents
        )

        self.documents = [
            document
            for document in self.documents
            if document.get(
                "source"
            ) != source
        ]

        removed = (
            original_count
            - len(self.documents)
        )

        if removed > 0:

            self.save()

            self._build_index()

        return removed

    # =====================================================
    # UPDATE SOURCE
    # =====================================================

    def update_source(
        self,
        text,
        source,
        title="Untitled",
    ):
        """
        Add or update a configured knowledge source.

        New source:
            Add normally.

        Existing source with identical content:
            Do nothing.

        Existing source with changed content:
            Remove old chunks and add new chunks.
        """

        text = self.clean_text(
            text
        )

        if not text:

            return {
                "success": False,
                "updated": False,
                "message": (
                    "Document is empty."
                ),
                "chunks_added": 0,
                "chunks_removed": 0,
            }

        new_hash = self.generate_hash(
            text
        )

        source_documents = [
            document
            for document in self.documents
            if document.get(
                "source"
            ) == source
        ]

        # -------------------------------------------------
        # Check existing source
        # -------------------------------------------------

        if source_documents:

            existing_hashes = {
                document.get(
                    "document_hash"
                )
                for document
                in source_documents
            }

            if new_hash in existing_hashes:

                return {
                    "success": True,
                    "updated": False,
                    "message": (
                        "Source has not changed."
                    ),
                    "chunks_added": 0,
                    "chunks_removed": 0,
                    "document_hash": new_hash,
                }

        # -------------------------------------------------
        # Remove old version
        # -------------------------------------------------

        removed = self.remove_source(
            source
        )

        # -------------------------------------------------
        # Add new version
        # -------------------------------------------------

        result = self.add_document(
            text=text,
            source=source,
            title=title,
        )

        if not result["success"]:

            return {
                "success": False,
                "updated": False,
                "message": result.get(
                    "message",
                    "Unable to update source.",
                ),
                "chunks_added": 0,
                "chunks_removed": removed,
            }

        if removed > 0:

            message = (
                "Knowledge source updated "
                "successfully."
            )

        else:

            message = (
                "New knowledge source added "
                "successfully."
            )

        return {
            "success": True,
            "updated": True,
            "message": message,
            "chunks_added": result.get(
                "chunks_added",
                0,
            ),
            "chunks_removed": removed,
            "document_hash": result.get(
                "document_hash"
            ),
        }

    # =====================================================
    # SEARCH
    # =====================================================

    def search(
        self,
        query,
        top_k=3,
        minimum_score=0.01,
    ):
        """
        Search knowledge using TF-IDF and
        cosine similarity.
        """

        query = self.clean_text(
            query
        )

        if not query:

            return []

        if self.vectorizer is None:

            return []

        if self.document_matrix is None:

            return []

        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        scores = cosine_similarity(
            query_vector,
            self.document_matrix,
        ).flatten()

        top_indices = (
            scores.argsort()[::-1][
                :top_k
            ]
        )

        results = []

        for index in top_indices:

            score = float(
                scores[index]
            )

            if score < minimum_score:

                continue

            document = (
                self.documents[index]
            )

            results.append(
                {
                    "id": document.get(
                        "id"
                    ),
                    "title": document.get(
                        "title"
                    ),
                    "source": document.get(
                        "source"
                    ),
                    "content": document.get(
                        "content"
                    ),
                    "score": round(
                        score,
                        4,
                    ),
                    "added_at": document.get(
                        "added_at"
                    ),
                }
            )

        return results

    def search_hybrid(self, query: str, ollama_client=None, model: str = None, top_k: int = 5) -> dict:
        """
        Hybrid Mode: Search Knowledge Base documents and synthesize response with Ollama LLM.
        """
        results = self.search(query=query, top_k=top_k)

        if not results:
            context = "No relevant knowledge base documents found."
        else:
            context_blocks = []
            for item in results:
                context_blocks.append(f"Title: {item['title']}\nSource: {item['source']}\nContent: {item['content']}")
            context = "\n\n---\n\n".join(context_blocks)

        if ollama_client is not None:
            answer = ollama_client.generate_rag_response(
                query=query,
                context=context,
                domain="Company Knowledge & Customer Support",
                model=model,
            )
            if answer:
                return {
                    "query": query,
                    "answer": answer,
                    "results": results,
                    "used_llm": True,
                }

        return {
            "query": query,
            "answer": f"Found {len(results)} matching document(s) in knowledge base.",
            "results": results,
            "used_llm": False,
        }


    # =====================================================
    # STATISTICS
    # =====================================================

    def get_stats(self):
        """
        Return knowledge-base statistics.
        """

        document_hashes = {
            document.get(
                "document_hash"
            )
            for document in self.documents
            if document.get(
                "document_hash"
            )
        }

        sources = {
            document.get(
                "source"
            )
            for document in self.documents
            if document.get(
                "source"
            )
        }

        last_update = None

        if self.documents:

            dates = [
                document.get(
                    "added_at"
                )
                for document
                in self.documents
                if document.get(
                    "added_at"
                )
            ]

            if dates:

                last_update = max(
                    dates
                )

        return {
            "documents": len(
                document_hashes
            ),
            "chunks": len(
                self.documents
            ),
            "sources": len(
                sources
            ),
            "last_update": last_update,
        }

    # =====================================================
    # CLEAR DATABASE
    # =====================================================

    def clear(self):
        """
        Remove all knowledge from the database.

        Mainly used for testing and development.
        """

        self.documents = []

        self.save()

        self._build_index()


# =========================================================
# MANUAL TEST
# =========================================================

if __name__ == "__main__":

    knowledge_base = (
        DynamicKnowledgeBase()
    )

    print()
    print("=" * 60)
    print("DYNAMIC KNOWLEDGE BASE")
    print("=" * 60)

    stats = (
        knowledge_base.get_stats()
    )

    print(
        "\nDocuments:",
        stats["documents"],
    )

    print(
        "Chunks:",
        stats["chunks"],
    )

    print(
        "Sources:",
        stats["sources"],
    )

    print(
        "Last Update:",
        stats["last_update"],
    )

    print()
    print("1. Add knowledge")
    print("2. Search knowledge")
    print("3. View statistics")

    choice = input(
        "\nSelect option: "
    ).strip()

    # -----------------------------------------------------
    # Add
    # -----------------------------------------------------

    if choice == "1":

        title = input(
            "Document title: "
        ).strip()

        source = input(
            "Source: "
        ).strip()

        text = input(
            "Enter information: "
        ).strip()

        result = (
            knowledge_base.add_document(
                text=text,
                source=(
                    source
                    or "Manual"
                ),
                title=(
                    title
                    or "Untitled"
                ),
            )
        )

        print()
        print("Result:")
        print(result)

    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    elif choice == "2":

        query = input(
            "Enter your question: "
        ).strip()

        results = (
            knowledge_base.search(
                query,
                top_k=3,
            )
        )

        if not results:

            print()
            print(
                "No relevant information found."
            )

        else:

            print()
            print("Search Results:")

            for number, result in enumerate(
                results,
                start=1,
            ):

                print()
                print(
                    f"Result {number}"
                )

                print(
                    "Title:",
                    result["title"],
                )

                print(
                    "Source:",
                    result["source"],
                )

                print(
                    "Score:",
                    result["score"],
                )

                print(
                    "Content:",
                    result["content"],
                )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    elif choice == "3":

        print()
        print("Knowledge Base Statistics")

        print(
            "Documents:",
            stats["documents"],
        )

        print(
            "Chunks:",
            stats["chunks"],
        )

        print(
            "Sources:",
            stats["sources"],
        )

        print(
            "Last Update:",
            stats["last_update"],
        )

    else:

        print(
            "\nInvalid option."
        )