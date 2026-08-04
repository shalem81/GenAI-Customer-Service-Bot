"""
Medical Q&A Module
GenAI Customer Service Bot

Provides:
1. Medical question retrieval
2. TF-IDF semantic-style search
3. Basic medical entity recognition
4. Source/category information

The final project can use a processed MedQuAD dataset stored as:
data/medquad.csv
"""

from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = BASE_DIR / "data" / "medquad.csv"


# Basic medical vocabulary for internship-level entity recognition.
# We can later replace/extend this with a biomedical NER model.
MEDICAL_ENTITIES = {
    "disease": [
        "diabetes",
        "asthma",
        "cancer",
        "hypertension",
        "arthritis",
        "pneumonia",
        "tuberculosis",
        "malaria",
        "migraine",
        "influenza",
        "flu",
    ],
    "symptom": [
        "fever",
        "cough",
        "headache",
        "fatigue",
        "nausea",
        "vomiting",
        "dizziness",
        "chest pain",
        "shortness of breath",
        "sore throat",
        "stomach pain",
    ],
    "treatment": [
        "insulin",
        "chemotherapy",
        "radiotherapy",
        "surgery",
        "antibiotics",
        "therapy",
        "vaccination",
    ],
}


def normalize_text(text: str) -> str:
    """Clean text before retrieval."""

    if not isinstance(text, str):
        return ""

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)

    return text


def extract_medical_entities(text: str) -> list:
    """
    Detect basic medical entities using a controlled vocabulary.

    Returns:
        [
            {
                "text": "diabetes",
                "type": "Disease"
            }
        ]
    """

    normalized = normalize_text(text)

    entities = []

    for entity_type, terms in MEDICAL_ENTITIES.items():

        for term in terms:

            pattern = r"\b" + re.escape(term) + r"\b"

            if re.search(pattern, normalized):

                entities.append(
                    {
                        "text": term,
                        "type": entity_type.title(),
                    }
                )

    return entities


class MedicalQABot:
    """Retrieve medical answers from the processed MedQuAD dataset."""

    def __init__(self, dataset_path=None):

        self.dataset_path = Path(dataset_path or DEFAULT_DATASET)

        self.data = None
        self.vectorizer = None
        self.question_matrix = None

        self._load_dataset()

    def _load_dataset(self):

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"MedQuAD dataset was not found at:\n"
                f"{self.dataset_path}\n\n"
                "Create data/medquad.csv before starting Medical Q&A."
            )

        self.data = pd.read_csv(self.dataset_path)

        required_columns = {
            "question",
            "answer",
            "source",
            "category",
        }

        missing = required_columns - set(self.data.columns)

        if missing:
            raise ValueError(
                "MedQuAD CSV is missing columns: "
                + ", ".join(sorted(missing))
            )

        self.data = self.data.dropna(
            subset=["question", "answer"]
        ).copy()

        self.data["question"] = (
            self.data["question"]
            .astype(str)
            .apply(normalize_text)
        )

        self.data["answer"] = (
            self.data["answer"]
            .astype(str)
            .str.strip()
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=30000,
        )

        self.question_matrix = self.vectorizer.fit_transform(
            self.data["question"]
        )

    def search(self, question: str, top_k: int = 3) -> list:
        """
        Retrieve the most relevant MedQuAD entries.
        """

        question = normalize_text(question)

        if not question:
            return []

        query_vector = self.vectorizer.transform([question])

        similarities = cosine_similarity(
            query_vector,
            self.question_matrix,
        ).flatten()

        top_indices = similarities.argsort()[::-1][:top_k]

        results = []

        for index in top_indices:

            row = self.data.iloc[index]

            results.append(
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "source": row["source"],
                    "category": row["category"],
                    "similarity": round(
                        float(similarities[index]),
                        4,
                    ),
                }
            )

        return results

    def answer_question(self, question: str) -> dict:
        """
        Return the best answer plus medical entities.
        """

        entities = extract_medical_entities(question)

        results = self.search(question, top_k=3)

        if not results:

            return {
                "question": question,
                "answer": (
                    "I couldn't find relevant information "
                    "in the medical knowledge base."
                ),
                "source": None,
                "category": None,
                "similarity": 0.0,
                "entities": entities,
            }

        best = results[0]

        # Avoid presenting extremely weak matches as useful answers.
        if best["similarity"] < 0.05:

            return {
                "question": question,
                "answer": (
                    "I couldn't find a sufficiently relevant "
                    "answer in the medical knowledge base."
                ),
                "source": None,
                "category": None,
                "similarity": best["similarity"],
                "entities": entities,
            }

        return {
            "question": question,
            "answer": best["answer"],
            "source": best["source"],
            "category": best["category"],
            "similarity": best["similarity"],
            "entities": entities,
        }

    def answer_hybrid(self, question: str, ollama_client=None, model: str = None) -> dict:
        """
        Hybrid Mode: Combine MedQuAD dataset retrieval with Ollama LLM generation.
        """
        entities = extract_medical_entities(question)
        search_results = self.search(question, top_k=3)

        if not search_results:
            db_context = "No specific MedQuAD dataset entry found for this query."
            source_info = "Ollama General AI Knowledge Base"
            category_info = "General Healthcare"
            top_similarity = 0.0
        else:
            context_blocks = []
            for item in search_results:
                context_blocks.append(f"Q: {item['question']}\nA: {item['answer']}\nSource: {item['source']}")
            db_context = "\n\n".join(context_blocks)
            source_info = search_results[0]["source"]
            category_info = search_results[0]["category"]
            top_similarity = search_results[0]["similarity"]

        if ollama_client is not None:
            system_prompt = (
                "You are an empathetic, professional Medical AI Assistant. "
                "Provide clear, informative, and medically responsible answers using the provided MedQuAD database context when relevant. "
                "Always include a gentle reminder that your guidance is for informational purposes and to consult a doctor."
            )
            prompt = (
                f"Medical Context from MedQuAD Database:\n{db_context}\n\n"
                f"User Question:\n{question}\n\n"
                "Synthesize a helpful, easy-to-understand explanation:"
            )
            llm_answer = ollama_client.generate(prompt, model=model, system=system_prompt)
            if llm_answer:
                return {
                    "question": question,
                    "answer": llm_answer,
                    "source": source_info,
                    "category": category_info,
                    "similarity": top_similarity,
                    "entities": entities,
                    "used_llm": True,
                    "search_results": search_results,
                }

        # Fallback to direct DB answer if LLM fails or is omitted
        raw_result = self.answer_question(question)
        raw_result["used_llm"] = False
        raw_result["search_results"] = search_results
        return raw_result



if __name__ == "__main__":

    try:

        bot = MedicalQABot()

        question = input(
            "Enter a medical question: "
        )

        result = bot.answer_question(question)

        print("\nQuestion:")
        print(result["question"])

        print("\nAnswer:")
        print(result["answer"])

        print("\nCategory:")
        print(result["category"])

        print("\nSource:")
        print(result["source"])

        print("\nSimilarity:")
        print(result["similarity"])

        print("\nDetected Entities:")

        if result["entities"]:

            for entity in result["entities"]:

                print(
                    f"- {entity['text']} "
                    f"({entity['type']})"
                )

        else:
            print("No entities detected.")

        print(
            "\nMedical Disclaimer: "
            "This chatbot provides educational information "
            "and is not a substitute for professional medical advice."
        )

    except Exception as error:

        print("\nMedical Q&A could not start.")
        print(error)