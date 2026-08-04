"""
MedQuAD Dataset Preparation
GenAI Customer Service Bot

Reads MedQuAD XML files recursively and creates:

    data/medquad.csv

Expected columns:
    question
    answer
    source
    category
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MEDQUAD_DIR = BASE_DIR / "data" / "MedQuAD"

OUTPUT_FILE = BASE_DIR / "data" / "medquad.csv"


def clean_text(text):
    """Normalize whitespace in XML text."""

    if not text:
        return ""

    return " ".join(str(text).split())


def get_element_text(element):
    """
    Extract all text contained inside an XML element,
    including text from nested tags.
    """

    if element is None:
        return ""

    return clean_text(" ".join(element.itertext()))


def find_text(element, names):
    """
    Search for the first matching child tag.

    Handles XML tags regardless of capitalization.
    """

    names = {name.lower() for name in names}

    for child in element.iter():

        tag = child.tag

        # Remove XML namespace if present
        if "}" in tag:
            tag = tag.split("}", 1)[1]

        if tag.lower() in names:
            text = get_element_text(child)

            if text:
                return text

    return ""


def extract_qa_pairs(xml_file):
    """
    Extract question-answer pairs from one MedQuAD XML file.
    """

    rows = []

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

    except ET.ParseError as error:
        print(f"Skipping invalid XML: {xml_file}")
        print(error)
        return rows

    # Information about the source page/topic
    focus = find_text(
        root,
        [
            "Focus",
            "Topic",
            "Disease",
            "Title",
        ],
    )

    source = find_text(
        root,
        [
            "Source",
            "URL",
        ],
    )

    if not source:
        source = xml_file.name

    # MedQuAD commonly stores questions inside QAPair elements.
    qa_elements = []

    for element in root.iter():

        tag = element.tag

        if "}" in tag:
            tag = tag.split("}", 1)[1]

        if tag.lower() == "qapair":
            qa_elements.append(element)

    for qa in qa_elements:

        question = find_text(
            qa,
            [
                "Question",
            ],
        )

        answer = find_text(
            qa,
            [
                "Answer",
            ],
        )

        if not question or not answer:
            continue

        # Question elements can contain a question type attribute.
        category = ""

        for element in qa.iter():

            tag = element.tag

            if "}" in tag:
                tag = tag.split("}", 1)[1]

            if tag.lower() == "question":

                category = (
                    element.attrib.get("type")
                    or element.attrib.get("Type")
                    or ""
                )

                break

        if not category:
            category = focus or xml_file.parent.name

        rows.append(
            {
                "question": question,
                "answer": answer,
                "source": source,
                "category": category,
            }
        )

    return rows


def prepare_dataset():
    """Process all MedQuAD XML files."""

    if not MEDQUAD_DIR.exists():

        print("\nMedQuAD directory was not found:")
        print(MEDQUAD_DIR)

        print(
            "\nDownload/clone MedQuAD and place the repository "
            "inside data/MedQuAD."
        )

        return

    xml_files = list(MEDQUAD_DIR.rglob("*.xml"))

    if not xml_files:

        print("\nNo XML files were found inside:")
        print(MEDQUAD_DIR)

        return

    print(f"\nFound {len(xml_files)} XML files.")

    all_rows = []

    for number, xml_file in enumerate(
        xml_files,
        start=1,
    ):

        rows = extract_qa_pairs(xml_file)

        all_rows.extend(rows)

        if number % 100 == 0:
            print(
                f"Processed {number}/{len(xml_files)} XML files..."
            )

    if not all_rows:

        print(
            "\nNo question-answer pairs were extracted."
        )

        return

    dataframe = pd.DataFrame(all_rows)

    # Remove empty values
    dataframe = dataframe.dropna(
        subset=[
            "question",
            "answer",
        ]
    )

    # Remove duplicate Q&A pairs
    dataframe = dataframe.drop_duplicates(
        subset=[
            "question",
            "answer",
        ]
    )

    # Remove rows containing blank strings
    dataframe = dataframe[
        dataframe["question"].str.strip().ne("")
        & dataframe["answer"].str.strip().ne("")
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print("\n--------------------------------")
    print("MedQuAD preparation completed.")
    print("--------------------------------")

    print(f"XML files: {len(xml_files)}")
    print(f"Q&A pairs: {len(dataframe)}")

    print("\nDataset saved to:")
    print(OUTPUT_FILE)

    print("\nColumns:")
    print(list(dataframe.columns))

    print("\nExample:")

    if not dataframe.empty:

        print(
            dataframe[
                [
                    "question",
                    "category",
                ]
            ].head()
        )


if __name__ == "__main__":
    prepare_dataset()