"""
Dynamic Knowledge Base Updater
GenAI Customer Service Bot

Reads configured knowledge sources from:

    data/knowledge_sources.json

and updates the persistent knowledge base.

Features:
- Loads configured knowledge sources
- Ignores disabled sources
- Reads text files
- Adds new sources
- Detects unchanged sources
- Replaces outdated source content
- Prevents duplicate source versions
- Reports update statistics

Currently supported source types:
- text_file
"""

from pathlib import Path
from datetime import datetime
import json
import sys


# =========================================================
# PROJECT PATH SETUP
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from modules.knowledge_base import DynamicKnowledgeBase


# =========================================================
# PATHS
# =========================================================

CONFIG_FILE = (
    BASE_DIR
    / "data"
    / "knowledge_sources.json"
)


# =========================================================
# LOAD SOURCE CONFIGURATION
# =========================================================

def load_source_config():
    """
    Load configured knowledge sources from:

        data/knowledge_sources.json

    Returns:
        list of source dictionaries
    """

    if not CONFIG_FILE.exists():

        raise FileNotFoundError(
            "Knowledge source configuration was not found:\n"
            f"{CONFIG_FILE}"
        )

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    sources = data.get(
        "sources",
        [],
    )

    if not isinstance(sources, list):

        raise ValueError(
            "'sources' must be a list inside "
            "knowledge_sources.json"
        )

    return sources


# =========================================================
# TEXT FILE READER
# =========================================================

def read_text_file(relative_path):
    """
    Read a UTF-8 text file.

    Example path:

        data/sources/products.txt
    """

    file_path = (
        BASE_DIR
        / relative_path
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Source file not found: {file_path}"
        )

    if not file_path.is_file():

        raise ValueError(
            f"Source path is not a file: {file_path}"
        )

    content = file_path.read_text(
        encoding="utf-8"
    )

    if not content.strip():

        raise ValueError(
            f"Source file is empty: {file_path}"
        )

    return content


# =========================================================
# SOURCE READER
# =========================================================

def read_source(source):
    """
    Read content from a configured knowledge source.

    Currently supported:

        text_file

    Additional source types such as PDF and websites
    can be added later.
    """

    source_type = source.get(
        "type"
    )

    if not source_type:

        raise ValueError(
            "Knowledge source does not contain a type."
        )

    if source_type == "text_file":

        path = source.get(
            "path"
        )

        if not path:

            raise ValueError(
                "Text source does not contain a path."
            )

        return read_text_file(
            path
        )

    raise ValueError(
        "Unsupported knowledge source type: "
        f"{source_type}"
    )


# =========================================================
# UPDATE KNOWLEDGE BASE
# =========================================================

def update_knowledge_base():
    """
    Read configured sources and synchronize them with
    the dynamic knowledge base.

    Behavior:

    NEW SOURCE
        -> added to knowledge base

    UNCHANGED SOURCE
        -> skipped

    CHANGED SOURCE
        -> old chunks removed
        -> new chunks inserted

    FAILED SOURCE
        -> error reported
        -> remaining sources continue processing
    """

    print()
    print("=" * 65)
    print("GENAI CUSTOMER SERVICE BOT")
    print("DYNAMIC KNOWLEDGE BASE UPDATE")
    print("=" * 65)

    started_at = datetime.now()

    print(
        "\nStarted:",
        started_at.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )

    # -----------------------------------------------------
    # Load configuration
    # -----------------------------------------------------

    try:

        sources = load_source_config()

    except Exception as error:

        print()
        print(
            "ERROR: Unable to load knowledge "
            "source configuration."
        )

        print(
            "Reason:",
            error,
        )

        return

    # -----------------------------------------------------
    # Initialize knowledge database
    # -----------------------------------------------------

    try:

        knowledge_base = DynamicKnowledgeBase()

    except Exception as error:

        print()
        print(
            "ERROR: Unable to initialize "
            "the knowledge base."
        )

        print(
            "Reason:",
            error,
        )

        return

    # -----------------------------------------------------
    # Filter enabled sources
    # -----------------------------------------------------

    enabled_sources = [
        source
        for source in sources
        if source.get(
            "enabled",
            True,
        )
    ]

    disabled_sources = [
        source
        for source in sources
        if not source.get(
            "enabled",
            True,
        )
    ]

    print()
    print(
        "Configured sources:",
        len(sources),
    )

    print(
        "Enabled sources:",
        len(enabled_sources),
    )

    print(
        "Disabled sources:",
        len(disabled_sources),
    )

    # -----------------------------------------------------
    # Counters
    # -----------------------------------------------------

    updated = 0
    unchanged = 0
    failed = 0

    total_chunks_added = 0
    total_chunks_removed = 0

    # -----------------------------------------------------
    # Process enabled sources
    # -----------------------------------------------------

    for number, source in enumerate(
        enabled_sources,
        start=1,
    ):

        name = source.get(
            "name",
            f"Source {number}",
        )

        source_type = source.get(
            "type",
            "Unknown",
        )

        print()
        print("-" * 65)

        print(
            f"[{number}/{len(enabled_sources)}] "
            f"{name}"
        )

        print(
            "Type:",
            source_type,
        )

        # -------------------------------------------------
        # Read source
        # -------------------------------------------------

        try:

            content = read_source(
                source
            )

        except Exception as error:

            failed += 1

            print(
                "Status: FAILED"
            )

            print(
                "Reason:",
                error,
            )

            continue

        # -------------------------------------------------
        # Update source
        # -------------------------------------------------

        try:

            result = (
                knowledge_base.update_source(
                    text=content,
                    source=name,
                    title=name,
                )
            )

        except Exception as error:

            failed += 1

            print(
                "Status: FAILED"
            )

            print(
                "Reason:",
                error,
            )

            continue

        # -------------------------------------------------
        # Updated source
        # -------------------------------------------------

        if (
            result.get("success")
            and result.get("updated")
        ):

            updated += 1

            chunks_added = result.get(
                "chunks_added",
                0,
            )

            chunks_removed = result.get(
                "chunks_removed",
                0,
            )

            total_chunks_added += (
                chunks_added
            )

            total_chunks_removed += (
                chunks_removed
            )

            print(
                "Status: UPDATED"
            )

            print(
                "Old chunks removed:",
                chunks_removed,
            )

            print(
                "New chunks added:",
                chunks_added,
            )

            print(
                "Message:",
                result.get(
                    "message",
                    "Knowledge source updated.",
                ),
            )

        # -------------------------------------------------
        # Unchanged source
        # -------------------------------------------------

        elif result.get("success"):

            unchanged += 1

            print(
                "Status: UNCHANGED"
            )

            print(
                "Message:",
                result.get(
                    "message",
                    "Source has not changed.",
                ),
            )

        # -------------------------------------------------
        # Failed update
        # -------------------------------------------------

        else:

            failed += 1

            print(
                "Status: FAILED"
            )

            print(
                "Reason:",
                result.get(
                    "message",
                    "Unknown error.",
                ),
            )

    # -----------------------------------------------------
    # Get final statistics
    # -----------------------------------------------------

    try:

        stats = (
            knowledge_base.get_stats()
        )

    except Exception:

        stats = {
            "documents": 0,
            "chunks": 0,
            "sources": 0,
            "last_update": None,
        }

    finished_at = datetime.now()

    duration = (
        finished_at
        - started_at
    ).total_seconds()

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print()
    print("=" * 65)
    print("UPDATE SUMMARY")
    print("=" * 65)

    print()
    print(
        "Sources updated:",
        updated,
    )

    print(
        "Sources unchanged:",
        unchanged,
    )

    print(
        "Sources failed:",
        failed,
    )

    print()
    print(
        "Old chunks removed:",
        total_chunks_removed,
    )

    print(
        "New chunks added:",
        total_chunks_added,
    )

    print()
    print("-" * 65)

    print(
        "Knowledge documents:",
        stats.get(
            "documents",
            0,
        ),
    )

    print(
        "Knowledge chunks:",
        stats.get(
            "chunks",
            0,
        ),
    )

    print(
        "Knowledge sources:",
        stats.get(
            "sources",
            0,
        ),
    )

    print(
        "Last update:",
        stats.get(
            "last_update",
        ),
    )

    print("-" * 65)

    print(
        "\nFinished:",
        finished_at.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )

    print(
        f"Duration: {duration:.2f} seconds"
    )

    print("=" * 65)

    # -----------------------------------------------------
    # Return summary for Streamlit later
    # -----------------------------------------------------

    return {
        "updated": updated,
        "unchanged": unchanged,
        "failed": failed,
        "chunks_added": total_chunks_added,
        "chunks_removed": total_chunks_removed,
        "documents": stats.get(
            "documents",
            0,
        ),
        "chunks": stats.get(
            "chunks",
            0,
        ),
        "sources": stats.get(
            "sources",
            0,
        ),
        "last_update": stats.get(
            "last_update",
        ),
        "duration": round(
            duration,
            2,
        ),
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    update_knowledge_base()