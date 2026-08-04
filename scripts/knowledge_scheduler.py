"""
Knowledge Base Scheduler
GenAI Customer Service Bot

Periodically runs the dynamic knowledge-base updater.

Development default:
    Every 5 minutes

Production frequency can be configured using:
    KNOWLEDGE_UPDATE_MINUTES

Example in .env:
    KNOWLEDGE_UPDATE_MINUTES=60
"""

from pathlib import Path
from datetime import datetime
import os
import sys
import time

from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv


# ---------------------------------------------------------
# Project Setup
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BASE_DIR),
    )


from scripts.update_knowledge_base import update_knowledge_base


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv(
    BASE_DIR / ".env"
)


def get_update_interval():
    """
    Read update interval from environment.

    Default:
        5 minutes
    """

    value = os.getenv(
        "KNOWLEDGE_UPDATE_MINUTES",
        "5",
    )

    try:

        minutes = int(value)

        if minutes < 1:
            raise ValueError

        return minutes

    except ValueError:

        print(
            "Invalid KNOWLEDGE_UPDATE_MINUTES."
        )

        print(
            "Using default interval: 5 minutes."
        )

        return 5


# ---------------------------------------------------------
# Scheduled Job
# ---------------------------------------------------------

def scheduled_update():
    """Run one scheduled knowledge update."""

    print()
    print(
        "[Scheduler] Update started:",
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )

    try:

        update_knowledge_base()

        print(
            "[Scheduler] Update completed."
        )

    except Exception as error:

        print(
            "[Scheduler] Update failed:",
            error,
        )


# ---------------------------------------------------------
# Scheduler
# ---------------------------------------------------------

def start_scheduler():

    minutes = get_update_interval()

    scheduler = BackgroundScheduler()

    scheduler.add_job(
        scheduled_update,
        trigger="interval",
        minutes=minutes,
        id="knowledge_base_update",
        name="Dynamic Knowledge Base Update",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.start()

    print()
    print("=" * 60)
    print("KNOWLEDGE BASE SCHEDULER")
    print("=" * 60)

    print(
        f"Update interval: {minutes} minute(s)"
    )

    print(
        "Started:",
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )

    print(
        "\nPress Ctrl+C to stop."
    )

    # Run immediately once when scheduler starts.

    scheduled_update()

    try:

        while True:
            time.sleep(1)

    except KeyboardInterrupt:

        print(
            "\nStopping knowledge scheduler..."
        )

        scheduler.shutdown(
            wait=False
        )

        print(
            "Scheduler stopped."
        )


if __name__ == "__main__":
    start_scheduler()