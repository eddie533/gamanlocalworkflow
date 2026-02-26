#!/usr/bin/env python3
"""
Stage 1 - Preliminary Filter

Reads companies from input/companies.csv, runs the preliminary filter
to eliminate obvious non-software companies, and writes the results
to the output/ directory.
"""

import asyncio
import csv
from pathlib import Path
from preliminary_filter import PreliminaryFilter

MAX_WORKERS = 50
INPUT_CSV = Path(__file__).parent / "input" / "companies.csv"
OUTPUT_DIR = Path(__file__).parent / "output"

# Map CSV columns to the keys PreliminaryFilter expects
COLUMN_MAP = {
    "company_name": "Company Name",
    "company_website": "Website",
    "description": "Description",
}


def load_companies(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            {COLUMN_MAP.get(k, k): v for k, v in row.items()}
            for row in reader
            if row.get("company_name")
        ]


async def filter_one(company: dict, semaphore: asyncio.Semaphore):
    """Run the filter for a single company in a thread pool."""
    async with semaphore:
        pf = PreliminaryFilter()  # fresh instance per task for thread safety
        result = await asyncio.to_thread(pf.filter_company, company)
        return company, result


async def filter_all(companies: list[dict]) -> list[tuple[dict, dict]]:
    semaphore = asyncio.Semaphore(MAX_WORKERS)
    tasks = [filter_one(company, semaphore) for company in companies]
    return await asyncio.gather(*tasks, return_exceptions=True)


def main():
    companies = load_companies(INPUT_CSV)
    print(f"Stage 1 — Filtering {len(companies)} companies\n")

    raw_results = asyncio.run(filter_all(companies))

    kept = []
    eliminated = []

    for entry in raw_results:
        if isinstance(entry, Exception):
            print(f"  error: {entry}")
            continue

        company, result = entry
        name = result.get("company_name", "?")
        if result["should_keep"]:
            kept.append(company)
            print(f"  + {name}")
        else:
            eliminated.append(company)
            print(f"  - {name}: {result.get('reason', '')}")

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if kept:
        kept_path = OUTPUT_DIR / "filtered_companies.csv"
        with open(kept_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=kept[0].keys())
            writer.writeheader()
            writer.writerows(kept)

    print(f"\nDone — kept {len(kept)}, eliminated {len(eliminated)}")
    print(f"Output: {OUTPUT_DIR}/filtered_companies.csv")


if __name__ == "__main__":
    main()
