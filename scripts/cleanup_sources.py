"""Clean up NotebookLM sources: remove duplicates and outdated content.

Lists all sources, identifies duplicates (by URL or similar titles),
and flags outdated/irrelevant sources for removal.

Usage:
  uv run python scripts/cleanup_sources.py --list          # List all sources
  uv run python scripts/cleanup_sources.py --duplicates     # Show duplicates
  uv run python scripts/cleanup_sources.py --cleanup        # Remove duplicates + outdated
"""

from __future__ import annotations

import asyncio
import re
import sys
from collections import defaultdict


def _inject_truststore() -> None:
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


async def _get_client():
    _inject_truststore()
    from notebooklm import NotebookLMClient
    return await NotebookLMClient.from_storage(timeout=120.0)


# Patterns for sources that are NOT relevant to MetaScaffold metacognition
IRRELEVANT_PATTERNS = [
    r"(?i)deepseek.?r1",          # DeepSeek-R1 specific articles (outdated focus)
    r"(?i)hiring|job|career",     # Job postings
    r"(?i)stock|market|invest",   # Finance
    r"(?i)news\s+roundup",        # Generic news
]

# Titles that indicate low-value content
LOW_VALUE_PATTERNS = [
    r"(?i)^\(PDF\)\s+\(PDF\)",     # Double PDF prefix
    r"(?i)zealous\s+system",       # Spam-like domains
    r"(?i)druid\s+ai",            # Marketing fluff
]


def normalize_title(title: str) -> str:
    """Normalize title for duplicate detection."""
    t = title.lower().strip()
    # Remove common prefixes
    t = re.sub(r"^\(pdf\)\s*", "", t)
    # Remove trailing site names after " - "
    t = re.sub(r"\s*-\s*(arxiv|arxiv\.org|researchgate|github|hugging face|openreview|acl anthology|emergent mind|goatstack\.ai|takara tldr).*$", "", t)
    # Remove special chars
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


async def main():
    mode = "--list"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    notebook_name = "MetaScaffold_Core"

    client = await _get_client()
    async with client:
        # Find notebook
        notebooks = await client.notebooks.list()
        nb = next((n for n in notebooks if n.title == notebook_name), None)
        if not nb:
            print(f"ERROR: Notebook '{notebook_name}' not found")
            sys.exit(1)

        sources = await client.sources.list(nb.id)
        print(f"Notebook: {notebook_name} ({nb.id})")
        print(f"Total sources: {len(sources)}")
        print()

        if mode == "--list":
            for i, s in enumerate(sources, 1):
                safe_title = s.title.encode("ascii", errors="replace").decode("ascii")
                print(f"  {i:3d}. [{s.id[:8]}] {safe_title}")

        elif mode in ("--duplicates", "--cleanup"):
            # Group by normalized title
            title_groups: dict[str, list] = defaultdict(list)
            for s in sources:
                norm = normalize_title(s.title)
                title_groups[norm].append(s)

            # Find duplicates
            duplicates = []
            for norm_title, group in title_groups.items():
                if len(group) > 1:
                    # Keep the first, mark rest as duplicates
                    for dup in group[1:]:
                        duplicates.append(dup)

            # Find irrelevant sources
            irrelevant = []
            for s in sources:
                for pattern in IRRELEVANT_PATTERNS + LOW_VALUE_PATTERNS:
                    if re.search(pattern, s.title):
                        irrelevant.append(s)
                        break

            # Deduplicate the removal list
            to_remove_ids = set()
            to_remove = []
            for s in duplicates + irrelevant:
                if s.id not in to_remove_ids:
                    to_remove_ids.add(s.id)
                    to_remove.append(s)

            print(f"Duplicates found: {len(duplicates)}")
            for s in duplicates:
                safe = s.title.encode("ascii", errors="replace").decode("ascii")
                print(f"  DUP: {safe}")

            print(f"\nIrrelevant/outdated found: {len(irrelevant)}")
            for s in irrelevant:
                safe = s.title.encode("ascii", errors="replace").decode("ascii")
                print(f"  OLD: {safe}")

            print(f"\nTotal to remove: {len(to_remove)}")

            if mode == "--cleanup" and to_remove:
                print(f"\nRemoving {len(to_remove)} sources...")
                removed = 0
                for s in to_remove:
                    try:
                        await client.sources.delete(nb.id, s.id)
                        safe = s.title.encode("ascii", errors="replace").decode("ascii")
                        print(f"  REMOVED: {safe}")
                        removed += 1
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        print(f"  FAILED to remove {s.id}: {e}")

                # Verify
                remaining = await client.sources.list(nb.id)
                print(f"\nCleanup complete: {removed} removed, {len(remaining)} remaining")
            elif mode == "--duplicates":
                print("\nRun with --cleanup to actually remove these sources.")

        else:
            print(f"Unknown mode: {mode}")
            print("Usage: --list | --duplicates | --cleanup")


if __name__ == "__main__":
    asyncio.run(main())
