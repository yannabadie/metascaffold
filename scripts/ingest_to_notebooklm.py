"""Ingest research sources into NotebookLM with Deep Research.

Creates the MetaScaffold_Core notebook, uploads curated papers/repos as URL
sources, then runs Deep Research sessions to discover additional knowledge.

Leverages Google AI Ultra quotas:
  - 600 sources/notebook
  - 200 Deep Research sessions/day
  - 5,000 chats/day

Requires prior authentication via `uv run python scripts/nlm_login.py`.

Usage: uv run python scripts/ingest_to_notebooklm.py [--skip-upload] [--skip-research]
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


def _inject_truststore() -> None:
    """Inject OS certificate store for corporate proxy SSL."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


async def _get_client(timeout: float = 120.0):
    """Create authenticated NotebookLM client with extended timeout."""
    _inject_truststore()
    from notebooklm import NotebookLMClient
    return await NotebookLMClient.from_storage(timeout=timeout)


async def find_or_create_notebook(client, name: str) -> str:
    """Find existing notebook by name or create a new one. Returns notebook_id."""
    notebooks = await client.notebooks.list()
    for nb in notebooks:
        if nb.title == name:
            print(f"  Found existing notebook: {nb.title} (id={nb.id})")
            return nb.id

    nb = await client.notebooks.create(title=name)
    print(f"  Created notebook: {name} (id={nb.id})")
    return nb.id


async def upload_sources(client, notebook_id: str, sources: dict) -> dict:
    """Upload papers and repos as URL sources. Returns stats."""
    stats = {"uploaded": 0, "failed": 0, "skipped": 0}

    # Get existing sources to avoid duplicates
    try:
        existing = await client.sources.list(notebook_id)
    except Exception:
        existing = []
    existing_titles = {s.title.lower().strip() for s in existing}
    print(f"  Existing sources in notebook: {len(existing)}")

    all_urls = []
    # Foundational papers
    for paper in sources.get("foundational_papers", []):
        all_urls.append({"url": paper["url"], "title": paper["title"], "type": "foundational"})
    # SOTA papers (2025-2026)
    for paper in sources.get("sota_papers", []):
        all_urls.append({"url": paper["url"], "title": paper["title"], "type": "SOTA"})
    # Legacy flat "papers" key (backward compat)
    for paper in sources.get("papers", []):
        all_urls.append({"url": paper["url"], "title": paper["title"], "type": "paper"})
    # GitHub repos
    for repo in sources.get("github_repos", []):
        all_urls.append({"url": repo["url"], "title": repo["name"], "type": "repo"})

    for i, item in enumerate(all_urls, 1):
        # Simple duplicate check by title
        if item["title"].lower().strip() in existing_titles:
            print(f"  [{i}/{len(all_urls)}] SKIP (exists): {item['title']}")
            stats["skipped"] += 1
            continue

        print(f"  [{i}/{len(all_urls)}] Uploading {item['type']}: {item['title']}")
        try:
            await client.sources.add_url(notebook_id, item["url"])
            print(f"    -> OK")
            stats["uploaded"] += 1
        except Exception as e:
            print(f"    -> FAILED: {e}")
            stats["failed"] += 1

        # Rate limiting - be gentle with the API
        await asyncio.sleep(2)

    return stats


async def run_deep_research(client, notebook_id: str, queries: list[dict]) -> dict:
    """Run Deep Research sessions on curated queries. Returns stats."""
    stats = {"started": 0, "completed": 0, "sources_imported": 0, "failed": 0}

    for i, q in enumerate(queries, 1):
        print(f"\n  [{i}/{len(queries)}] Deep Research: {q['query'][:60]}...")
        print(f"    Purpose: {q['purpose']}")

        try:
            # Start deep research
            task = await client.research.start(
                notebook_id=notebook_id,
                query=q["query"],
                source="web",
                mode="deep",
            )

            if not task:
                print(f"    -> FAILED: No task returned")
                stats["failed"] += 1
                continue

            task_id = task["task_id"]
            print(f"    -> Started (task_id={task_id})")
            stats["started"] += 1

            # Poll until complete (deep research can take 2-5 minutes)
            max_polls = 60  # 5 minutes at 5s intervals
            for poll_num in range(max_polls):
                await asyncio.sleep(5)
                try:
                    result = await client.research.poll(notebook_id)
                except Exception as poll_err:
                    print(f"    -> Poll error (retrying): {poll_err}")
                    continue

                status = result.get("status", "unknown")
                if status == "completed":
                    sources_found = result.get("sources", [])
                    summary = result.get("summary", "")
                    print(f"    -> Completed! Found {len(sources_found)} sources")
                    if summary:
                        # Show first 200 chars of summary
                        print(f"    -> Summary: {summary[:200]}...")

                    # Import discovered sources (up to 20 per query to stay under 600 limit)
                    if sources_found:
                        importable = [s for s in sources_found if s.get("url")]
                        to_import = importable[:20]
                        if to_import:
                            try:
                                imported = await client.research.import_sources(
                                    notebook_id, task_id, to_import
                                )
                                n_imported = len(imported)
                                stats["sources_imported"] += n_imported
                                print(f"    -> Imported {n_imported} sources into notebook")
                            except Exception as e:
                                print(f"    -> Import failed: {e}")
                    stats["completed"] += 1
                    break
                elif status == "in_progress":
                    if poll_num % 6 == 0:  # Log every 30s
                        print(f"    -> Still researching... ({(poll_num + 1) * 5}s)")
                else:
                    print(f"    -> Unexpected status: {status}")
                    break
            else:
                print(f"    -> Timeout after {max_polls * 5}s")
                stats["failed"] += 1

        except Exception as e:
            print(f"    -> ERROR: {e}")
            stats["failed"] += 1

        # Pause between research sessions
        if i < len(queries):
            print(f"    Waiting 10s before next research session...")
            await asyncio.sleep(10)

    return stats


async def verify_notebook(client, notebook_id: str) -> dict:
    """Verify notebook contents and return summary."""
    try:
        sources = await client.sources.list(notebook_id)
    except Exception:
        sources = []
    return {
        "total_sources": len(sources),
        "source_titles": [s.title for s in sources[:30]],  # First 30
    }


async def main():
    # Parse simple flags
    skip_upload = "--skip-upload" in sys.argv
    skip_research = "--skip-research" in sys.argv

    sources_path = Path("docs/research_sources.json")
    if not sources_path.exists():
        print("ERROR: Run scripts/source_research.py first to generate source list.")
        sys.exit(1)

    with open(sources_path) as f:
        sources = json.load(f)

    notebook_name = "MetaScaffold_Core"

    n_foundational = len(sources.get("foundational_papers", []))
    n_sota = len(sources.get("sota_papers", []))
    n_papers = len(sources.get("papers", []))
    n_repos = len(sources.get("github_repos", []))
    n_queries = len(sources.get("deep_research_queries", []))
    n_total_upload = n_foundational + n_sota + n_papers + n_repos

    print("=" * 60)
    print("MetaScaffold Knowledge Base Ingestion")
    print("=" * 60)
    print(f"Notebook: {notebook_name}")
    print(f"Foundational papers: {n_foundational}")
    print(f"SOTA papers (2025-2026): {n_sota}")
    if n_papers:
        print(f"Legacy papers: {n_papers}")
    print(f"GitHub repos: {n_repos}")
    print(f"Total URL sources to upload: {n_total_upload}")
    print(f"Deep Research queries: {n_queries}")
    print(f"Flags: upload={'skip' if skip_upload else 'yes'}, research={'skip' if skip_research else 'yes'}")
    print()

    client = await _get_client()
    async with client:
        # Step 1: Find or create notebook
        print("[1/4] Finding or creating notebook...")
        notebook_id = await find_or_create_notebook(client, notebook_name)
        print()

        # Step 2: Upload curated sources
        if not skip_upload:
            print(f"[2/4] Uploading {n_total_upload} curated sources...")
            upload_stats = await upload_sources(client, notebook_id, sources)
            print(f"\n  Upload summary: {upload_stats}")
        else:
            print("[2/4] Skipping source upload (--skip-upload)")
            upload_stats = {"skipped": "all"}
        print()

        # Step 3: Run Deep Research sessions
        if not skip_research:
            queries = sources.get("deep_research_queries", [])
            if queries:
                print(f"[3/4] Running {len(queries)} Deep Research sessions (AI Ultra)...")
                research_stats = await run_deep_research(client, notebook_id, queries)
                print(f"\n  Research summary: {research_stats}")
            else:
                print("[3/4] No Deep Research queries defined")
                research_stats = {}
        else:
            print("[3/4] Skipping Deep Research (--skip-research)")
            research_stats = {"skipped": "all"}
        print()

        # Step 4: Verify
        print("[4/4] Verifying notebook contents...")
        verification = await verify_notebook(client, notebook_id)
        print(f"  Total sources in notebook: {verification['total_sources']}")
        print(f"  Source titles (first 30):")
        for title in verification["source_titles"]:
            # Handle non-ASCII titles (e.g. Chinese chars) on Windows cp1252 console
            safe_title = title.encode("ascii", errors="replace").decode("ascii")
            print(f"    - {safe_title}")

    print()
    print("=" * 60)
    print("JALON 2 — Knowledge Base Ingestion")
    print("=" * 60)
    print(f"  Notebook: {notebook_name}")
    print(f"  Total sources: {verification['total_sources']}")
    if not skip_upload:
        print(f"  Uploads: {upload_stats.get('uploaded', 0)} OK, {upload_stats.get('failed', 0)} failed, {upload_stats.get('skipped', 0)} skipped")
    if not skip_research:
        print(f"  Deep Research: {research_stats.get('completed', 0)}/{research_stats.get('started', 0)} completed, {research_stats.get('sources_imported', 0)} sources imported")
    print()
    print("Verify in NotebookLM web UI: https://notebooklm.google.com/")


if __name__ == "__main__":
    asyncio.run(main())
