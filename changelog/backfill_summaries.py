#!/usr/bin/env python3
"""
Backfill Summaries - Add AI summaries to existing changelog entries.

Reads CHANGELOG.md, finds entries without summaries, and adds them using LLM.
Respects .changelog-prompt.txt if it exists in the repo.

Usage:
    python changelog/backfill_summaries.py --limit 20
    python changelog/backfill_summaries.py --since 2024-06-01
    python changelog/backfill_summaries.py --dry-run
"""

import argparse
import re
import json
import os
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, List


# Provider registry - same as run.py
PROVIDERS = {
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "endpoint": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-5-haiku-20241022",
    },
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
    },
    "gemini": {
        "env_var": "GEMINI_API_KEY",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-2.0-flash-lite",
    },
}


def detect_provider() -> Optional[str]:
    """Return the first provider with an API key set."""
    for name, config in PROVIDERS.items():
        if os.environ.get(config["env_var"]):
            return name
    return None


def build_llm_request(provider: str, model: str, prompt: str) -> tuple[str, dict, dict]:
    """Build endpoint, headers, and body for a provider."""
    config = PROVIDERS[provider]
    api_key = os.environ.get(config["env_var"])
    endpoint = config["endpoint"]

    if provider == "anthropic":
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": model,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        }
    else:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.3,
        }

    return endpoint, headers, body


def parse_llm_response(provider: str, response_json: dict) -> str:
    """Extract text from provider response."""
    if provider == "anthropic":
        return response_json.get("content", [{}])[0].get("text", "")
    else:
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "")


def load_custom_prompt(repo_root: Path) -> Optional[str]:
    """Load custom prompt from .changelog-prompt.txt if it exists."""
    prompt_file = repo_root / ".changelog-prompt.txt"
    try:
        if prompt_file.exists():
            with open(prompt_file, "r") as f:
                prompt = f.read().strip()
                if prompt:
                    print(f"Using custom prompt from {prompt_file}")
                    return prompt
    except IOError as e:
        print(f"Warning: Could not read custom prompt: {e}")
    return None


def get_pr_body(pr_number: str) -> Optional[str]:
    """Get PR description using GitHub CLI."""
    import subprocess
    try:
        cmd = ["gh", "pr", "view", pr_number, "--json", "body"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        pr_data = json.loads(result.stdout)
        return pr_data.get("body", "").strip()
    except Exception:
        return None


def generate_summary(
    pr_number: str,
    title: str,
    provider: str,
    model: str,
    custom_prompt: Optional[str],
    context_limit: int,
) -> Optional[List[str]]:
    """Generate summary for a PR."""
    pr_body = get_pr_body(pr_number)

    if not pr_body:
        return None

    # Construct context
    context_parts = [f"PR Title: {title}"]
    if pr_body:
        context_parts.append(f"PR Description:\n{pr_body[:context_limit]}")
    context = "\n\n".join(context_parts)

    # Use custom prompt or default
    if custom_prompt:
        prompt = custom_prompt.replace("{context}", context)
    else:
        prompt = f"""Please analyze this pull request and provide 2-4 concise bullet points summarizing what was changed and why. Focus on the functional changes and their purpose, not technical implementation details.

{context}

Please respond with only bullet points, starting each with "•". Keep each point under 25 words."""

    try:
        endpoint, headers, body = build_llm_request(provider, model, prompt)
        response = requests.post(endpoint, headers=headers, json=body, timeout=30)

        if response.status_code == 200:
            summary_text = parse_llm_response(provider, response.json())

            # Parse bullet points
            bullets = []
            for line in summary_text.split("\n"):
                line = line.strip()
                if line.startswith("•") or line.startswith("-") or line.startswith("*"):
                    bullets.append(line[1:].strip())
                elif line and not bullets:
                    bullets.append(line)

            return bullets[:4] if bullets else None
        else:
            print(f"  Warning: API error {response.status_code}")
            return None

    except Exception as e:
        print(f"  Warning: Could not generate summary: {e}")
        return None


def parse_changelog_entries(content: str) -> List[dict]:
    """Parse changelog entries from content."""
    entries = []

    # Match blockquote entries: > ### 📅 DATE | TITLE
    pattern = r'(> ### 📅 (\d{4}-\d{2}-\d{2}) \| (.+?)(?:\s*\(\[#(\d+)\]|\s*\(#(\d+)\)).*?)(?=\n> ### 📅|\n## |\Z)'

    for match in re.finditer(pattern, content, re.DOTALL):
        full_block = match.group(1)
        date_str = match.group(2)
        title = match.group(3).strip()
        pr_number = match.group(4) or match.group(5)

        # Check if entry has summary bullets (lines starting with > •)
        has_summary = bool(re.search(r'^> •', full_block, re.MULTILINE))

        entries.append({
            "full_block": full_block,
            "date": date_str,
            "title": title,
            "pr_number": pr_number,
            "has_summary": has_summary,
            "start": match.start(),
            "end": match.end(),
        })

    return entries


def add_summary_to_entry(entry_block: str, summary: List[str]) -> str:
    """Add summary bullets to an entry block."""
    lines = entry_block.split("\n")
    result = []

    # Find where to insert summary (after metadata line, before commit link)
    inserted = False
    for i, line in enumerate(lines):
        result.append(line)
        # Insert after the metadata line (Author/Approved/Ticket line)
        if not inserted and line.startswith("> **Author:**"):
            result.append(">")
            for bullet in summary:
                result.append(f"> • {bullet}")
            inserted = True

    # If no metadata line found, insert after first line
    if not inserted and len(result) > 1:
        new_result = [result[0], ">"]
        for bullet in summary:
            new_result.append(f"> • {bullet}")
        new_result.extend(result[1:])
        result = new_result

    return "\n".join(result)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill AI summaries for existing changelog entries",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of entries to process",
    )

    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Only process entries after this date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=list(PROVIDERS.keys()),
        default=None,
        help="LLM provider (default: auto-detect)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model (default: provider's default)",
    )

    parser.add_argument(
        "--context-limit",
        type=int,
        default=3000,
        help="Max chars of PR description (default: 3000)",
    )

    args = parser.parse_args()

    repo_root = Path.cwd()
    changelog_path = repo_root / "CHANGELOG.md"

    if not changelog_path.exists():
        print("Error: CHANGELOG.md not found")
        return 1

    # Setup provider
    provider = args.provider or detect_provider()
    if not provider:
        print("Error: No LLM API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY")
        return 1

    model = args.model or PROVIDERS[provider]["model"]
    print(f"Using {provider} with model {model}")

    # Load custom prompt
    custom_prompt = load_custom_prompt(repo_root)

    # Read changelog
    with open(changelog_path, "r") as f:
        content = f.read()

    # Parse entries
    entries = parse_changelog_entries(content)
    print(f"Found {len(entries)} changelog entries")

    # Filter entries
    entries_to_process = []
    for entry in entries:
        if entry["has_summary"]:
            continue
        if not entry["pr_number"]:
            continue
        if args.since:
            if entry["date"] < args.since:
                continue
        entries_to_process.append(entry)

    print(f"Found {len(entries_to_process)} entries without summaries")

    if args.limit:
        entries_to_process = entries_to_process[:args.limit]
        print(f"Processing {len(entries_to_process)} entries (limited)")

    if not entries_to_process:
        print("Nothing to process")
        return 0

    # Process entries (in reverse order to maintain positions)
    updated_content = content
    processed = 0

    for entry in reversed(entries_to_process):
        print(f"Processing PR #{entry['pr_number']}: {entry['title'][:50]}...")

        summary = generate_summary(
            entry["pr_number"],
            entry["title"],
            provider,
            model,
            custom_prompt,
            args.context_limit,
        )

        if summary:
            new_block = add_summary_to_entry(entry["full_block"], summary)
            updated_content = (
                updated_content[:entry["start"]] +
                new_block +
                updated_content[entry["end"]:]
            )
            processed += 1
            print(f"  Added {len(summary)} bullet points")
        else:
            print(f"  Skipped (no summary generated)")

    print(f"\nProcessed {processed}/{len(entries_to_process)} entries")

    if args.dry_run:
        print("\nDRY RUN - No changes written")
        print("\n--- Preview of changes ---")
        print(updated_content[:2000])
        if len(updated_content) > 2000:
            print("... (truncated)")
    else:
        with open(changelog_path, "w") as f:
            f.write(updated_content)
        print(f"Updated {changelog_path}")

    return 0


if __name__ == "__main__":
    exit(main())
