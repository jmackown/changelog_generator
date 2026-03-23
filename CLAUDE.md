# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated changelog generator that extracts PR merge commits from a git repository's main branch and creates Markdown changelogs enriched with GitHub metadata and optional AI-generated summaries.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Preview changelog (dry run, safe)
python changelog/run.py --dry-run

# Generate and write changelog
python changelog/run.py --write --recent 30

# Generate with AI summaries (requires ANTHROPIC_API_KEY or OPENAI_API_KEY)
python changelog/run.py --write --with-summaries

# Incremental update (for CI/CD)
python changelog/run.py --write --since-last-commit

# Generate entry for a specific PR
python changelog/run.py --for-pr 123
```

## Architecture

Single-file Python script (`changelog/run.py`) with one main class:

**`ChangelogGenerator`** - Orchestrates the entire workflow:
- Extracts PR merge commits from main branch using `git log --first-parent --grep='#'`
- Enriches entries via GitHub CLI (`gh pr list`, `gh run list`) for author, approver, workflow data
- Optionally generates AI summaries via Anthropic or OpenAI APIs
- Caches summaries in `.changelog-summaries.json` to avoid re-processing
- Merges new entries with existing CHANGELOG.md content

**`ChangeEntry`** - NamedTuple holding commit metadata (hash, title, PR number, JIRA ticket, author, approver, workflow run, summary)

## Key Design Decisions

- **Safe by default**: `--dry-run` is the default mode; `--write` required to modify files
- **PR-only commits**: Uses `--first-parent` and `--grep=#` to filter to only PR merges on main
- **Merge with existing**: New entries are prepended to year sections, preserving existing content
- **GitHub CLI dependency**: Requires `gh` CLI for PR metadata (author, approver, workflow runs)
- **On-merge workflow**: When a PR merges to main, the workflow extracts the PR number from the merge commit, generates a changelog entry with metadata and optional AI summary, and commits directly to CHANGELOG.md. Concurrent merges are serialised via GitHub Actions concurrency groups.

## GitHub Actions Integration

The workflow (`.github/workflows/changelog.yml`) is a **reusable workflow** called from other repositories. It fires on push to main, extracts the merged PR number from the commit message, generates a changelog entry, and commits to CHANGELOG.md with `[skip ci]`.
