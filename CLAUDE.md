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
- **Merge with existing**: New entries are prepended to the `[Unreleased]` section, preserving existing content
- **GitHub CLI dependency**: Requires `gh` CLI for PR metadata (author, approver, workflow runs)

## GitHub Actions Integration

The workflow (`.github/workflows/changelog.yml`) is designed as a **reusable workflow** called from other repositories. Supports two modes:
- `use-pull-request: true` (default) - Creates PR for changelog updates
- `use-pull-request: false` - Direct commit to main with `[skip ci]`
