# Changelog Generator

Automated changelog generator that extracts PR merge commits from a git repository's main branch and creates beautiful, enriched Markdown changelogs with GitHub metadata and optional AI-generated summaries.

## Features

- Extracts only PR merge commits from the main branch (excludes direct commits)
- Enriches entries with GitHub metadata:
  - PR author and approver
  - GitHub Actions workflow run links
  - JIRA ticket references (any PROJECT-123 pattern)
- Optional AI-generated summaries using Anthropic Claude, OpenAI, or Google Gemini
- Customizable prompts per repository
- Smart caching to avoid re-processing commits
- Multiple generation modes: date range, commit range, recent N commits, or incremental
- Safe dry-run mode by default

## Quick Start: Reusable GitHub Action

The easiest way to use this tool is as a **reusable workflow** - just add one small file to your repository and the changelog generates automatically!

### 1. Add Workflow File

Create `.github/workflows/changelog.yml` in your repository:

```yaml
name: Generate Changelog

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  changelog:
    uses: jmackown/changelog_generator/.github/workflows/changelog.yml@main
    secrets:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Replace `jmackown` with the GitHub username/org where this repo lives.

**That's it!** The changelog will generate automatically when PRs merge to main.

### 2. Configure Your Repository

**Required:**

1. **Add workflow file** (above)
2. **Enable workflow permissions**:
   - Go to Settings > Actions > General
   - Select "Read and write permissions"
   - Check "Allow GitHub Actions to create and approve pull requests"

**Optional:**

3. **Add API key** (for AI summaries):
   - Go to Settings > Secrets and variables > Actions
   - Add `ANTHROPIC_API_KEY` with your Anthropic API key
   - (Or `OPENAI_API_KEY` if using OpenAI)

4. **Enable auto-merge** (recommended):
   - Go to Settings > General > Pull Requests
   - Check "Allow auto-merge"
   - Without this, you'll need to manually merge each changelog PR

**Repository access:**
- Works automatically if in same GitHub account
- Works if `changelog_generator` is public
- Won't work if private and in different account

### 3. Test It

1. Merge a test PR to main
2. Go to **Actions** tab and watch the "Generate Changelog" workflow run
3. Go to **Pull Requests** tab and look for the automated changelog PR
4. View the updated `CHANGELOG.md` after it merges

You can also manually trigger via Actions > Generate Changelog > Run workflow.

### How It Works

1. After a PR merges to main, the workflow triggers
2. Fetches the changelog generator script from `changelog_generator`
3. Generates a changelog from your PR history
4. Creates a PR with the changelog updates
5. Auto-merges if you have auto-merge enabled

## Configuration

### Reusable Workflow Options

When calling the reusable workflow, you can pass inputs:

**Disable AI summaries (faster, free):**
```yaml
jobs:
  changelog:
    uses: jmackown/changelog_generator/.github/workflows/changelog.yml@main
    with:
      with-summaries: false
    # No API key needed
```

**Use OpenAI instead of Anthropic:**
```yaml
jobs:
  changelog:
    uses: jmackown/changelog_generator/.github/workflows/changelog.yml@main
    with:
      provider: 'openai'
      model: 'gpt-4o-mini'
    secrets:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

**Use Gemini:**
```yaml
jobs:
  changelog:
    uses: jmackown/changelog_generator/.github/workflows/changelog.yml@main
    with:
      provider: 'gemini'
      model: 'gemini-2.0-flash-lite'
    secrets:
      GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
```

**Use better AI model (more expensive):**
```yaml
jobs:
  changelog:
    uses: jmackown/changelog_generator/.github/workflows/changelog.yml@main
    with:
      model: 'claude-3-5-sonnet-20241022'  # Better quality
    secrets:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

**Change Python version:**
```yaml
jobs:
  changelog:
    uses: jmackown/changelog_generator/.github/workflows/changelog.yml@main
    with:
      python-version: '3.12'
    secrets:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

### Changelog Modes

The workflow supports three modes for committing changelog updates:

| Mode | Trigger | Behavior | Best for |
|------|---------|----------|----------|
| `direct-commit` | PR merges to main | Commits changelog directly to main | Personal projects, no branch protection |
| `pull-request` | PR merges to main | Creates separate PR for changelog | Repos with branch protection |
| `update-branch` | PR is approved | Adds changelog to the PR branch before merge | Clean history, changelog in PR |

**Mode: direct-commit (default)**

Commits changelog directly to main after a PR merges:

```yaml
name: Generate Changelog
on:
  push:
    branches: [main]

jobs:
  changelog:
    uses: jmackown/changelog_generator/.github/workflows/changelog.yml@main
    with:
      mode: 'direct-commit'
    secrets:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

**Mode: pull-request**

Creates a separate PR for changelog updates (required for protected branches):

```yaml
name: Generate Changelog
on:
  push:
    branches: [main]

jobs:
  changelog:
    uses: jmackown/changelog_generator/.github/workflows/changelog.yml@main
    with:
      mode: 'pull-request'
    secrets:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

**Mode: update-branch**

Adds changelog entry to the PR branch when approved (cleanest option - changelog is part of the PR):

```yaml
name: Generate Changelog
on:
  pull_request_review:
    types: [submitted]

jobs:
  changelog:
    if: github.event.review.state == 'approved'
    uses: jmackown/changelog_generator/.github/workflows/changelog.yml@main
    with:
      mode: 'update-branch'
      pr-number: ${{ github.event.pull_request.number }}
    secrets:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

**Note:** With `update-branch` mode, there's potential for merge conflicts if multiple PRs touch the changelog simultaneously. This is rare if PRs are merged promptly after approval.

**All available options:**
```yaml
jobs:
  changelog:
    uses: jmackown/changelog_generator/.github/workflows/changelog.yml@main
    with:
      python-version: '3.11'           # Python version (default: 3.11)
      with-summaries: true             # AI summaries (default: true)
      model: 'claude-3-5-haiku-20241022'  # LLM model (auto-detected if not set)
      provider: ''                     # LLM provider: anthropic, openai, gemini (auto-detected from API keys)
      mode: 'direct-commit'            # How to commit: direct-commit, pull-request, update-branch
      pr-number: ''                    # PR number (required for update-branch mode)
    secrets:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
```

**Pin to specific version (recommended for stability):**
```yaml
jobs:
  changelog:
    uses: jmackown/changelog_generator/.github/workflows/changelog.yml@v1.0.0
    #                                                                     ^^^^^^
    #                                                                     Use git tag
```

## Custom Prompts

You can customize the AI summary prompt per repository by creating a `.changelog-prompt.txt` file in your repo root.

**Example `.changelog-prompt.txt`:**
```
Summarize this pull request for a non-technical audience. Focus on user-facing changes and business value.

{context}

Respond with 2-3 bullet points starting with "•". Keep each under 20 words.
```

The `{context}` placeholder is replaced with:
- PR Title
- PR Description (up to `--context-limit` characters)
- Files Changed

**Use cases:**
- Different tone (technical vs user-friendly)
- Different language
- Repo-specific instructions ("always mention API version affected")
- Different output format (categories, longer summaries, etc.)

If no `.changelog-prompt.txt` exists, the default prompt is used.

## Output Format

The generated changelog uses year-based sections with blockquote-style cards:

```markdown
# Changelog

## 2025

> ### 📅 2025-01-28 | Add user authentication ([#42](https://github.com/org/repo/pull/42))
> **Author:** @alice | **Approved:** @bob | **Ticket:** AUTH-123 | **Run:** [#123](https://github.com/org/repo/actions/runs/123)
>
> • Implemented JWT-based authentication for API endpoints
> • Added user registration and login flows
> • Integrated password hashing with bcrypt
> [abc1234](https://github.com/org/repo/commit/abc1234)

## 2024

> ### 📅 2024-12-15 | Initial release ([#1](https://github.com/org/repo/pull/1))
> **Author:** @alice
> [def5678](https://github.com/org/repo/commit/def5678)

---
*Generated on 2025-01-28 10:30:00*
```

## Advanced: Running Locally

For development, testing, or custom workflows, you can run the script directly.

### Requirements

- Python 3.11+
- Git repository with GitHub remote
- GitHub CLI (`gh`) installed and authenticated
- (Optional) Anthropic or OpenAI API key for AI summaries

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For AI summaries, set environment variable
export ANTHROPIC_API_KEY="your-key-here"
# OR
export OPENAI_API_KEY="your-key-here"

# Install and authenticate GitHub CLI (if not already)
brew install gh  # macOS
gh auth login
```

### Usage

**Default behavior** generates changelog from 2025-01-01 without AI summaries (fast, free):

```bash
# Generate from 2025 onwards (default)
python changelog/run.py

# Dry run to preview
python changelog/run.py --dry-run
```

**Other date ranges:**
```bash
# From specific date
python changelog/run.py --from-date 2024-01-01

# Between two commits
python changelog/run.py --between abc123 def456

# Recent N commits
python changelog/run.py --recent 50

# Incremental update (since last changelog commit)
python changelog/run.py --since-last-commit
```

**With AI summaries:**
```bash
# Generate with summaries (uses tokens)
python changelog/run.py --with-summaries

# Use specific provider
python changelog/run.py --with-summaries --provider gemini
```

### Backfilling Summaries

For existing changelogs, use the separate backfill script to add AI summaries without regenerating everything:

```bash
# Add summaries to entries from last 6 months
python changelog/backfill_summaries.py --since 2024-07-01

# Limit to 20 entries (to control token usage)
python changelog/backfill_summaries.py --limit 20

# Preview without writing
python changelog/backfill_summaries.py --dry-run --limit 10

# Use cheapest provider for bulk backfill
python changelog/backfill_summaries.py --provider gemini --limit 50
```

The backfill script:
- Reads existing `CHANGELOG.md`
- Finds entries without summaries
- Generates summaries using LLM (respects `.changelog-prompt.txt`)
- Updates the file in place

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dry-run` | Preview only, don't write files | `false` (writes by default) |
| `--from-date YYYY-MM-DD` | Generate from specific date | `2025-01-01` |
| `--between COMMIT1 COMMIT2` | Generate between two commits | - |
| `--recent N` | Generate from recent N commits (overrides default date) | - |
| `--since-last-commit` | Incremental update (for CI/CD) | - |
| `--for-pr NUMBER` | Generate entry for a specific PR (for update-branch mode) | - |
| `--with-summaries` | Generate AI summaries | `false` |
| `--model MODEL` | LLM model to use | Auto-detected from provider |
| `--provider NAME` | LLM provider (`anthropic`, `openai`, `gemini`) | Auto-detected from API keys |
| `--context-limit N` | Max chars of PR description for LLM context | 3000 |

## Troubleshooting

### "gh: command not found" in CI

GitHub Actions runners have `gh` pre-installed. If running locally without `gh`:
```bash
# Install GitHub CLI
brew install gh  # macOS
# Or visit: https://cli.github.com/

# Authenticate
gh auth login
```

### "ANTHROPIC_API_KEY not set" warning

Either:
1. Set the environment variable: `export ANTHROPIC_API_KEY="your-key"`
2. Add it to GitHub Secrets (see setup steps above)
3. Run without `--with-summaries` flag (or `with-summaries: false` in workflow)

### Workflow runs but doesn't create PR

Check:
1. Workflow permissions are set to "Read and write" in repo settings
2. Check **Allow GitHub Actions to create and approve pull requests** is enabled
3. There were actually new PRs since last changelog update (script generates changes)
4. Look in **Pull Requests** tab for the automated PR
5. Check the workflow logs for error messages

### PR created but doesn't auto-merge

Check:
1. **Settings > General > Pull Requests > Allow auto-merge** is enabled
2. All required status checks are passing
3. Branch protection rules don't block auto-merge
4. The PR has the auto-merge label/setting enabled

### Infinite loop of changelog PRs

The workflow includes `[skip ci]` in commit messages to prevent this. The PR approach naturally prevents loops because:
- The changelog PR merge triggers the workflow
- But no new PRs have been merged, so no new changelog entries
- No changes = no new PR created

If loops still happen:
1. Check that your CI respects `[skip ci]` commits
2. Add more skip patterns in the workflow's `if:` condition

## How It Works

1. **Fetch commits**: Uses `git log --first-parent --grep='(#'` to find only PR merge commits on main
2. **Enrich metadata**: Calls GitHub CLI (`gh pr list`, `gh run list`) to get author, approver, workflow data
3. **Extract ticket refs**: Parses commit messages for JIRA/ticket references (any PROJECT-123 pattern)
4. **Generate summaries** (optional): Sends PR context to Anthropic/OpenAI/Gemini API for bullet-point summaries
5. **Cache results**: Stores summaries in `.changelog-summaries.json` to avoid re-processing
6. **Format output**: Generates Markdown with clickable links and blockquote cards
7. **Write file**: Saves to `CHANGELOG.md` (unless `--dry-run` specified)
8. **Create PR**: Uses `peter-evans/create-pull-request` to open a PR with the changes
9. **Auto-merge** (optional): If enabled, PR automatically merges after checks pass

## Cost Considerations

AI summaries cost money:
- **Anthropic Claude Haiku**: ~$0.001 per PR summary (very cheap)
- **Anthropic Claude Sonnet**: ~$0.003 per PR summary (better quality)
- **OpenAI GPT-4o-mini**: ~$0.0001 per PR summary (cheapest)
- **Google Gemini Flash Lite**: Free tier available, then ~$0.0001 per PR summary

For 100 PRs/month with Haiku: ~$0.10/month. Caching prevents re-processing commits.

To minimize costs:
- Use `with-summaries: false` to disable AI entirely (free)
- Use `gemini-2.0-flash-lite` or `gpt-4o-mini` (cheapest options)
- Caching prevents re-generating summaries for already-processed commits

## License

MIT License - see LICENSE file for details.
