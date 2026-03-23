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

The easiest way to use this tool is as a **reusable workflow** — just add one small file to your repository and the changelog generates automatically.

### 1. Add Workflow File

Create `.github/workflows/changelog.yml` in your repository:

```yaml
name: Changelog

on:
  push:
    branches: [main]

jobs:
  changelog:
    if: "!contains(github.event.head_commit.message, '[skip ci]')"
    uses: jmackown/changelog_generator/.github/workflows/changelog.yml@main
    with:
      with-summaries: true
    secrets:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Replace `jmackown` with the GitHub username/org where this repo lives.

**That's it!** Every time a PR merges to main, a changelog entry is generated and committed to `CHANGELOG.md` automatically.

> See the [`examples/`](examples/) directory for ready-to-copy workflow files.

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

### How It Works

When a PR merges to main, the workflow:

1. Fires on the `push` event to the main branch
2. Extracts the PR number from the merge commit message (handles both merge commits and squash merges)
3. Fetches PR metadata via GitHub CLI: author, approver, JIRA ticket reference, workflow run link
4. Optionally generates an AI summary of the PR
5. Commits the new entry to `CHANGELOG.md` with `[skip ci]` to avoid re-triggering the workflow

Concurrent merges are serialised via a `concurrency` group so simultaneous pushes to main never produce conflicting changelog commits.

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

**All available options:**
```yaml
jobs:
  changelog:
    uses: jmackown/changelog_generator/.github/workflows/changelog.yml@main
    with:
      python-version: '3.11'              # Python version (default: 3.11)
      with-summaries: true                # AI summaries (default: true)
      model: 'claude-3-5-haiku-20241022'  # LLM model (auto-detected if not set)
      provider: ''                        # LLM provider: anthropic, openai, gemini (auto-detected from API keys)
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

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dry-run` | Preview only, don't write files | `false` (writes by default) |
| `--from-date YYYY-MM-DD` | Generate from specific date | `2025-01-01` |
| `--between COMMIT1 COMMIT2` | Generate between two commits | - |
| `--recent N` | Generate from recent N commits (overrides default date) | - |
| `--since-last-commit` | Incremental update (for CI/CD) | - |
| `--for-pr NUMBER` | Generate entry for a specific PR | - |
| `--with-summaries` | Generate AI summaries | `false` |
| `--model MODEL` | LLM model to use | Auto-detected from provider |
| `--provider NAME` | LLM provider (`anthropic`, `openai`, `gemini`) | Auto-detected from API keys |
| `--context-limit N` | Max chars of PR description for LLM context | 3000 |

## Backfilling Summaries

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

### Workflow runs but changelog doesn't update

Check:
1. Workflow permissions are set to "Read and write" in repo settings
2. There was actually a PR reference in the merge commit (see "No PR number found" below)
3. Check the workflow logs for error messages

### Infinite loop

The workflow commits to `CHANGELOG.md` with `[skip ci]` in the commit message to prevent re-triggering the workflow. If you see repeated runs:
1. Confirm your caller workflow has `if: "!contains(github.event.head_commit.message, '[skip ci]')"` on the job
2. Check that your branch settings don't strip commit message annotations

### No PR number found

The workflow extracts the PR number from the merge commit message. If a commit is pushed directly to main without a PR reference, no changelog entry is generated and the workflow exits cleanly.

To ensure entries are captured, always merge via a pull request rather than pushing directly to main.

## How It Works (Internals)

When the workflow runs for a given PR:

1. Fetches metadata for the PR via GitHub CLI (`gh pr view`)
2. Enriches with author, approver, and workflow run data
3. Extracts JIRA ticket references from the commit message and PR title
4. Optionally generates an AI summary
5. Writes the new entry to `CHANGELOG.md`, merging with existing content and preserving manually-written sections

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
