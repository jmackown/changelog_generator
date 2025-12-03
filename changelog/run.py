#!/usr/bin/env python3
"""
Changelog Generator - generates changelog from PRs merged to main branch.
Run with --dry-run to preview, or without flags to write CHANGELOG.md.
"""

import argparse
import subprocess
import re
import json
import os
import requests
from datetime import datetime
from typing import List, Optional, NamedTuple
from pathlib import Path


# Provider registry - auto-detects based on available API keys
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
    """Return the first provider with an API key set, or None."""
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
        # OpenAI-compatible format (OpenAI, Gemini)
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
        # OpenAI-compatible format
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "")


class ChangeEntry(NamedTuple):
    """Represents a single changelog entry."""

    commit_hash: str
    short_hash: str
    title: str
    pr_number: Optional[str]
    jira_ticket: Optional[str]
    date: datetime
    author: Optional[str]
    approver: Optional[str]
    workflow_run_number: Optional[str]
    workflow_run_url: Optional[str]
    summary: Optional[List[str]]


class ChangelogGenerator:
    """Generates changelog from git history."""

    def __init__(
        self,
        dry_run: bool = True,
        with_summaries: bool = False,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self.dry_run = dry_run
        self.with_summaries = with_summaries

        # Auto-detect provider if not specified
        self.provider = provider or detect_provider()
        # Use provider's default model if not specified
        if model:
            self.model = model
        elif self.provider:
            self.model = PROVIDERS[self.provider]["model"]
        else:
            self.model = None

        self.repo_root = Path.cwd()
        self.cache_file = self.repo_root / ".changelog-summaries.json"
        self._summary_cache = self._load_cache()
        self.github_repo_url = self._get_github_repo_url()

    def run_git_command(self, cmd: List[str]) -> str:
        """Run a git command and return output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_root,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error running git command {' '.join(cmd)}: {e}")
            print(f"Error output: {e.stderr}")
            return ""

    def _load_cache(self) -> dict:
        """Load summary cache from file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "r") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return {}

    def _save_cache(self) -> None:
        """Save summary cache to file."""
        if not self.dry_run:
            try:
                with open(self.cache_file, "w") as f:
                    json.dump(self._summary_cache, f, indent=2)
            except IOError as e:
                print(f"Warning: Could not save cache: {e}")

    def _get_github_repo_url(self) -> Optional[str]:
        """Extract GitHub repo URL from git remote."""
        try:
            cmd = ["git", "remote", "get-url", "origin"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_root,
            )
            remote_url = result.stdout.strip()

            # Convert SSH to HTTPS and clean up
            if remote_url.startswith("git@github.com:"):
                # git@github.com:user/repo.git -> https://github.com/user/repo
                repo_path = remote_url.replace("git@github.com:", "").replace(
                    ".git", ""
                )
                return f"https://github.com/{repo_path}"
            elif "github.com" in remote_url:
                # Already HTTPS, just clean up
                return remote_url.replace(".git", "")

            return None
        except subprocess.CalledProcessError:
            return None

    def get_commits_since_date(self, since_date: str) -> List[str]:
        """Get commit hashes since a specific date (main branch PRs only)."""
        cmd = [
            "git",
            "log",
            "--format=%H",
            "--first-parent",
            "main",  # Only commits directly on main
            f"--since={since_date}",
            "--grep=#",  # PR merge commits (both squash and regular merges)
        ]
        output = self.run_git_command(cmd)
        return output.split("\n") if output else []

    def get_commits_between(self, commit1: str, commit2: str) -> List[str]:
        """Get commits between two commit hashes (main branch PRs only)."""
        cmd = [
            "git",
            "log",
            "--format=%H",
            "--first-parent",
            "main",  # Only commits directly on main
            f"{commit1}..{commit2}",
        ]
        output = self.run_git_command(cmd)
        return output.split("\n") if output else []

    def get_recent_commits(self, count: int = 50) -> List[str]:
        """Get recent commits (main branch PRs only)."""
        cmd = [
            "git",
            "log",
            "--format=%H",
            "--first-parent",
            "main",  # Only commits directly on main
            f"-n{count}",
            "--grep=#",  # PR merge commits (both squash and regular merges)
        ]
        output = self.run_git_command(cmd)
        return output.split("\n") if output else []

    def get_last_changelog_commit(self) -> Optional[str]:
        """Find the last commit that modified CHANGELOG.md."""
        changelog_path = self.repo_root / "CHANGELOG.md"

        # Check if CHANGELOG.md exists
        if not changelog_path.exists():
            if not self.dry_run:
                print("No existing CHANGELOG.md found, will generate from beginning")
            return None

        # Get the last commit that modified CHANGELOG.md
        cmd = ["git", "log", "--format=%H", "-n1", "CHANGELOG.md"]
        output = self.run_git_command(cmd)

        if not output:
            if not self.dry_run:
                print("CHANGELOG.md exists but has no git history")
            return None

        return output.strip()

    def get_pr_details(
        self, commit_hash: str, pr_number: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Get PR author and approver using GitHub CLI."""
        try:
            # Get PR details by search commit SHA
            cmd = [
                "gh",
                "pr",
                "list",
                "--state",
                "merged",
                "--search",
                commit_hash,
                "--json",
                "author,reviews,number",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            prs = json.loads(result.stdout)

            if not prs:
                return None, None

            pr_data = prs[0]  # Should match by commit

            # Get author
            author = pr_data.get("author", {}).get("login")

            # Find approver (first approved review)
            approver = None
            reviews = pr_data.get("reviews", [])
            for review in reviews:
                if review.get("state") == "APPROVED":
                    approver = review.get("author", {}).get("login")
                    break

            return author, approver

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            if not self.dry_run:
                print(f"Warning: Could not get PR details for {commit_hash[:7]}: {e}")
            return None, None

    def get_workflow_run(self, commit_hash: str) -> tuple[Optional[str], Optional[str]]:
        """Get GitHub Actions workflow run number and URL for commit."""
        try:
            cmd = ["gh", "run", "list", "--json", "headSha,number,url,event"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            runs = json.loads(result.stdout)

            for run in runs:
                if run.get("headSha") == commit_hash and run.get("event") == "push":
                    return str(run.get("number")), run.get("url")

            return None, None

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            if not self.dry_run:
                print(f"Warning: Could not get workflow run for {commit_hash[:7]}: {e}")
            return None, None

    def get_pr_body(self, pr_number: str) -> Optional[str]:
        """Get PR description/body using GitHub CLI."""
        if not pr_number:
            return None

        try:
            cmd = ["gh", "pr", "view", pr_number, "--json", "body"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            pr_data = json.loads(result.stdout)
            return pr_data.get("body", "").strip()
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            if not self.dry_run:
                print(f"Warning: Could not get PR body for #{pr_number}: {e}")
            return None

    def get_file_stats(self, commit_hash: str) -> Optional[str]:
        """Get file change statistics for a commit."""
        try:
            cmd = ["git", "diff", "--stat", f"{commit_hash}^..{commit_hash}"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_root,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            if not self.dry_run:
                print(f"Warning: Could not get file stats for {commit_hash[:7]}: {e}")
            return None

    def get_llm_summary(
        self, commit_hash: str, pr_number: str, title: str
    ) -> Optional[List[str]]:
        """Generate LLM summary bullets for a PR using auto-detected provider."""
        if not self.with_summaries:
            return None

        # Check cache first
        if commit_hash in self._summary_cache:
            return self._summary_cache[commit_hash]

        if not self.provider:
            if not self.dry_run:
                print("  Warning: No LLM API key found, skipping summaries")
            return None

        # Get PR context
        pr_body = self.get_pr_body(pr_number) if pr_number else None
        file_stats = self.get_file_stats(commit_hash)

        if not pr_body and not file_stats:
            return None

        # Construct context for LLM
        context_parts = [f"PR Title: {title}"]
        if pr_body and len(pr_body) > 0:
            context_parts.append(f"PR Description:\n{pr_body[:1000]}")
        if file_stats:
            context_parts.append(f"Files Changed:\n{file_stats}")
        context = "\n\n".join(context_parts)

        prompt = f"""Please analyze this pull request and provide 2-4 concise bullet points summarizing what was changed and why. Focus on the functional changes and their purpose, not technical implementation details.

{context}

Please respond with only bullet points, starting each with "•". Keep each point under 25 words."""

        try:
            if not self.dry_run:
                print(f"  Generating {self.provider} summary for {commit_hash[:7]}...")

            endpoint, headers, body = build_llm_request(self.provider, self.model, prompt)
            response = requests.post(endpoint, headers=headers, json=body, timeout=30)

            if response.status_code == 200:
                summary_text = parse_llm_response(self.provider, response.json())

                # Parse bullet points
                bullets = []
                for line in summary_text.split("\n"):
                    line = line.strip()
                    if line.startswith("•") or line.startswith("-") or line.startswith("*"):
                        bullets.append(line[1:].strip())
                    elif line and not bullets:
                        bullets.append(line)

                # Cache the result
                summary = bullets[:4] if bullets else None
                self._summary_cache[commit_hash] = summary
                self._save_cache()
                return summary

            else:
                if not self.dry_run:
                    print(f"  Warning: {self.provider} API error {response.status_code}: {response.text[:100]}")
                return None

        except Exception as e:
            if not self.dry_run:
                print(f"  Warning: Could not generate {self.provider} summary for {commit_hash[:7]}: {e}")
            return None

    def parse_commit(self, commit_hash: str) -> Optional[ChangeEntry]:
        """Parse a commit into a ChangeEntry."""
        # Get commit details
        cmd = ["git", "log", "--format=%H%n%s%n%ad", "--date=iso", "-n1", commit_hash]
        output = self.run_git_command(cmd)

        if not output:
            return None

        lines = output.split("\n")
        if len(lines) < 3:
            return None

        full_hash = lines[0]
        subject = lines[1]
        date_str = lines[2]

        # Skip non-PR commits or commits we don't want
        if "#" not in subject:
            return None

        # Parse PR number - supports both squash merge "(#123)" and regular merge "#123"
        pr_match = re.search(r"#(\d+)", subject)
        pr_number = pr_match.group(1) if pr_match else None

        # Parse JIRA ticket (matches common pattern: PROJECT-123)
        # Looks for uppercase letters followed by dash and numbers
        jira_match = re.search(r"([A-Z]+-\d+)", subject)
        jira_ticket = jira_match.group(1) if jira_match else None

        # Clean up title (remove PR number for both squash and merge commits)
        # Remove "(#123)" at end for squash merges
        title = re.sub(r"\s*\(#\d+\)\s*$", "", subject)
        # For regular merge commits, extract branch name from "Merge pull request #123 from user/branch"
        merge_match = re.match(r"^Merge pull request #\d+ from \S+/(.+)$", subject)
        if merge_match:
            branch_name = merge_match.group(1)
            # Convert branch name to readable title (replace _ and - with spaces, capitalize)
            title = branch_name.replace("_", " ").replace("-", " ").title()
        title = title.strip()

        # Parse date
        try:
            date = datetime.fromisoformat(date_str.replace(" ", "T", 1).rstrip("Z"))
        except ValueError:
            date = datetime.now()

        # Get additional PR and workflow information
        if not self.dry_run:
            print(f"  Fetching details for {full_hash[:7]}...")

        author, approver = (
            self.get_pr_details(full_hash, pr_number) if pr_number else (None, None)
        )
        workflow_run_number, workflow_run_url = self.get_workflow_run(full_hash)
        summary = self.get_llm_summary(full_hash, pr_number, title)

        return ChangeEntry(
            commit_hash=full_hash,
            short_hash=full_hash[:7],
            title=title,
            pr_number=pr_number,
            jira_ticket=jira_ticket,
            date=date,
            author=author,
            approver=approver,
            workflow_run_number=workflow_run_number,
            workflow_run_url=workflow_run_url,
            summary=summary,
        )

    def generate_changelog_content(self, entries: List[ChangeEntry]) -> str:
        """Generate the changelog markdown content, merging with existing file."""
        if not entries:
            return "# Changelog\n\nNo changes found for the specified criteria.\n"

        # Sort entries by date (newest first)
        sorted_entries = sorted(entries, key=lambda x: x.date, reverse=True)

        # Read existing CHANGELOG.md if it exists
        changelog_path = self.repo_root / "CHANGELOG.md"
        existing_content = ""
        existing_entries_section = ""

        if changelog_path.exists():
            with open(changelog_path, "r") as f:
                existing_content = f.read()

            # Extract existing entries from [Unreleased] section
            # Find everything after "## [Unreleased]" and before the next "##" or "---"
            unreleased_match = re.search(
                r"## \[Unreleased\]\s*\n(.*?)(?=\n##|\n---|$)",
                existing_content,
                re.DOTALL
            )
            if unreleased_match:
                existing_entries_section = unreleased_match.group(1).strip()

        # Build new entries markdown
        new_entries_content = []

        for entry in sorted_entries:
            # Beautiful card-style blockquote format
            date_str = entry.date.strftime("%Y-%m-%d")

            # Create PR link if we have repo URL and PR number
            title_with_link = entry.title
            if entry.pr_number and self.github_repo_url:
                pr_link = f"{self.github_repo_url}/pull/{entry.pr_number}"
                title_with_link += f" ([#{entry.pr_number}]({pr_link}))"
            elif entry.pr_number:
                title_with_link += f" (#{entry.pr_number})"

            # Start the blockquote card
            new_entries_content.append(f"> ### 📅 {date_str} | {title_with_link}")

            # Add author, approver, ticket, and workflow run info
            details = []
            if entry.author:
                details.append(f"**Author:** @{entry.author}")
            if entry.approver:
                details.append(f"**Approved:** @{entry.approver}")
            if entry.jira_ticket:
                details.append(f"**Ticket:** {entry.jira_ticket}")
            if entry.workflow_run_number and entry.workflow_run_url:
                details.append(
                    f"**Run:** [#{entry.workflow_run_number}]({entry.workflow_run_url})"
                )
            elif entry.workflow_run_number:
                details.append(f"**Run:** #{entry.workflow_run_number}")

            if details:
                new_entries_content.append(f"> {' | '.join(details)}  ")

            # Add empty line before summary bullets
            if entry.summary:
                new_entries_content.append(">")
                for bullet in entry.summary:
                    new_entries_content.append(f"> • {bullet}  ")

            # Add commit hash at the end
            commit_link = (
                f"{self.github_repo_url}/commit/{entry.commit_hash}"
                if self.github_repo_url
                else None
            )
            if commit_link:
                new_entries_content.append(f"> [{entry.short_hash}]({commit_link})")
            else:
                new_entries_content.append(f"> [{entry.short_hash}]")

            new_entries_content.append("")  # Empty line between cards

        # Build final content
        if existing_content:
            # Replace the [Unreleased] section with merged entries
            new_entries_text = "\n".join(new_entries_content)

            # Combine new entries with existing entries
            combined_entries = new_entries_text
            if existing_entries_section:
                combined_entries = new_entries_text + "\n" + existing_entries_section

            # Replace the [Unreleased] section
            final_content = re.sub(
                r"(## \[Unreleased\]\s*\n)(.*?)(?=\n##|\n---|$)",
                r"\1" + combined_entries + "\n\n",
                existing_content,
                flags=re.DOTALL
            )

            # Update the generation timestamp at the bottom
            final_content = re.sub(
                r"\*Generated on .*?\*",
                f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
                final_content
            )

            return final_content
        else:
            # No existing file, create from scratch
            content = ["# Changelog\n"]
            content.append("## [Unreleased]\n")
            content.extend(new_entries_content)
            content.append("")
            content.append("---")
            content.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
            content.append("")
            return "\n".join(content)

    def write_changelog(self, content: str) -> None:
        """Write the changelog to file."""
        changelog_path = self.repo_root / "CHANGELOG.md"

        if self.dry_run:
            print(f"\n{'='*60}")
            print("DRY RUN - Would write to CHANGELOG.md:")
            print(f"{'='*60}")
            print(content)
            print(f"{'='*60}")
            print(f"File would be written to: {changelog_path}")
        else:
            with open(changelog_path, "w") as f:
                f.write(content)
            print(f"Changelog written to {changelog_path}")

    def _generate(self, commits: List[str], empty_msg: str) -> None:
        """Process commits and generate changelog."""
        if not commits or commits == [""]:
            print(empty_msg)
            return

        print(f"Found {len(commits)} commits to process.")
        entries = [e for c in commits if c and (e := self.parse_commit(c))]

        if not entries:
            print("No PR entries found.")
            return

        print(f"Processed {len(entries)} changelog entries.")
        self.write_changelog(self.generate_changelog_content(entries))

    def generate_from_date(self, since_date: str) -> None:
        """Generate changelog from a specific date."""
        print(f"Generating changelog since {since_date}...")
        self._generate(self.get_commits_since_date(since_date), "No commits found.")

    def generate_between_commits(self, commit1: str, commit2: str) -> None:
        """Generate changelog between two commits."""
        print(f"Generating changelog between {commit1} and {commit2}...")
        self._generate(self.get_commits_between(commit1, commit2), "No commits found.")

    def generate_recent(self, count: int = 20) -> None:
        """Generate changelog from recent commits."""
        print(f"Generating changelog from recent {count} PRs...")
        self._generate(self.get_recent_commits(count), "No commits found.")

    def generate_since_last_commit(self) -> None:
        """Generate changelog since the last CHANGELOG.md update."""
        last_commit = self.get_last_changelog_commit()
        if last_commit:
            print(f"Generating changelog since last update ({last_commit[:7]})...")
            commits = self.get_commits_between(last_commit, "HEAD")
        else:
            print("No previous changelog, generating from recent 50 PRs...")
            commits = self.get_recent_commits(50)
        self._generate(commits, "No new commits found.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate changelog from git history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate from recent commits (default: 20)
    python changelog/run.py --recent 30

    # Generate from specific date
    python changelog/run.py --from-date 2025-08-01

    # Generate between two commits
    python changelog/run.py --between 381e367 53d0f3d

    # Generate since last changelog update (for CI/CD)
    python changelog/run.py --since-last-commit --with-summaries

    # Preview without writing
    python changelog/run.py --dry-run
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only, don't write files",
    )

    parser.add_argument(
        "--from-date",
        type=str,
        help="Generate changelog from specific date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--between",
        nargs=2,
        metavar=("COMMIT1", "COMMIT2"),
        help="Generate changelog between two commits",
    )

    parser.add_argument(
        "--recent",
        type=int,
        default=20,
        help="Generate from recent N commits (default: 20)",
    )

    parser.add_argument(
        "--since-last-commit",
        action="store_true",
        help="Generate changelog since the last CHANGELOG.md update (useful for CI/CD)",
    )

    parser.add_argument(
        "--with-summaries",
        action="store_true",
        help="Generate LLM summaries for each PR (auto-detects provider from API keys)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model to use (default: auto-detected based on provider)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=list(PROVIDERS.keys()),
        default=None,
        help="LLM provider to use (default: auto-detect from available API keys)",
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN MODE: No files will be modified")

    generator = ChangelogGenerator(
        dry_run=args.dry_run,
        with_summaries=args.with_summaries,
        model=args.model,
        provider=args.provider,
    )

    if args.from_date:
        generator.generate_from_date(args.from_date)
    elif args.between:
        generator.generate_between_commits(args.between[0], args.between[1])
    elif args.since_last_commit:
        generator.generate_since_last_commit()
    else:
        generator.generate_recent(args.recent)


if __name__ == "__main__":
    main()