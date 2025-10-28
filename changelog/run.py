#!/usr/bin/env python3
"""
Changelog Generator

Generates a changelog from PRs merged to main branch only.
Safe to run with multiple dry-run and testing modes.

Usage:
    python changelog/run.py --dry-run
    python changelog/run.py --from-date 2025-08-01
    python changelog/run.py --between commit1 commit2
    python changelog/run.py --since-last-commit --write
"""

import argparse
import subprocess
import re
import sys
import json
import os
import requests
from datetime import datetime
from typing import List, Dict, Optional, NamedTuple
from pathlib import Path


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
    workflow_run_id: Optional[str]
    workflow_run_number: Optional[str]
    workflow_run_url: Optional[str]
    summary: Optional[List[str]]


class ChangelogGenerator:
    """Generates changelog from git history."""

    def __init__(
        self,
        dry_run: bool = True,
        with_summaries: bool = False,
        model: str = "gpt-4o-mini",
        use_openai: bool = False,
    ):
        self.dry_run = dry_run
        self.with_summaries = with_summaries
        self.model = model
        self.use_openai = use_openai
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

    def get_workflow_run(
        self, commit_hash: str
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Get GitHub Actions workflow run database ID, number, and URL for commit."""
        try:
            # Get all workflow runs and find matching commit
            cmd = ["gh", "run", "list", "--json", "headSha,databaseId,number,url,event"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            runs = json.loads(result.stdout)

            # Find push event run for this commit (merge to main)
            for run in runs:
                if run.get("headSha") == commit_hash and run.get("event") == "push":
                    return (
                        str(run.get("databaseId")),
                        str(run.get("number")),
                        run.get("url"),
                    )

            return None, None, None

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            if not self.dry_run:
                print(f"Warning: Could not get workflow run for {commit_hash[:7]}: {e}")
            return None, None, None

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
        """Generate LLM summary bullets for a PR (optional, requires API key)."""
        if not self.with_summaries:
            return None

        # Check cache first
        if commit_hash in self._summary_cache:
            return self._summary_cache[commit_hash]

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            if not self.dry_run:
                print("  Warning: ANTHROPIC_API_KEY not set, skipping LLM summaries")
            return None

        # Get PR context
        pr_body = self.get_pr_body(pr_number) if pr_number else None
        file_stats = self.get_file_stats(commit_hash)

        if not pr_body and not file_stats:
            return None

        # Construct context for LLM
        context_parts = [f"PR Title: {title}"]

        if pr_body and len(pr_body) > 0:
            context_parts.append(f"PR Description:\n{pr_body[:1000]}")  # Limit size

        if file_stats:
            context_parts.append(f"Files Changed:\n{file_stats}")

        context = "\n\n".join(context_parts)

        # Create prompt for summary
        prompt = f"""Please analyze this pull request and provide 2-4 concise bullet points summarizing what was changed and why. Focus on the functional changes and their purpose, not technical implementation details.

{context}

Please respond with only bullet points, starting each with "•". Keep each point under 15 words."""

        try:
            # Minimal Claude API call
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

            data = {
                "model": self.model,
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
            }

            if not self.dry_run:
                print(f"  Generating LLM summary for {commit_hash[:7]}...")

            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                summary_text = result.get("content", [{}])[0].get("text", "")

                # Parse bullet points
                bullets = []
                for line in summary_text.split("\n"):
                    line = line.strip()
                    if (
                        line.startswith("•")
                        or line.startswith("-")
                        or line.startswith("*")
                    ):
                        bullets.append(line[1:].strip())
                    elif (
                        line and not bullets
                    ):  # First non-empty line if no bullets found
                        bullets.append(line)

                # Cache the result
                summary = bullets[:4] if bullets else None
                self._summary_cache[commit_hash] = summary
                self._save_cache()

                return summary

            else:
                if not self.dry_run:
                    print(
                        f"  Warning: LLM API error {response.status_code}: {response.text[:100]}"
                    )
                return None

        except Exception as e:
            if not self.dry_run:
                print(
                    f"  Warning: Could not generate LLM summary for {commit_hash[:7]}: {e}"
                )
            return None

    def get_llm_summary_openai(
        self, commit_hash: str, pr_number: str, title: str
    ) -> Optional[List[str]]:
        """Generate LLM summary using OpenAI API."""
        if not self.with_summaries or not self.use_openai:
            return None

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            if not self.dry_run:
                print("  Warning: OPENAI_API_KEY not set, skipping LLM summaries")
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

Please respond with only bullet points, starting each with "•". Keep each point under 15 words."""

        try:
            if not self.dry_run:
                print(f"  Generating OpenAI summary for {commit_hash[:7]}...")

            # Simple OpenAI API call
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.3,
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                summary_text = (
                    result.get("choices", [{}])[0].get("message", {}).get("content", "")
                )

                # Parse bullet points
                bullets = []
                for line in summary_text.split("\n"):
                    line = line.strip()
                    if (
                        line.startswith("•")
                        or line.startswith("-")
                        or line.startswith("*")
                    ):
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
                    print(
                        f"  Warning: OpenAI API error {response.status_code}: {response.text[:100]}"
                    )
                return None

        except Exception as e:
            if not self.dry_run:
                print(
                    f"  Warning: Could not generate OpenAI summary for {commit_hash[:7]}: {e}"
                )
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
        workflow_run_id, workflow_run_number, workflow_run_url = self.get_workflow_run(
            full_hash
        )

        # Choose LLM method based on configuration
        if self.use_openai:
            summary = self.get_llm_summary_openai(full_hash, pr_number, title)
        else:
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
            workflow_run_id=workflow_run_id,
            workflow_run_number=workflow_run_number,
            workflow_run_url=workflow_run_url,
            summary=summary,
        )

    def generate_changelog_content(self, entries: List[ChangeEntry]) -> str:
        """Generate the changelog markdown content."""
        if not entries:
            return "# Changelog\n\nNo changes found for the specified criteria.\n"

        # Sort entries by date (newest first)
        sorted_entries = sorted(entries, key=lambda x: x.date, reverse=True)

        # Build simple markdown
        content = ["# Changelog\n"]
        content.append("## [Unreleased]\n")

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
            content.append(f"> ### 📅 {date_str} | {title_with_link}")

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
                content.append(f"> {' | '.join(details)}  ")

            # Add empty line before summary bullets
            if entry.summary:
                content.append(">")
                for bullet in entry.summary:
                    content.append(f"> • {bullet}  ")

            # Add commit hash at the end
            commit_link = (
                f"{self.github_repo_url}/commit/{entry.commit_hash}"
                if self.github_repo_url
                else None
            )
            if commit_link:
                content.append(f"> [{entry.short_hash}]({commit_link})")
            else:
                content.append(f"> [{entry.short_hash}]")

            content.append("")  # Empty line between cards

        content.append("")  # Empty line

        # Add generation info
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

    def generate_from_date(self, since_date: str) -> None:
        """Generate changelog from a specific date."""
        print(f"Generating changelog for PRs merged to main since {since_date}...")

        commits = self.get_commits_since_date(since_date)
        if not commits or commits == [""]:
            print("No commits found since that date.")
            return

        print(f"Found {len(commits)} commits to process.")

        entries = []
        for commit in commits:
            if not commit:
                continue
            entry = self.parse_commit(commit)
            if entry:
                entries.append(entry)

        print(f"Processed {len(entries)} changelog entries.")

        content = self.generate_changelog_content(entries)
        self.write_changelog(content)

    def generate_between_commits(self, commit1: str, commit2: str) -> None:
        """Generate changelog between two commits."""
        print(
            f"Generating changelog for PRs merged to main between {commit1} and {commit2}..."
        )

        commits = self.get_commits_between(commit1, commit2)
        if not commits or commits == [""]:
            print("No commits found between those commits.")
            return

        print(f"Found {len(commits)} commits to process.")

        entries = []
        for commit in commits:
            if not commit:
                continue
            entry = self.parse_commit(commit)
            if entry:
                entries.append(entry)

        print(f"Processed {len(entries)} changelog entries.")

        content = self.generate_changelog_content(entries)
        self.write_changelog(content)

    def generate_recent(self, count: int = 20) -> None:
        """Generate changelog from recent commits."""
        print(f"Generating changelog from recent {count} PRs merged to main...")

        commits = self.get_recent_commits(count)
        if not commits or commits == [""]:
            print("No recent commits found.")
            return

        print(f"Found {len(commits)} commits to process.")

        entries = []
        for commit in commits:
            if not commit:
                continue
            entry = self.parse_commit(commit)
            if entry:
                entries.append(entry)

        print(f"Processed {len(entries)} changelog entries.")

        content = self.generate_changelog_content(entries)
        self.write_changelog(content)

    def generate_since_last_commit(self) -> None:
        """Generate changelog since the last CHANGELOG.md update."""
        last_commit = self.get_last_changelog_commit()

        if last_commit:
            print(
                f"Generating changelog for PRs merged to main since last changelog update ({last_commit[:7]})..."
            )
            commits = self.get_commits_between(last_commit, "HEAD")
        else:
            # No previous changelog, generate from recent history
            print(
                "No previous changelog found, generating from recent 50 PRs merged to main..."
            )
            commits = self.get_recent_commits(50)

        if not commits or commits == [""]:
            print("No new commits found since last changelog update.")
            return

        print(f"Found {len(commits)} commits to process.")

        entries = []
        for commit in commits:
            if not commit:
                continue
            entry = self.parse_commit(commit)
            if entry:
                entries.append(entry)

        if not entries:
            print("No new PR entries found.")
            return

        print(f"Processed {len(entries)} changelog entries.")

        content = self.generate_changelog_content(entries)
        self.write_changelog(content)


def main():
    parser = argparse.ArgumentParser(
        description="Generate changelog from git history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run (safe, shows output without writing)
    python changelog/run.py --dry-run

    # Generate from specific date
    python changelog/run.py --from-date 2025-08-01

    # Generate between two commits
    python changelog/run.py --between 381e367 53d0f3d

    # Generate from recent commits (default: 20)
    python changelog/run.py --recent 30

    # Generate since last changelog update (for CI/CD)
    python changelog/run.py --since-last-commit --write --with-summaries

    # Actually write the file (remove --dry-run)
    python changelog/run.py --from-date 2025-08-01 --write
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be generated without writing files (default)",
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write the changelog file (overrides --dry-run)",
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
        help="Generate LLM summaries for each PR (requires OPENAI_API_KEY or ANTHROPIC_API_KEY env var)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use for summaries (default: gpt-4o-mini for OpenAI, claude-3-haiku-20240307 for Anthropic)",
    )

    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI API instead of Anthropic (requires OPENAI_API_KEY env var)",
    )

    args = parser.parse_args()

    # Determine if we should actually write
    should_write = args.write and not args.dry_run
    if args.write:
        print("WRITE MODE: Files will be modified!")
    else:
        print("DRY RUN MODE: No files will be modified (safe)")

    generator = ChangelogGenerator(
        dry_run=not should_write,
        with_summaries=args.with_summaries,
        model=args.model,
        use_openai=args.use_openai,
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