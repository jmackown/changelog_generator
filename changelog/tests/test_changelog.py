"""Unit tests for changelog generator."""

import re
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Import the module - we'll test the regex patterns and ChangeEntry
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from run import ChangeEntry, ChangelogGenerator


class TestExtractPrNumber:
    """Test PR number extraction from commit messages."""

    # The regex used in parse_commit: re.search(r"#(\d+)", subject)
    PR_PATTERN = re.compile(r"#(\d+)")

    def extract_pr_number(self, subject: str) -> str | None:
        """Extract PR number using the same logic as parse_commit."""
        match = self.PR_PATTERN.search(subject)
        return match.group(1) if match else None

    def test_squash_merge_format(self):
        """Standard squash merge: 'feat: add feature (#123)'"""
        assert self.extract_pr_number("feat: add feature (#123)") == "123"

    def test_regular_merge_format(self):
        """Regular merge: 'Merge pull request #456 from user/branch'"""
        assert self.extract_pr_number("Merge pull request #456 from user/branch") == "456"

    def test_pr_number_in_middle(self):
        """PR number in middle of message."""
        assert self.extract_pr_number("fix: resolve #789 issue with login") == "789"

    def test_multiple_pr_numbers_returns_first(self):
        """When multiple PR numbers exist, returns first match."""
        # Note: the actual code uses search() which returns first match
        assert self.extract_pr_number("feat (#123) related to #456") == "123"

    def test_no_pr_number(self):
        """Commit without PR number."""
        assert self.extract_pr_number("direct commit to main") is None

    def test_hash_without_number(self):
        """Hash symbol without following number."""
        assert self.extract_pr_number("fix: update # comments") is None

    def test_large_pr_number(self):
        """Large PR numbers work correctly."""
        assert self.extract_pr_number("feat: big repo (#99999)") == "99999"


class TestExtractJiraTicket:
    """Test JIRA ticket extraction from commit messages."""

    # The regex used in parse_commit: re.search(r"([A-Z]+-\d+)", subject)
    JIRA_PATTERN = re.compile(r"([A-Z]+-\d+)")

    def extract_jira_ticket(self, subject: str) -> str | None:
        """Extract JIRA ticket using the same logic as parse_commit."""
        match = self.JIRA_PATTERN.search(subject)
        return match.group(1) if match else None

    def test_standard_jira_ticket(self):
        """Standard JIRA format: 'ABC-123: fix bug'"""
        assert self.extract_jira_ticket("ABC-123: fix bug") == "ABC-123"

    def test_jira_in_middle(self):
        """JIRA ticket in middle of message."""
        assert self.extract_jira_ticket("fix: resolve PROJ-456 login issue") == "PROJ-456"

    def test_jira_at_end(self):
        """JIRA ticket at end of message."""
        assert self.extract_jira_ticket("feat: add feature (TEAM-789)") == "TEAM-789"

    def test_multiple_tickets_returns_first(self):
        """When multiple tickets exist, returns first match."""
        assert self.extract_jira_ticket("ABC-123 DEF-456: combined fix") == "ABC-123"

    def test_no_jira_ticket(self):
        """Commit without JIRA ticket."""
        assert self.extract_jira_ticket("feat: add feature (#123)") is None

    def test_lowercase_not_matched(self):
        """Lowercase project keys are not matched."""
        assert self.extract_jira_ticket("abc-123: fix bug") is None

    def test_single_letter_project(self):
        """Single letter project keys work."""
        assert self.extract_jira_ticket("X-1: minimal ticket") == "X-1"

    def test_long_project_key(self):
        """Long project keys work."""
        assert self.extract_jira_ticket("VERYLONGPROJECT-12345: big project") == "VERYLONGPROJECT-12345"


class TestTitleCleaning:
    """Test commit title cleaning logic."""

    def clean_title(self, subject: str) -> str:
        """Clean title using the same logic as parse_commit."""
        # Remove "(#123)" at end for squash merges
        title = re.sub(r"\s*\(#\d+\)\s*$", "", subject)
        # For regular merge commits, extract branch name
        merge_match = re.match(r"^Merge pull request #\d+ from \S+/(.+)$", subject)
        if merge_match:
            branch_name = merge_match.group(1)
            title = branch_name.replace("_", " ").replace("-", " ").title()
        return title.strip()

    def test_squash_merge_removes_pr_suffix(self):
        """Squash merge PR number suffix is removed."""
        assert self.clean_title("feat: add user auth (#123)") == "feat: add user auth"

    def test_squash_merge_with_spaces(self):
        """Handles extra spaces around PR number."""
        assert self.clean_title("fix: bug fix  (#456)  ") == "fix: bug fix"

    def test_regular_merge_extracts_branch(self):
        """Regular merge extracts and formats branch name."""
        result = self.clean_title("Merge pull request #789 from user/add-new-feature")
        assert result == "Add New Feature"

    def test_regular_merge_underscores(self):
        """Branch names with underscores are converted to spaces."""
        result = self.clean_title("Merge pull request #123 from org/fix_login_bug")
        assert result == "Fix Login Bug"

    def test_no_pr_number_unchanged(self):
        """Title without PR number is unchanged."""
        assert self.clean_title("feat: direct commit") == "feat: direct commit"

    def test_pr_in_middle_not_removed(self):
        """PR numbers in middle of title are not removed."""
        assert self.clean_title("fix #123 related to auth (#456)") == "fix #123 related to auth"


class TestChangeEntry:
    """Test ChangeEntry NamedTuple."""

    def test_create_minimal_entry(self):
        """Create entry with required fields only."""
        entry = ChangeEntry(
            commit_hash="abc123def456",
            short_hash="abc123d",
            title="feat: add feature",
            pr_number="123",
            jira_ticket=None,
            date=datetime(2025, 1, 15),
            author=None,
            approver=None,
            workflow_run_number=None,
            workflow_run_url=None,
            summary=None,
        )
        assert entry.commit_hash == "abc123def456"
        assert entry.pr_number == "123"
        assert entry.jira_ticket is None

    def test_create_full_entry(self):
        """Create entry with all fields populated."""
        entry = ChangeEntry(
            commit_hash="abc123def456",
            short_hash="abc123d",
            title="feat: add feature",
            pr_number="123",
            jira_ticket="PROJ-456",
            date=datetime(2025, 1, 15),
            author="octocat",
            approver="reviewer",
            workflow_run_number="789",
            workflow_run_url="https://github.com/org/repo/actions/runs/789",
            summary=["Added new feature", "Updated docs"],
        )
        assert entry.author == "octocat"
        assert entry.approver == "reviewer"
        assert len(entry.summary) == 2


class TestGenerateChangelogContent:
    """Test changelog markdown generation."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance with mocked git commands."""
        with patch.object(ChangelogGenerator, '_get_github_repo_url', return_value="https://github.com/test/repo"):
            with patch.object(ChangelogGenerator, '_load_cache', return_value={}):
                gen = ChangelogGenerator(dry_run=True)
                gen.repo_root = Path("/tmp/test")
                return gen

    def test_empty_entries_returns_placeholder(self, generator):
        """Empty entry list returns placeholder message."""
        content = generator.generate_changelog_content([])
        assert "No changes found" in content

    def test_single_entry_format(self, generator):
        """Single entry generates correct markdown structure."""
        entry = ChangeEntry(
            commit_hash="abc123def456789",
            short_hash="abc123d",
            title="feat: add login",
            pr_number="42",
            jira_ticket=None,
            date=datetime(2025, 1, 15),
            author="dev",
            approver="reviewer",
            workflow_run_number=None,
            workflow_run_url=None,
            summary=None,
        )
        content = generator.generate_changelog_content([entry])

        # Check blockquote structure
        assert "> ### 📅 2025-01-15" in content
        assert "feat: add login" in content
        assert "[#42]" in content
        assert "@dev" in content
        assert "@reviewer" in content

    def test_entry_with_summary_bullets(self, generator):
        """Entry with summary includes bullet points."""
        entry = ChangeEntry(
            commit_hash="abc123def456789",
            short_hash="abc123d",
            title="feat: add feature",
            pr_number="99",
            jira_ticket=None,
            date=datetime(2025, 1, 15),
            author=None,
            approver=None,
            workflow_run_number=None,
            workflow_run_url=None,
            summary=["First change", "Second change"],
        )
        content = generator.generate_changelog_content([entry])

        assert "> • First change" in content
        assert "> • Second change" in content

    def test_entry_with_jira_ticket(self, generator):
        """Entry with JIRA ticket includes ticket reference."""
        entry = ChangeEntry(
            commit_hash="abc123def456789",
            short_hash="abc123d",
            title="fix: resolve bug",
            pr_number="55",
            jira_ticket="PROJ-123",
            date=datetime(2025, 1, 15),
            author=None,
            approver=None,
            workflow_run_number=None,
            workflow_run_url=None,
            summary=None,
        )
        content = generator.generate_changelog_content([entry])

        assert "PROJ-123" in content
        assert "**Ticket:**" in content

    def test_entry_with_workflow_run(self, generator):
        """Entry with workflow run includes run link."""
        entry = ChangeEntry(
            commit_hash="abc123def456789",
            short_hash="abc123d",
            title="feat: add CI",
            pr_number="77",
            jira_ticket=None,
            date=datetime(2025, 1, 15),
            author=None,
            approver=None,
            workflow_run_number="1234",
            workflow_run_url="https://github.com/test/repo/actions/runs/1234",
            summary=None,
        )
        content = generator.generate_changelog_content([entry])

        assert "**Run:**" in content
        assert "[#1234]" in content

    def test_entries_sorted_by_date_newest_first(self, generator):
        """Multiple entries are sorted newest first."""
        old_entry = ChangeEntry(
            commit_hash="old123",
            short_hash="old123",
            title="old feature",
            pr_number="1",
            jira_ticket=None,
            date=datetime(2025, 1, 1),
            author=None,
            approver=None,
            workflow_run_number=None,
            workflow_run_url=None,
            summary=None,
        )
        new_entry = ChangeEntry(
            commit_hash="new456",
            short_hash="new456",
            title="new feature",
            pr_number="2",
            jira_ticket=None,
            date=datetime(2025, 1, 15),
            author=None,
            approver=None,
            workflow_run_number=None,
            workflow_run_url=None,
            summary=None,
        )
        # Pass in wrong order
        content = generator.generate_changelog_content([old_entry, new_entry])

        # New entry should appear first
        new_pos = content.find("new feature")
        old_pos = content.find("old feature")
        assert new_pos < old_pos, "Newer entry should appear before older entry"

    def test_commit_hash_link(self, generator):
        """Commit hash is linked to GitHub."""
        entry = ChangeEntry(
            commit_hash="abc123def456789",
            short_hash="abc123d",
            title="feat: test",
            pr_number="1",
            jira_ticket=None,
            date=datetime(2025, 1, 15),
            author=None,
            approver=None,
            workflow_run_number=None,
            workflow_run_url=None,
            summary=None,
        )
        content = generator.generate_changelog_content([entry])

        assert "[abc123d]" in content
        assert "github.com/test/repo/commit/abc123def456789" in content


class TestMergeWithExisting:
    """Test merging new entries with existing CHANGELOG.md."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create generator with temp directory."""
        with patch.object(ChangelogGenerator, '_get_github_repo_url', return_value="https://github.com/test/repo"):
            with patch.object(ChangelogGenerator, '_load_cache', return_value={}):
                gen = ChangelogGenerator(dry_run=True)
                gen.repo_root = tmp_path
                return gen

    def test_no_existing_file_creates_new(self, generator):
        """Without existing file, creates new changelog."""
        entry = ChangeEntry(
            commit_hash="abc123",
            short_hash="abc123",
            title="first commit",
            pr_number="1",
            jira_ticket=None,
            date=datetime(2025, 1, 15),
            author=None,
            approver=None,
            workflow_run_number=None,
            workflow_run_url=None,
            summary=None,
        )
        content = generator.generate_changelog_content([entry])

        assert "# Changelog" in content
        assert "## 2025" in content  # Year-based section
        assert "first commit" in content

    def test_merge_prepends_to_year_section(self, generator, tmp_path):
        """New entries are prepended to existing year section."""
        # Create existing changelog with year-based sections
        existing = """# Changelog

## 2025

> ### 📅 2025-01-10 | old entry ([#1](https://github.com/test/repo/pull/1))
> [old123](https://github.com/test/repo/commit/old123)

---
*Generated on 2025-01-10 10:00:00*
"""
        changelog_path = tmp_path / "CHANGELOG.md"
        changelog_path.write_text(existing)

        new_entry = ChangeEntry(
            commit_hash="new456",
            short_hash="new456",
            title="new entry",
            pr_number="2",
            jira_ticket=None,
            date=datetime(2025, 1, 15),
            author=None,
            approver=None,
            workflow_run_number=None,
            workflow_run_url=None,
            summary=None,
        )
        content = generator.generate_changelog_content([new_entry])

        # New entry should be present
        assert "new entry" in content
        # Old entry should be preserved
        assert "old entry" in content
        # New should come before old (newer date)
        new_pos = content.find("new entry")
        old_pos = content.find("old entry")
        assert new_pos < old_pos

    def test_updates_generation_timestamp(self, generator, tmp_path):
        """Generation timestamp is updated."""
        existing = """# Changelog

## 2025

---
*Generated on 2025-01-01 00:00:00*
"""
        changelog_path = tmp_path / "CHANGELOG.md"
        changelog_path.write_text(existing)

        entry = ChangeEntry(
            commit_hash="abc123",
            short_hash="abc123",
            title="test",
            pr_number="1",
            jira_ticket=None,
            date=datetime(2025, 1, 15),
            author=None,
            approver=None,
            workflow_run_number=None,
            workflow_run_url=None,
            summary=None,
        )
        content = generator.generate_changelog_content([entry])

        # Old timestamp should be replaced
        assert "2025-01-01 00:00:00" not in content
        assert "*Generated on" in content
