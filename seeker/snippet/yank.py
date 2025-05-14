#date: 2025-05-14T17:04:01Z
#url: https://api.github.com/gists/71552817e67a765321fb5769e5ee69d5
#owner: https://api.github.com/users/daveio

#!/usr/bin/env python3
"""
yank-all - Fetch and pull all git repositories in the current directory in parallel.
Pretty UI with rich progress display.
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
import math
import importlib.util
import signal
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple



def handle_rich_install_error(error_prefix, exception):
    """Handle errors during rich installation with consistent error reporting"""
    print(f"{error_prefix}{exception}")
    print("Please install rich manually with: pip install rich")
    sys.exit(1)

def ensure_rich_installed():
    """Check if rich is installed, and install it if not."""
    if importlib.util.find_spec("rich") is not None:
        return

    print("The 'rich' package is not installed. Installing it now...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
        print("Rich successfully installed.")
    except subprocess.CalledProcessError as e:
        handle_rich_install_error('Error installing rich: ', e)
    except Exception as e:
        handle_rich_install_error('Unexpected error installing rich: ', e)

    # Force reload modules to make the newly installed rich available
    if "rich" in sys.modules:
        del sys.modules["rich"]
    if "rich.console" in sys.modules:
        del sys.modules["rich.console"]


# Ensure rich is installed before attempting to import it
ensure_rich_installed()

# Now import rich modules
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich import box

# Status enums
class Status(Enum):
    WAITING = "waiting"
    FETCHING = "fetching"
    PULLING = "pulling"
    DONE = "done"
    ERROR = "error"

# Status colors
STATUS_STYLES = {
    Status.WAITING: "blue",
    Status.FETCHING: "yellow",
    Status.PULLING: "magenta",  # Changed from green to magenta
    Status.DONE: "green",      # Changed from magenta to green
    Status.ERROR: "red",
}

# Display names for each status
STATUS_NAMES = {
    Status.WAITING: "Waiting",
    Status.FETCHING: "Fetching",
    Status.PULLING: "Pulling",
    Status.DONE: "Completed",
    Status.ERROR: "Error",
}

class GitRepo:
    """Represents a git repository with status tracking"""
    def __init__(self, path: Path):
        self.path = path.absolute()  # Store absolute path
        self.name = path.name
        self.status = Status.WAITING
        self.fetch_status = Status.WAITING
        self.pull_status = Status.WAITING
        self.error_message: Optional[str] = None
        self.output_message: Optional[str] = None

    def __str__(self) -> str:
        return self.name

    def get_status_renderable(self) -> Text:
        """Get a compact representation of the repo status for the grid view"""
        # Determine the overall status color
        if self.error_message:
            color = STATUS_STYLES[Status.ERROR]
        elif self.status == Status.DONE:
            color = STATUS_STYLES[Status.DONE]
        elif self.pull_status == Status.PULLING:
            color = STATUS_STYLES[Status.PULLING]
        elif self.fetch_status == Status.FETCHING:
            color = STATUS_STYLES[Status.FETCHING]
        else:
            color = STATUS_STYLES[Status.WAITING]

        # Create the text object with appropriate color
        return Text(f"{self.name}", style=color)

class GitRepoManager:
    """Manages multiple git repositories with parallel operations"""
    def __init__(self, max_workers: int = 8, use_grid: bool = True):
        self.console = Console()
        self.repos: List[GitRepo] = []
        self.max_workers = max_workers
        self.use_grid = use_grid
        # Store the original directory
        self.original_dir = Path.cwd().absolute()
        self.executor = None
        self._scan_repos()

    def _scan_repos(self) -> None:
        """Find all git repositories in the current directory"""
        for item in Path(".").iterdir():
            if not item.is_dir():
                continue

            git_dir = item / ".git"
            if git_dir.exists() and git_dir.is_dir():
                self.repos.append(GitRepo(item))

        # Sort repos by name for consistent display
        self.repos.sort(key=lambda x: x.name)

    def _make_status_key(self) -> Panel:
        """Create a color key panel to explain the colors"""
        table = Table(box=None, show_header=False, padding=(0, 1, 0, 1))
        table.add_column("Color", style="bold")
        table.add_column("Meaning")

        for status in Status:
            color = STATUS_STYLES[status]
            name = STATUS_NAMES[status]
            table.add_row(Text("■", style=color), Text(name))

        return Panel(
            table,
            title="[bold]Color Key[/bold]",
            border_style="dim"
        )

    def _make_grid_view(self) -> Panel:
        """Create a grid view of repositories"""
        renderables = [repo.get_status_renderable() for repo in self.repos]

        # Calculate appropriate number of columns based on terminal width
        terminal_width = self.console.width

        # Calculate estimated item width based on actual repo names
        max_name_length = max((len(repo.name) for repo in self.repos), default=15)
        estimated_item_width = max_name_length + 6  # Add padding

        # Calculate columns to fit terminal with some margin
        available_width = terminal_width - 10  # Leave some margin
        num_columns = max(1, min(4, available_width // estimated_item_width))

        # Create columns with the correct parameter name
        columns = Columns(
            renderables,
            equal=True,           # Equal width columns
            expand=True,          # Expand to fill available width
            padding=(0, 2)        # Add horizontal padding between columns
        )

        # Count status types
        waiting = sum(1 for repo in self.repos if repo.status == Status.WAITING)
        fetching = sum(1 for repo in self.repos if repo.fetch_status == Status.FETCHING)
        pulling = sum(1 for repo in self.repos if repo.pull_status == Status.PULLING)
        done = sum(1 for repo in self.repos if repo.status == Status.DONE)
        error = sum(1 for repo in self.repos if repo.status == Status.ERROR)

        # Create status summary
        status_text = f"Total: {len(self.repos)} | "
        if waiting > 0:
            status_text += f"Starting: {waiting} | "
        if fetching > 0:
            status_text += f"Fetching: {fetching} | "
        if pulling > 0:
            status_text += f"Pulling: {pulling} | "
        if done > 0:
            status_text += f"Completed: {done} | "
        if error > 0:
            status_text += f"Error: {error}"

        # Remove trailing separator if needed
        status_text = status_text.rstrip(" | ")

        return Panel(
            columns,
            title="[bold blue]Git Repository Updates[/bold blue]",
            subtitle=f"[italic]{status_text}[/italic]"
        )

    def _make_color_key(self) -> Panel:
        """Create a color key table"""
        table = Table(box=None, show_header=False, padding=(0, 2), expand=True)

        # Add color columns
        for status in Status:
            table.add_column(style=f"bold {STATUS_STYLES[status]}", justify="center")

        # Add status name row
        table.add_row(*[STATUS_NAMES[status] for status in Status])

        # Wrap in a panel to match style of main display
        return Panel(
            table,
            border_style="dim",
            expand=True,
        )

    def _make_status_table(self) -> Panel:
        """Create a rich table with repository status"""
        table = Table(box=box.ROUNDED, expand=True)
        table.add_column("Repository", style="bold white")
        table.add_column("Status", justify="center")
        table.add_column("Details", style="dim")

        for repo in self.repos:
            # Determine which status to show
            if repo.error_message:
                status = Status.ERROR
                status_text = Text("Error", style=STATUS_STYLES[status])
                details = Text(f"{repo.error_message}", style="red")
            elif repo.status == Status.DONE:
                status = Status.DONE
                status_text = Text("Completed", style=STATUS_STYLES[status])
                details = Text(repo.output_message if repo.output_message else "")
            elif repo.pull_status == Status.PULLING:
                status = Status.PULLING
                status_text = Text("Pulling", style=STATUS_STYLES[status])
                details = Text("")
            elif repo.fetch_status == Status.FETCHING:
                status = Status.FETCHING
                status_text = Text("Fetching", style=STATUS_STYLES[status])
                details = Text("")
            else:
                status = Status.WAITING
                status_text = Text("Starting", style=STATUS_STYLES[status])
                details = Text("")

            # Add the row to the table with the repo name colored by its status
            table.add_row(
                Text(f"{repo.name}", style=STATUS_STYLES[status]),
                status_text,
                details
            )

        return Panel(
            table,
            title="[bold blue]Git Repository Updates[/bold blue]",
            subtitle=f"[italic]{len(self.repos)} repositories[/italic]",
            expand=True,
        )

    def _set_repo_error(self, repo: GitRepo, error_msg: str, context: str = "") -> None:
        """Set error status on a repository with proper logging"""
        if context:
            error_text = f"[ERROR] {repo.name} ({context}): {error_msg}"
        else:
            error_text = f"[ERROR] {repo.name}: {error_msg}"

        self.console.print(error_text)

        repo.status = Status.ERROR
        repo.error_message = error_msg
        if repo.fetch_status == Status.WAITING:
            repo.fetch_status = Status.ERROR
        if repo.pull_status == Status.WAITING:
            repo.pull_status = Status.ERROR

    def _run_git_command(self, repo: GitRepo, cmd: List[str], operation: str) -> bool:
        """Run a git command and handle results and errors consistently"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(repo.path)
            )
            if result.stdout.strip():
                repo.output_message = result.stdout.strip()
            return True
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() or f"{operation} failed"
            self._set_repo_error(repo, error_msg, operation.lower())
            return False
        except Exception as e:
            error_msg = f"Unexpected error during {operation.lower()}: {str(e)}"
            self._set_repo_error(repo, error_msg)
            return False

    def _process_repo(self, repo: GitRepo) -> None:
        """Process a single repository (fetch and pull)"""
        try:
            # Verify the repository path exists
            if not os.path.exists(str(repo.path)) or not os.path.isdir(str(repo.path)):
                error_msg = "Repository directory not found or not accessible"
                self._set_repo_error(repo, error_msg)
                return

            # Verify it's actually a git repository
            git_dir = os.path.join(str(repo.path), ".git")
            if not os.path.exists(git_dir) or not os.path.isdir(git_dir):
                error_msg = "Not a git repository (missing .git directory)"
                self._set_repo_error(repo, error_msg)
                return

            # Fetch
            repo.fetch_status = Status.FETCHING
            fetch_cmd = ["git", "fetch", "--quiet", "--all", "--tags", "--prune",
                        "--jobs=8", "--recurse-submodules=yes"]
            if not self._run_git_command(repo, fetch_cmd, "Fetch"):
                return
            repo.fetch_status = Status.DONE

            # Pull
            repo.pull_status = Status.PULLING
            pull_cmd = ["git", "pull", "--quiet", "--tags", "--prune",
                       "--jobs=8", "--recurse-submodules=yes", "--rebase"]
            if self._run_git_command(repo, pull_cmd, "Pull"):
                repo.pull_status = Status.DONE
                repo.status = Status.DONE

        except Exception as e:
            # Catch any other exceptions
            error_msg = f"Unexpected error: {str(e)}"
            self._set_repo_error(repo, error_msg)

    def run(self) -> None:
        """Run updates on all repositories in parallel"""
        if not self.repos:
            self.console.print("[yellow]No git repositories found in the current directory.[/yellow]")
            return

        # Print initial information
        self.console.print(f"[bold blue]Found {len(self.repos)} repositories[/bold blue]")
        self.console.print(f"[cyan]Using {self.max_workers} worker threads[/cyan]")

        # Set up signal handler for graceful termination
        original_sigint_handler = signal.getsignal(signal.SIGINT)

        def handle_sigint(sig, frame):
            """Handle Ctrl+C by terminating subprocesses and exiting gracefully"""
            if self.executor:
                self.console.print("\n[bold yellow]Terminating... Please wait for subprocesses to clean up[/bold yellow]")
                self.executor.shutdown(wait=False, cancel_futures=True)
            signal.signal(signal.SIGINT, original_sigint_handler)  # Restore original handler
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_sigint)

        # Function to get the appropriate display with color key
        def get_display():
            # Only use grid for many repos or when explicitly requested
            if self.use_grid or len(self.repos) > 15:
                main_display = self._make_grid_view()
            else:
                main_display = self._make_status_table()

            color_key = self._make_color_key()

            # Combine main display with color key
            table = Table.grid(padding=0)
            table.add_column(justify="center")
            table.add_row(main_display)
            table.add_row(color_key)
            return table

        # Set up live display
        try:
            with Live(get_display(), refresh_per_second=1) as live:
                # Process repos in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    self.executor = executor
                    # Start all tasks
                    future_to_repo = {
                        executor.submit(self._process_repo, repo): repo
                        for repo in self.repos
                    }

                    # Update display as tasks complete
                    for future in concurrent.futures.as_completed(future_to_repo):
                        repo = future_to_repo[future]
                        try:
                            future.result()  # This will propagate any exceptions
                        except Exception as e:
                            repo.status = Status.ERROR
                            repo.error_message = f"Unexpected error: {str(e)}"

                        # Update the live display
                        live.update(get_display())

                    self.executor = None
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            self.console.print("\n[bold yellow]Operation canceled by user[/bold yellow]")
            return
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint_handler)

        # Final statistics
        success_count = sum(1 for repo in self.repos if repo.status == Status.DONE)
        error_count = sum(1 for repo in self.repos if repo.status == Status.ERROR)

        self.console.print(f"\n[bold green]{success_count} repositories updated successfully[/bold green]")
        if error_count:
            self.console.print(f"[bold red]{error_count} repositories had errors[/bold red]")


def main():
    parser = argparse.ArgumentParser(description="Update all git repositories in the current directory")
    parser.add_argument(
        "threads",
        nargs="?",
        default="8",
        help="Number of threads to use (default: 8, use 'unlimited' for one thread per repo)"
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Force table view instead of grid view"
    )
    args = parser.parse_args()

    # Determine max workers
    if args.threads.lower() == "unlimited":
        # Will be adjusted based on repo count
        max_workers = 9999
    else:
        try:
            max_workers = int(args.threads)
            if max_workers < 1:
                raise ValueError("Thread count must be at least 1")
        except ValueError:
            print(f"Error: Invalid thread count '{args.threads}'")
            sys.exit(1)

    # Run the manager
    manager = GitRepoManager(max_workers=max_workers, use_grid=not args.table)
    manager.run()


if __name__ == "__main__":
    main()
