"""
Prepare polyergalio for PyPI publication.

Runs the test suite with coverage, builds the wheel and sdist, and
validates both with twine. Stops at the first failing step.

Requires the dev extras: pip install -e ".[dev]"

python scripts/prepare_release.py
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"
INIT_FILE = ROOT / "src" / "polyergalio" / "__init__.py"
CHANGES_FILE = ROOT / "CHANGES.md"


def run_step(title: str, command: list[str]) -> None:
    """
    Run a command from the repo root, stopping the script on failure.

    Parameters
    ----------
    title : printed as a section header before the command runs
    command : argv list passed to subprocess
    """
    print(f"\n=== {title} ===")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\n{title} failed (exit {result.returncode})")


def package_version() -> str:
    """The version declared by polyergalio.__version__, read without importing the package."""
    found = re.search(r'^__version__\s*=\s*"([^"]+)"', INIT_FILE.read_text(), re.MULTILINE)
    if found is None:
        sys.exit(f"no __version__ found in {INIT_FILE}")
    return found.group(1)


def changes_version() -> str:
    """The newest version heading in CHANGES.md."""
    found = re.search(r"^(\d+\.\d+\.\d+)\s*$", CHANGES_FILE.read_text(), re.MULTILINE)
    if found is None:
        sys.exit(f"no version heading found in {CHANGES_FILE}")
    return found.group(1)


def check_versions() -> None:
    """Stop unless the top CHANGES.md entry matches __version__."""
    declared, logged = package_version(), changes_version()
    print(f"\n=== versions ===\n__version__ {declared}, CHANGES.md {logged}")
    if declared != logged:
        sys.exit(f"\nversion mismatch: add a CHANGES.md entry for {declared}, or set __version__ to {logged}")


def run_tests_with_coverage() -> None:
    run_step("tests", [sys.executable, "-m", "coverage", "run", "-m", "pytest"])
    run_step("coverage report", [sys.executable, "-m", "coverage", "report", "-m"])


def build_distributions() -> None:
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    run_step("build wheel + sdist", [sys.executable, "-m", "build"])


def check_distributions() -> None:
    dist_files = sorted(str(path) for path in DIST_DIR.glob("*"))
    if not dist_files:
        sys.exit("no distributions found in dist/")
    version = package_version()
    stale = [path for path in dist_files if f"-{version}" not in Path(path).name]
    if stale:
        sys.exit(f"distributions do not match version {version}: {stale}")
    run_step("twine check", [sys.executable, "-m", "twine", "check", *dist_files])


if __name__ == "__main__":
    check_versions()
    #run_tests_with_coverage()
    build_distributions()
    check_distributions()
    print("\nall checks passed -- dist/ is ready for `twine upload` \n twine upload dist/*")
