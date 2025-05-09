#!/usr/bin/env python3
import os
import subprocess
import sys

# === Configuration ===
PROJECT_ROOT = "./ragtest"
INPUT_DIR = f"{PROJECT_ROOT}/input"
QUERY_TEXT = "Tell me general theme of the data"


def cli_init(root: str):
    """
    Scaffold a new GraphRAG project (creates .env and settings.yaml).
    """
    subprocess.run(
        ["graphrag", "init", "--root", root],
        check=True
    )  # :contentReference[oaicite:4]{index=4}


def cli_index(root: str):
    """
    Index all documents under the project root using default parameters.
    """
    subprocess.run(
        ["graphrag", "index", "--root", root],
        check=True
    )  # :contentReference[oaicite:5]{index=5}


def cli_query(root: str, method: str, query: str):
    """
    Run a query with the specified method: global, local, or drift.
    """
    subprocess.run(
        [
            "graphrag", "query",
            "--root", root,
            "--method", method,
            "--query", query
        ],
        check=True
    )  # :contentReference[oaicite:6]{index=6}


def main_init():
    # 1. Initialize the project (only needed once)
    print("Initializing GraphRAG project…")
    cli_init(PROJECT_ROOT)  # :contentReference[oaicite:7]{index=7}


def main():
    # 1. Initialize (only if not already done)
    settings_path = os.path.join(PROJECT_ROOT, "settings.yaml")
    if not os.path.exists(settings_path):
        print("Initializing GraphRAG project…")
        try:
            cli_init(PROJECT_ROOT)
        except subprocess.CalledProcessError as e:
            print(f"[INIT ERROR] '{e.cmd}' exited with code {e.returncode}", file=sys.stderr)
            sys.exit(e.returncode)
    else:
        print("Project already initialized, skipping init.")

    # 2. Index documents
    print("Indexing documents…")
    try:
        cli_index(PROJECT_ROOT)
    except subprocess.CalledProcessError as e:
        print(f"[INDEX ERROR] '{e.cmd}' exited with code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

    # 3. Query using each method
    for method in ("global", "local", "drift"):
        print(f"\nRunning {method.upper()} search…")
        try:
            cli_query(PROJECT_ROOT, method, QUERY_TEXT)
        except subprocess.CalledProcessError as e:
            print(f"[QUERY ERROR] '{e.cmd}' exited with code {e.returncode}", file=sys.stderr)
            sys.exit(e.returncode)


if __name__ == "__main__":
    try:
        main()

    except subprocess.CalledProcessError as e:
        print(f"Error during '{e.cmd}': exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
