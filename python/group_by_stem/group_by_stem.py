#!/usr/bin/env python3
"""
Group related media/annotation files into base-stem-named directories.

Base stem rule:
- If filename (without extension) contains "_ed-", we take everything
  BEFORE "_ed-" as the grouping key.
- Otherwise we use the full stem.

Example group:
    Chilc_Botan_RMM302_itskwinpahwits-Solanaceae_2008-09-12-e.wav
    Chilc_Botan_RMM302_itskwinpahwits-Solanaceae_2008-09-12-e_ed-2020-01-16.eaf

Both end up in:
    root/
      Chilc_Botan_RMM302_itskwinpahwits-Solanaceae_2008-09-12-e/
        Chilc_Botan_RMM302_itskwinpahwits-Solanaceae_2008-09-12-e.wav
        Chilc_Botan_RMM302_itskwinpahwits-Solanaceae_2008-09-12-e_ed-2020-01-16.eaf

Singletons (only one file for a key) are left where they are.
"""

import argparse
import csv
import shutil
from pathlib import Path


def canonical_key(stem: str) -> str:
    """
    Compute a grouping key from a filename stem.

    If the stem contains "_ed-", strip everything from "_ed-" onward.
    Otherwise return the stem as-is.
    """
    marker = "_ed-"
    idx = stem.find(marker)
    if idx != -1:
        return stem[:idx]
    return stem


def collect_files_by_key(root: Path, extensions):
    """
    Walk `root` and collect all files with the given extensions,
    grouped by canonical key.
    """
    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in extensions}
    by_key = {}

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            key = canonical_key(path.stem)
            by_key.setdefault(key, []).append(path)

    return by_key


def move_files(root: Path, by_key: dict, dry_run: bool = False):
    """
    For each key with more than one file, create a directory under `root`
    named after the key and move all associated files into it.

    Returns a list of log rows describing the moves.
    """
    log_rows = []

    for key, files in sorted(by_key.items()):
        # Only bother if we actually have multiple related files
        if len(files) < 2:
            continue

        dest_dir = root / key

        if not dry_run:
            dest_dir.mkdir(exist_ok=True)

        for src in files:
            dest = dest_dir / src.name
            final_dest = dest
            counter = 1

            # Avoid overwriting anything that might already be there
            while final_dest.exists() and not dry_run:
                final_dest = dest_dir / f"{src.stem}_{counter}{src.suffix}"
                counter += 1

            if dry_run:
                print(f"[DRY RUN] Would move:\n  {src}\n  -> {final_dest}")
            else:
                shutil.move(str(src), str(final_dest))
                print(f"Moved:\n  {src}\n  -> {final_dest}")

            log_rows.append(
                {
                    "group_key": key,
                    "created_directory": str(dest_dir),
                    "original_path": str(src),
                    "new_path": str(final_dest),
                }
            )

    return log_rows


def write_log(root: Path, log_rows, dry_run: bool = False):
    """
    Write a CSV log to `move_log.csv` in the root directory.
    """
    log_path = root / "move_log.csv"

    if dry_run:
        print(f"[DRY RUN] Would write log to: {log_path}")
        return

    fieldnames = ["group_key", "created_directory", "original_path", "new_path"]
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nLog written to: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Group .wav/.eaf/.trs files by base stem into stem-named directories."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Path to the root directory that contains the files (default: current directory).",
    )
    parser.add_argument(
        "--extensions",
        "-e",
        nargs="+",
        default=[".wav", ".eaf", ".trs"],
        help="File extensions to include (default: .wav .eaf .trs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually moving files or writing the log.",
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()

    if not root.is_dir():
        raise SystemExit(f"Root path is not a directory: {root}")

    print(f"Scanning in: {root}")
    print(f"Looking for extensions: {', '.join(args.extensions)}\n")

    by_key = collect_files_by_key(root, args.extensions)

    if not by_key:
        print("No matching files found.")
        return

    print(f"Found {len(by_key)} grouping keys (before filtering singletons).\n")

    log_rows = move_files(root, by_key, dry_run=args.dry_run)
    write_log(root, log_rows, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run complete. No files were moved and no log was written.")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()