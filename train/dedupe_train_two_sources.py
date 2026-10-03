# -*- coding: utf-8 -*-
"""Remove duplicate Train_two_new_mech source records by source priority.

The source files contain a record count on the first line, followed by lines
of ``<raw_score> <duo>``.  A duplicate key is formed from the two members of
the duo after the first whitespace-separated score token, sorted so that
reversing the two members is also treated as the same group.  The first source
in ``--files`` wins; duplicates within one source keep their first occurrence.

Default priority matches Train_two_new_mech.cpp's current inputs while giving
trate the requested precedence:

    trate.txt > ge4500_2.txt > lt4500_2.txt > ai-n0.txt

The default mode is a dry run.  Use ``--apply`` to rewrite the source files.
Use ``--backup-dir`` with ``--apply`` to retain the original files first.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_FILES = ["trate.txt", "ge4500_2.txt", "lt4500_2.txt", "ai-n0.txt"]


@dataclass
class Record:
    line_no: int
    raw_line: str
    key: str
    score_token: str


@dataclass
class Source:
    path: Path
    declared_count: int
    records: list[Record]
    newline: str


def parse_source(path: Path) -> Source:
    raw = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8-sig")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"empty source file: {path}")
    try:
        declared_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"first line is not an integer count: {path}: {lines[0]!r}") from exc

    records: list[Record] = []
    for line_no, line in enumerate(lines[1:], start=2):
        if not line.strip():
            raise ValueError(f"blank record line at {path}:{line_no}")
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError(f"expected '<score> <duo>' at {path}:{line_no}")
        score_token, duo = parts
        members = duo.strip().split("+")
        if len(members) != 2 or not all(member.strip() for member in members):
            raise ValueError(f"expected exactly two duo members at {path}:{line_no}")
        key = "+".join(sorted(member.strip() for member in members))
        records.append(Record(line_no, line, key, score_token))

    if declared_count != len(records):
        raise ValueError(
            f"declared count mismatch in {path}: header={declared_count}, records={len(records)}"
        )
    return Source(path, declared_count, records, newline)


def short_key(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def write_report(path: Path, removed: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "removed_file",
                "removed_line",
                "kept_file",
                "kept_line",
                "key_sha1_12",
                "duo",
            ]
        )
        for row in removed:
            writer.writerow(
                [
                    row["removed_file"],
                    row["removed_line"],
                    row["kept_file"],
                    row["kept_line"],
                    row["key_sha1_12"],
                    row["duo"],
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="source files in descending priority order",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="rewrite source files after validation; default is dry-run",
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="backup directory used before --apply (recommended)",
    )
    parser.add_argument(
        "--report",
        default="train_two_dedup_report.tsv",
        help="TSV report for every removed duplicate",
    )
    args = parser.parse_args()

    paths = [Path(value) for value in args.files]
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("--files contains the same path more than once")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error("missing source files: " + ", ".join(missing))

    sources = [parse_source(path) for path in paths]
    seen: dict[str, tuple[Path, int]] = {}
    kept: dict[Path, list[Record]] = {source.path: [] for source in sources}
    removed: list[dict[str, object]] = []

    for source in sources:
        for record in source.records:
            previous = seen.get(record.key)
            if previous is None:
                seen[record.key] = (source.path, record.line_no)
                kept[source.path].append(record)
                continue
            kept_path, kept_line = previous
            removed.append(
                {
                    "removed_file": str(source.path),
                    "removed_line": record.line_no,
                    "kept_file": str(kept_path),
                    "kept_line": kept_line,
                    "key_sha1_12": short_key(record.key),
                    "duo": record.key,
                }
            )

    write_report(Path(args.report), removed)

    print("[INFO] priority (highest first):")
    for rank, source in enumerate(sources, start=1):
        count = len(kept[source.path])
        print(
            f"  {rank}. {source.path}: original={source.declared_count}, "
            f"kept={count}, removed={source.declared_count - count}"
        )
    print(f"[INFO] unique duo groups kept: {len(seen)}")
    print(f"[INFO] duplicate records removed: {len(removed)}")
    print(f"[INFO] report: {args.report}")

    if not args.apply:
        print("[INFO] dry-run only; source files were not changed")
        return 0

    if args.backup_dir is None:
        parser.error("--apply requires --backup-dir so original source files are recoverable")
    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=False)
    for source in sources:
        shutil.copy2(source.path, backup_dir / source.path.name)

    for source in sources:
        output_lines = [str(len(kept[source.path]))]
        output_lines.extend(record.raw_line for record in kept[source.path])
        # Write bytes so Windows does not translate an already selected CRLF
        # into CRCRLF.  The source files are UTF-8 and may contain emoji.
        output_text = source.newline.join(output_lines) + source.newline
        source.path.write_bytes(output_text.encode("utf-8"))

    print(f"[INFO] source files rewritten; backups: {backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
