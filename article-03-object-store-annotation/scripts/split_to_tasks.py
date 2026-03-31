#!/usr/bin/env python3
"""
split_to_tasks.py

Converts a JSONL file (output of generate_qa_pairs.py) into individual
Label Studio task JSON files and uploads them to the MinIO exports bucket.

Each line of the JSONL becomes one task file, named task_NNNNN.json.
Label Studio's storage sync then picks them up from MinIO automatically.

Usage:
  python3 split_to_tasks.py \
    --input /data/fast/datasets/exports/synthetic_v1.jsonl \
    --prefix synthetic/v1/tasks \
    --minio-alias local

Requirements:
  pip install minio
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Split JSONL into Label Studio task files and upload to MinIO"
    )
    parser.add_argument("--input",       required=True,
                        help="Input JSONL file path")
    parser.add_argument("--prefix",      default="tasks",
                        help="MinIO key prefix within the exports bucket "
                             "(default: tasks)")
    parser.add_argument("--minio-alias", default="local",
                        help="mc alias name for your MinIO instance "
                             "(default: local)")
    parser.add_argument("--bucket",      default="exports",
                        help="MinIO bucket name (default: exports)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Parse and validate without uploading")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {args.input}")
        raise SystemExit(1)

    total   = 0
    skipped = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        with open(input_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  Skipping line {i}: {e}")
                    skipped += 1
                    continue

                # Build Label Studio task — wrap in list as LS expects
                task = [{
                    "context":  example.get("context",  ""),
                    "question": example.get("question", ""),
                    "answer":   example.get("answer",   ""),
                }]

                task_file = tmp / f"task_{i:05d}.json"
                task_file.write_text(
                    json.dumps(task, ensure_ascii=False, indent=2)
                )
                total += 1

        print(f"Prepared {total} task files ({skipped} skipped)")

        if args.dry_run:
            print("Dry run — no files uploaded.")
            return

        print(f"Uploading to {args.minio_alias}/{args.bucket}/{args.prefix}/...")
        result = subprocess.run(
            [
                "mc", "mirror", "--overwrite",
                str(tmp) + "/",
                f"{args.minio_alias}/{args.bucket}/{args.prefix}/",
            ],
            capture_output=False,
        )
        if result.returncode != 0:
            print("Error: mc mirror failed. Check your mc alias and bucket name.")
            raise SystemExit(1)

        print(f"Done. {total} tasks uploaded to "
              f"{args.minio_alias}/{args.bucket}/{args.prefix}/")
        print("Trigger a Label Studio storage sync to import them.")


if __name__ == "__main__":
    main()
