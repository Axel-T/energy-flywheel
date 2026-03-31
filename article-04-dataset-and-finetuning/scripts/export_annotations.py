#!/usr/bin/env python3
"""
export_annotations.py

Exports completed annotations from Label Studio and converts them to an
Alpaca-format JSONL training file for use with Unsloth / TRL.

Verdict logic:
  - "Correct"           → use original model answer as training target
  - "Partially correct" → use human correction as training target
  - "Wrong"             → use human correction as training target
  - "Wrong" with no correction → skip (incomplete annotation)

Usage:
  python3 export_annotations.py \
    --label-studio-url http://annotation-server-ip:8080 \
    --token your-label-studio-api-token \
    --project-id 1 \
    --output /data/fast/datasets/training/energy_qa_v1.jsonl

Requirements:
  pip install requests
"""

import argparse
import json
from pathlib import Path

import requests

SYSTEM_PROMPT = (
    "You are an expert home energy advisor with deep knowledge of "
    "photovoltaic solar systems, solar thermal collectors, and pellet "
    "heating. You are given a real-time sensor snapshot from a residential "
    "installation and a question from the homeowner. Provide accurate, "
    "practical guidance grounded in the data. Be specific — reference "
    "actual values from the snapshot rather than speaking in generalities."
)


def fetch_annotations(base_url: str, token: str, project_id: int) -> list:
    """Download all completed annotations from a Label Studio project."""
    url     = f"{base_url}/api/projects/{project_id}/export"
    headers = {"Authorization": f"Token {token}"}
    resp    = requests.get(url, headers=headers,
                           params={"exportType": "JSON"}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def extract_result(results: list, from_name: str) -> str | None:
    """Pull a single result value from a Label Studio annotation result list."""
    for r in results:
        if r.get("from_name") != from_name:
            continue
        value = r.get("value", {})
        # Choices return a list
        if "choices" in value and value["choices"]:
            return value["choices"][0]
        # TextArea returns a list of strings
        if "text" in value and value["text"]:
            return value["text"][0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Export Label Studio annotations to Alpaca JSONL"
    )
    parser.add_argument("--label-studio-url", required=True,
                        help="Label Studio base URL, e.g. http://host:8080")
    parser.add_argument("--token",            required=True,
                        help="Label Studio API token (Account → Access Token)")
    parser.add_argument("--project-id",       type=int, default=1,
                        help="Label Studio project ID (default: 1)")
    parser.add_argument("--output",           required=True,
                        help="Output JSONL file path")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching annotations from project {args.project_id}...")
    tasks = fetch_annotations(
        args.label_studio_url, args.token, args.project_id
    )
    print(f"  {len(tasks)} tasks retrieved")

    written = 0
    skipped = 0

    with open(out_path, "w") as out_f:
        for task in tasks:
            if not task.get("annotations"):
                skipped += 1
                continue

            annotation = task["annotations"][0]
            results    = annotation.get("result", [])

            verdict    = extract_result(results, "verdict")
            correction = extract_result(results, "correction")

            if verdict is None:
                skipped += 1
                continue

            # Skip incomplete wrong annotations
            if verdict == "Wrong" and not correction:
                skipped += 1
                continue

            data = task.get("data", {})

            # Choose the training target answer
            if verdict in ("Wrong", "Partially correct") and correction:
                answer = correction.strip()
            else:
                answer = data.get("answer", "").strip()

            if not answer:
                skipped += 1
                continue

            example = {
                "instruction": SYSTEM_PROMPT,
                "input": (
                    f"Sensor context:\n{data.get('context', '')}\n\n"
                    f"Question: {data.get('question', '')}"
                ),
                "output": answer,
            }

            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nExported {written} training examples ({skipped} skipped)")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
