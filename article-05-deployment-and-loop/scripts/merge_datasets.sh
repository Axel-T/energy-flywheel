#!/usr/bin/env bash
# merge_datasets.sh
#
# Combines all JSONL dataset versions from MinIO into a single deduplicated
# training file. Run before each new training cycle to include all historical
# examples alongside newly annotated ones.
#
# Deduplication is by MD5 hash of each JSON line, so identical examples
# from multiple export batches are counted only once.
#
# Usage:
#   bash merge_datasets.sh [--alias <mc-alias>] [--output <path>]
#
# Options:
#   --alias   mc alias for MinIO (default: local)
#   --output  local output path (default: /tmp/energy_qa_combined.jsonl)

set -euo pipefail

MINIO_ALIAS="local"
OUTPUT="/tmp/energy_qa_combined.jsonl"
STAGING="/tmp/dataset-merge-staging"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --alias)  MINIO_ALIAS="$2"; shift 2 ;;
    --output) OUTPUT="$2";       shift 2 ;;
    *)        echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "==> Pulling all dataset versions from MinIO..."
rm -rf "$STAGING"
mkdir -p "$STAGING"
mc mirror "$MINIO_ALIAS/datasets/training/" "$STAGING/"

echo "==> Merging and deduplicating..."
python3 << PYEOF
import hashlib, json, glob, os

staging = "$STAGING"
output  = "$OUTPUT"

seen   = set()
output_lines = []

for path in sorted(glob.glob(os.path.join(staging, "**/*.jsonl"),
                              recursive=True)):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h = hashlib.md5(line.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                output_lines.append(line)

with open(output, "w") as f:
    f.write("\n".join(output_lines) + "\n")

print(f"Combined dataset: {len(output_lines):,} unique examples")
print(f"Output: {output}")
PYEOF

echo "==> Uploading combined dataset to MinIO..."
mc cp "$OUTPUT" \
  "$MINIO_ALIAS/datasets/training/energy_qa_combined.jsonl"

echo ""
echo "Done. Use energy_qa_combined.jsonl as the --data argument for"
echo "train_energy_adapter.py in your next training run."
echo ""
echo "To rsync to cloud GPU:"
echo "  rsync -avz --progress $OUTPUT user@cloud-gpu-ip:/workspace/data/"
