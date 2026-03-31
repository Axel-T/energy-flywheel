#!/usr/bin/env bash
# setup_buckets.sh
# Creates the MinIO bucket structure and policies for the energy flywheel pipeline.
# Run once after MinIO is installed and the mc alias is configured.
#
# Prerequisites:
#   mc alias set local http://localhost:9000 admin <your-password>
#
# Usage:
#   bash setup_buckets.sh

set -euo pipefail

ALIAS="local"

echo "==> Creating buckets..."
mc mb --ignore-existing "$ALIAS/exports"
mc mb --ignore-existing "$ALIAS/datasets"
mc mb --ignore-existing "$ALIAS/adapters"
mc mb --ignore-existing "$ALIAS/models"

echo "==> Setting lifecycle rule on exports/ (90-day expiry)..."
mc ilm rule add --expire-days 90 "$ALIAS/exports"

echo "==> Creating IAM policies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mc admin policy create "$ALIAS" edge-device-policy \
  "$SCRIPT_DIR/edge-device-policy.json"
mc admin policy create "$ALIAS" labelstudio-policy \
  "$SCRIPT_DIR/labelstudio-policy.json"

echo ""
echo "Done. Bucket layout:"
mc ls "$ALIAS"
echo ""
echo "Next steps:"
echo "  1. Create users:  mc admin user add $ALIAS <username> <password>"
echo "  2. Attach policy: mc admin policy attach $ALIAS <policy> --user <username>"
echo "  See Article 3 for the full user setup."
