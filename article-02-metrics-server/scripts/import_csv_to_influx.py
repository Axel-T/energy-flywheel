#!/usr/bin/env python3
"""
import_csv_to_influx.py

Imports historical sensor data from a CSV file into InfluxDB v2.

CSV format expected:
  timestamp,field_name,value
  2022-01-01T08:00:00+00:00,yield_kwh,12.4
  2022-01-01T08:00:00+00:00,ac_power_w,3840.0

Usage:
  python3 import_csv_to_influx.py \
    --file solarpv_history.csv \
    --bucket solarpv \
    --measurement solarpv

Requirements:
  pip install influxdb-client
"""

import argparse
import csv
from datetime import datetime
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

INFLUX_URL    = "http://metrics-server-ip:8086"
INFLUX_TOKEN  = "your-influxdb-admin-token"
INFLUX_ORG    = "homelab"
BATCH_SIZE    = 5000


def main():
    parser = argparse.ArgumentParser(
        description="Import CSV sensor history into InfluxDB v2"
    )
    parser.add_argument("--file",        required=True,
                        help="Path to input CSV file")
    parser.add_argument("--bucket",      required=True,
                        help="InfluxDB target bucket name")
    parser.add_argument("--measurement", required=True,
                        help="InfluxDB measurement name")
    parser.add_argument("--url",   default=INFLUX_URL,
                        help="InfluxDB URL (default: %(default)s)")
    parser.add_argument("--token", default=INFLUX_TOKEN,
                        help="InfluxDB admin token")
    parser.add_argument("--org",   default=INFLUX_ORG,
                        help="InfluxDB organisation (default: %(default)s)")
    args = parser.parse_args()

    client = InfluxDBClient(
        url=args.url, token=args.token, org=args.org
    )
    write_api = client.write_api(write_options=SYNCHRONOUS)

    batch   = []
    written = 0
    skipped = 0

    print(f"Importing {args.file} → bucket={args.bucket} "
          f"measurement={args.measurement}")

    with open(args.file, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                ts    = datetime.fromisoformat(row["timestamp"])
                field = row["field_name"].strip()
                value = float(row["value"])
            except (KeyError, ValueError) as e:
                skipped += 1
                continue

            point = (
                Point(args.measurement)
                .tag("source", "import")
                .field(field, value)
                .time(ts, WritePrecision.SECONDS)
            )
            batch.append(point)

            if len(batch) >= BATCH_SIZE:
                write_api.write(bucket=args.bucket, record=batch)
                written += len(batch)
                batch    = []
                print(f"  {written:,} rows written...")

    if batch:
        write_api.write(bucket=args.bucket, record=batch)
        written += len(batch)

    client.close()
    print(f"\nDone. {written:,} rows written, {skipped} skipped.")


if __name__ == "__main__":
    main()
