#!/usr/bin/env python3
"""
generate_qa_pairs.py

Generates synthetic question-answer pairs from InfluxDB sensor history
using a local llama.cpp model. Outputs a JSONL file suitable for upload
to the MinIO exports/ bucket and subsequent annotation in Label Studio.

The script queries random timestamps from the past HISTORY_DAYS days,
assembles the same JSON context block used at inference time, and asks a
local language model to generate a plausible homeowner question and a
correct, data-grounded answer.

Usage:
  python3 generate_qa_pairs.py \
    --count 500 \
    --output /data/fast/datasets/exports/synthetic_v1.jsonl

Requirements:
  pip install influxdb-client
  llama.cpp built and a GGUF model downloaded (see Article 4)
"""

import argparse
import json
import os
import random
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from influxdb_client import InfluxDBClient

# ── Configuration — edit these to match your setup ─────────────────────────────

INFLUX_URL   = "http://metrics-server-ip:8086"
INFLUX_TOKEN = "your-influxdb-admin-token"
INFLUX_ORG   = "homelab"

LLAMA_BIN    = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
LLAMA_MODEL  = os.path.expanduser(
    "~/llama-models/Llama-3.2-8B-Instruct-Q4_K_M.gguf"
)

# Sampling window — how far back to draw random timestamps from
HISTORY_DAYS = 365 * 3   # 3 years

# ── System prompt ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a home energy expert helping the owner of a house with a "
    "photovoltaic solar system, solar thermal collectors, and a pellet "
    "heating system. Your answers are technically precise, grounded in "
    "the sensor data provided, and written in plain English for a "
    "non-expert homeowner."
)

GENERATION_PROMPT_TEMPLATE = (
    "Given the following sensor snapshot from a home energy system, "
    "generate ONE realistic question that the homeowner might ask, and "
    "provide a correct, detailed answer grounded in the data.\n\n"
    "Sensor snapshot:\n{context}\n\n"
    "Output format — respond with valid JSON only, no other text:\n"
    '{{\n  "question": "...",\n  "answer": "..."\n}}\n\n'
    "Focus on practical guidance: should the homeowner run an appliance "
    "now, is consumption in an expected range, are there any anomalies "
    "worth investigating?"
)

# ── InfluxDB helpers ────────────────────────────────────────────────────────────

def fetch_snapshot(client: InfluxDBClient, ts: datetime) -> dict | None:
    """Fetch a sensor context snapshot for a given timestamp."""
    query_api = client.query_api()
    w_start   = (ts - timedelta(minutes=5)).isoformat()
    w_end     = ts.isoformat()

    def last_val(bucket, measurement, field):
        q = (
            f'from(bucket: "{bucket}")'
            f"  |> range(start: {w_start}Z, stop: {w_end}Z)"
            f'  |> filter(fn: (r) => r._measurement == "{measurement}"'
            f'       and r._field == "{field}")'
            f"  |> last()"
        )
        for table in query_api.query(q, org=INFLUX_ORG):
            for record in table.records:
                return record.get_value()
        return None

    def day_sum(bucket, measurement, field):
        day_start = ts.replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
        q = (
            f'from(bucket: "{bucket}")'
            f"  |> range(start: {day_start}Z, stop: {w_end}Z)"
            f'  |> filter(fn: (r) => r._measurement == "{measurement}"'
            f'       and r._field == "{field}")'
            f"  |> sum()"
        )
        for table in query_api.query(q, org=INFLUX_ORG):
            for record in table.records:
                return record.get_value()
        return None

    def rolling_avg(bucket, measurement, field, days, agg="mean"):
        start = (ts - timedelta(days=days)).isoformat()
        q = (
            f'from(bucket: "{bucket}")'
            f"  |> range(start: {start}Z, stop: {w_end}Z)"
            f'  |> filter(fn: (r) => r._measurement == "{measurement}"'
            f'       and r._field == "{field}")'
            f"  |> aggregateWindow(every: 1d, fn: {agg}, createEmpty: false)"
            f"  |> mean()"
        )
        for table in query_api.query(q, org=INFLUX_ORG):
            for record in table.records:
                return record.get_value()
        return None

    snapshot = {
        "generated_at": ts.isoformat(),
        "solarpv": {
            "current_power_w":   last_val("solarpv", "solarpv", "ac_power_w"),
            "yield_today_kwh":   day_sum("solarpv", "solarpv", "yield_kwh"),
            "yield_7d_avg_kwh":  rolling_avg("solarpv", "solarpv",
                                             "yield_kwh", 7, "max"),
            "yield_30d_avg_kwh": rolling_avg("solarpv", "solarpv",
                                             "yield_kwh", 30, "max"),
        },
        "solarthermie": {
            "collector_temp_c": last_val("solarthermie", "solarthermie",
                                         "collector_temp_c"),
            "buffer_top_c":     last_val("solarthermie", "solarthermie",
                                         "buffer_temp_top_c"),
            "buffer_mid_c":     last_val("solarthermie", "solarthermie",
                                         "buffer_temp_mid_c"),
            "buffer_bottom_c":  last_val("solarthermie", "solarthermie",
                                         "buffer_temp_bottom_c"),
        },
        "heizung": {
            "boiler_temp_c":      last_val("heizung", "heizung",
                                           "boiler_temp_c"),
            "pellets_today_kg":   day_sum("heizung", "heizung",
                                          "pellets_consumed_kg"),
            "pellets_7d_avg_kg":  rolling_avg("heizung", "heizung",
                                              "pellets_consumed_kg", 7, "sum"),
            "pellets_30d_avg_kg": rolling_avg("heizung", "heizung",
                                              "pellets_consumed_kg", 30, "sum"),
        },
        "weather": {
            "outdoor_temp_c": last_val("solarpv", "solarpv", "outdoor_temp_c"),
        },
    }

    # Discard snapshots with too many null fields
    null_count = sum(
        1
        for sub in snapshot.values()
        if isinstance(sub, dict)
        for v in sub.values()
        if v is None
    )
    return None if null_count > 4 else snapshot


# ── LLM generation ──────────────────────────────────────────────────────────────

def call_llama(prompt: str) -> str | None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False) as f:
        f.write(prompt)
        prompt_file = f.name
    try:
        result = subprocess.run(
            [
                LLAMA_BIN,
                "-m", LLAMA_MODEL,
                "-f", prompt_file,
                "-n", "512",
                "--temp", "0.7",
                "--top-p", "0.9",
                "--repeat-penalty", "1.1",
                "--ctx-size", "4096",
                "--no-display-prompt",
                "-t", str(max(1, os.cpu_count() - 2)),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.stdout.strip() or None
    except subprocess.TimeoutExpired:
        return None
    finally:
        Path(prompt_file).unlink(missing_ok=True)


def extract_json(text: str) -> dict | None:
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def random_timestamp() -> datetime:
    now   = datetime.now(timezone.utc)
    delta = timedelta(
        days=random.randint(0, HISTORY_DAYS),
        hours=random.randint(6, 20),
        minutes=random.randint(0, 59),
    )
    return now - delta


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic QA pairs from InfluxDB history"
    )
    parser.add_argument("--count",  type=int, default=200,
                        help="Number of QA pairs to generate (default: 200)")
    parser.add_argument("--output", type=str,
                        default="/tmp/synthetic_qa.jsonl",
                        help="Output JSONL file path")
    args = parser.parse_args()

    client    = InfluxDBClient(url=INFLUX_URL,
                               token=INFLUX_TOKEN, org=INFLUX_ORG)
    out_path  = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generated    = 0
    attempts     = 0
    max_attempts = args.count * 5

    print(f"Generating {args.count} QA pairs → {out_path}")
    print(f"InfluxDB: {INFLUX_URL} | Model: {LLAMA_MODEL}")

    with open(out_path, "w") as out_f:
        while generated < args.count and attempts < max_attempts:
            attempts += 1
            ts = random_timestamp()

            snapshot = fetch_snapshot(client, ts)
            if snapshot is None:
                continue

            context_str = json.dumps(snapshot, indent=2)
            prompt = (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"<|user|>\n"
                + GENERATION_PROMPT_TEMPLATE.format(context=context_str)
                + "\n<|assistant|>\n"
            )

            raw = call_llama(prompt)
            if raw is None:
                continue

            qa = extract_json(raw)
            if qa is None or "question" not in qa or "answer" not in qa:
                continue

            example = {
                "context":   context_str,
                "question":  qa["question"].strip(),
                "answer":    qa["answer"].strip(),
                "source":    "synthetic",
                "timestamp": ts.isoformat(),
            }
            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
            generated += 1

            if generated % 10 == 0:
                print(f"  {generated}/{args.count} "
                      f"({attempts} attempts, "
                      f"{generated/attempts*100:.0f}% success)")

    client.close()
    print(f"\nDone. {generated} pairs written to {out_path}")
    print(f"Success rate: {generated/attempts*100:.1f}% "
          f"({attempts} attempts)")


if __name__ == "__main__":
    main()
