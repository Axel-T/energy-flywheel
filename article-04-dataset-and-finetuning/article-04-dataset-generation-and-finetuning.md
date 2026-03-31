# Building an AI Data Flywheel for Home Energy Management
## Part 4 of 5: Dataset Generation and Fine-Tuning

*This is the fourth article in a five-part click-along series. Articles 1–3 established the concept, the metrics server, the object store, and the annotation infrastructure. This article puts all of that to work: we generate synthetic training data from real sensor history, curate it through Label Studio, and fine-tune Qwen2.5-14B-Instruct on a rented cloud GPU using QLoRA. By the end you will have a trained LoRA adapter ready for deployment.*

---

### What we are building in this article

By the end of this article you will have:

- A Python script that generates synthetic question-answer pairs from your InfluxDB history using llama.cpp running locally on the annotation server
- A Label Studio export script that converts completed annotations into a JSONL training file
- The training dataset uploaded to your MinIO `datasets` bucket and rsynced to a cloud GPU instance
- A fully trained QLoRA adapter — approximately 263MB — stored in your MinIO `adapters` bucket

If you already have a curated JSONL dataset you want to use directly, skip to **Step 5: Preparing the cloud GPU instance** — the synthetic generation steps are optional if you have sufficient real annotation data.

---

### Why generate synthetic data at all?

When the flywheel is first spun up, there are no model outputs to annotate — the model has not been deployed yet. You need training data to train the model, but the model needs to exist before it can generate the failures that become training data. This is the cold-start problem.

Synthetic data generation breaks the deadlock. The approach is to use the same InfluxDB history that will eventually inform the model's real-time context, and ask a general-purpose language model — running locally on the annotation server via llama.cpp — to produce plausible question-answer pairs grounded in that data. These synthetic pairs are not perfect, but they are good enough to give the fine-tuning run a meaningful starting point. Once the model is deployed and generating real responses, the synthetic data is progressively displaced by corrected real interactions.

There is a second reason to keep the synthetic generation script running even after the cold start: diversity. Real user queries cluster around a handful of patterns — the questions people actually ask day to day. Synthetic generation can deliberately probe under-represented scenarios: unusual weather conditions, edge cases in pellet consumption, thermal buffer states that occur only a few times per year. A training set that includes these cases produces a more robust model than one built entirely from the questions a single user happened to ask.

---

### Step 1: Install llama.cpp on the annotation server

llama.cpp is used on the annotation server for two purposes: generating synthetic QA pairs during dataset creation, and providing a fast local inference check before uploading data to the cloud. It runs on CPU — the annotation server does not need a GPU for this workload.

```bash
# On the annotation server
sudo apt update && sudo apt install -y \
  build-essential cmake git python3-pip python3-venv

# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_CURL=ON
cmake --build build --config Release -j$(nproc)

# Verify the build
./build/bin/llama-cli --version
```

Download a capable general-purpose model for generation. A 7B or 8B instruct model at Q4_K_M quantisation is sufficient — it runs comfortably on CPU with 16–32GB RAM, though slowly. The generation script runs offline overnight so speed is not critical.

```bash
# Create a models directory
mkdir -p ~/llama-models

# Download Llama-3.2-8B-Instruct Q4_K_M (~4.9GB)
# Use the Hugging Face Hub CLI or wget from a direct GGUF source
pip install huggingface-hub

huggingface-cli download \
  bartowski/Llama-3.2-8B-Instruct-GGUF \
  --include "Llama-3.2-8B-Instruct-Q4_K_M.gguf" \
  --local-dir ~/llama-models/
```

Test that inference works before proceeding:

```bash
./build/bin/llama-cli \
  -m ~/llama-models/Llama-3.2-8B-Instruct-Q4_K_M.gguf \
  -p "What is the capital of France?" \
  -n 64 --temp 0
# Should produce: Paris
```

---

### Step 2: The synthetic QA generation script

This script queries InfluxDB for historical snapshots at random timestamps, formats each snapshot as the context block defined in Article 2, and asks the local llama.cpp model to generate a plausible question a homeowner might ask — along with a correct answer grounded in the data.

Save this as `generate_qa_pairs.py` on the annotation server:

```python
#!/usr/bin/env python3
"""
generate_qa_pairs.py

Generates synthetic question-answer pairs from InfluxDB sensor history.
Outputs a JSONL file suitable for upload to MinIO exports/ bucket.

Usage:
  python3 generate_qa_pairs.py \
    --count 500 \
    --output /data/fast/datasets/exports/synthetic_v1.jsonl
"""

import argparse
import json
import random
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from influxdb_client import InfluxDBClient

# ── Configuration ──────────────────────────────────────────────────────────────

INFLUX_URL    = "http://metrics-server-ip:8086"
INFLUX_TOKEN  = "your-influxdb-admin-token"
INFLUX_ORG    = "homelab"

LLAMA_BIN     = "/home/user/llama.cpp/build/bin/llama-cli"
LLAMA_MODEL   = "/home/user/llama-models/Llama-3.2-8B-Instruct-Q4_K_M.gguf"

# How far back to sample timestamps from
HISTORY_DAYS  = 365 * 3   # 3 years

# ── InfluxDB helpers ────────────────────────────────────────────────────────────

def fetch_snapshot(client: InfluxDBClient, ts: datetime) -> dict:
    """Fetch a sensor context snapshot for a given timestamp."""
    query_api = client.query_api()
    window_start = (ts - timedelta(minutes=5)).isoformat()
    window_end   = ts.isoformat()

    def last_value(bucket: str, measurement: str, field: str) -> float | None:
        q = f'''
        from(bucket: "{bucket}")
          |> range(start: {window_start}Z, stop: {window_end}Z)
          |> filter(fn: (r) => r._measurement == "{measurement}"
               and r._field == "{field}")
          |> last()
        '''
        tables = query_api.query(q, org=INFLUX_ORG)
        for table in tables:
            for record in table.records:
                return record.get_value()
        return None

    def day_sum(bucket: str, measurement: str, field: str) -> float | None:
        day_start = ts.replace(hour=0, minute=0, second=0,
                               microsecond=0).isoformat()
        q = f'''
        from(bucket: "{bucket}")
          |> range(start: {day_start}Z, stop: {window_end}Z)
          |> filter(fn: (r) => r._measurement == "{measurement}"
               and r._field == "{field}")
          |> sum()
        '''
        tables = query_api.query(q, org=INFLUX_ORG)
        for table in tables:
            for record in table.records:
                return record.get_value()
        return None

    def rolling_avg(bucket: str, measurement: str, field: str,
                    days: int, agg_fn: str = "mean") -> float | None:
        start = (ts - timedelta(days=days)).isoformat()
        q = f'''
        from(bucket: "{bucket}")
          |> range(start: {start}Z, stop: {window_end}Z)
          |> filter(fn: (r) => r._measurement == "{measurement}"
               and r._field == "{field}")
          |> aggregateWindow(every: 1d, fn: {agg_fn}, createEmpty: false)
          |> mean()
        '''
        tables = query_api.query(q, org=INFLUX_ORG)
        for table in tables:
            for record in table.records:
                return record.get_value()
        return None

    snapshot = {
        "generated_at": ts.isoformat(),
        "solarpv": {
            "current_power_w":    last_value("solarpv", "solarpv", "ac_power_w"),
            "yield_today_kwh":    day_sum("solarpv", "solarpv", "yield_kwh"),
            "yield_7d_avg_kwh":   rolling_avg("solarpv", "solarpv",
                                              "yield_kwh", 7, "max"),
            "yield_30d_avg_kwh":  rolling_avg("solarpv", "solarpv",
                                              "yield_kwh", 30, "max"),
        },
        "solarthermie": {
            "collector_temp_c":  last_value("solarthermie", "solarthermie",
                                            "collector_temp_c"),
            "buffer_top_c":      last_value("solarthermie", "solarthermie",
                                            "buffer_temp_top_c"),
            "buffer_mid_c":      last_value("solarthermie", "solarthermie",
                                            "buffer_temp_mid_c"),
            "buffer_bottom_c":   last_value("solarthermie", "solarthermie",
                                            "buffer_temp_bottom_c"),
        },
        "heizung": {
            "boiler_temp_c":       last_value("heizung", "heizung",
                                              "boiler_temp_c"),
            "pellets_today_kg":    day_sum("heizung", "heizung",
                                          "pellets_consumed_kg"),
            "pellets_7d_avg_kg":   rolling_avg("heizung", "heizung",
                                               "pellets_consumed_kg", 7, "sum"),
            "pellets_30d_avg_kg":  rolling_avg("heizung", "heizung",
                                               "pellets_consumed_kg", 30, "sum"),
        },
        "weather": {
            "outdoor_temp_c": last_value("solarpv", "solarpv", "outdoor_temp_c"),
        }
    }

    # Filter out snapshots with too many null values
    null_count = sum(
        1 for sub in snapshot.values() if isinstance(sub, dict)
        for v in sub.values() if v is None
    )
    if null_count > 4:
        return None   # not enough data at this timestamp

    return snapshot

# ── LLM generation ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a home energy expert helping the owner of a house
with a photovoltaic solar system, solar thermal collectors, and a pellet heating
system. Your answers are technically precise, grounded in the sensor data
provided, and written in plain English for a non-expert homeowner."""

GENERATION_PROMPT_TEMPLATE = """Given the following sensor snapshot from a home
energy system, generate ONE realistic question that the homeowner might ask, and
provide a correct, detailed answer grounded in the data.

Sensor snapshot:
{context}

Output format — respond with valid JSON only, no other text:
{{
  "question": "...",
  "answer": "..."
}}

Focus on practical guidance: should the homeowner run an appliance now, is
consumption in an expected range, are there any anomalies worth investigating?"""


def call_llama(prompt: str) -> str | None:
    """Call llama.cpp CLI and return the generated text."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
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
                "-t", str(max(1, __import__('os').cpu_count() - 2)),
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
    """Extract the first valid JSON object from llama output."""
    start = text.find('{')
    end   = text.rfind('}')
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

# ── Main ────────────────────────────────────────────────────────────────────────

def random_timestamp() -> datetime:
    """Return a random timestamp within the last HISTORY_DAYS days."""
    now    = datetime.now(timezone.utc)
    delta  = timedelta(days=random.randint(0, HISTORY_DAYS),
                       hours=random.randint(6, 20),   # daylight hours only
                       minutes=random.randint(0, 59))
    return now - delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count",  type=int, default=200,
                        help="Number of QA pairs to generate")
    parser.add_argument("--output", type=str,
                        default="/tmp/synthetic_qa.jsonl",
                        help="Output JSONL file path")
    args = parser.parse_args()

    client   = InfluxDBClient(url=INFLUX_URL,
                              token=INFLUX_TOKEN, org=INFLUX_ORG)
    output   = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    attempts  = 0
    max_attempts = args.count * 5   # allow retries for null snapshots

    print(f"Generating {args.count} QA pairs → {output}")

    with open(output, "w") as out_f:
        while generated < args.count and attempts < max_attempts:
            attempts += 1
            ts = random_timestamp()

            # Fetch sensor snapshot for this timestamp
            snapshot = fetch_snapshot(client, ts)
            if snapshot is None:
                continue   # not enough data at this point in history

            # Build generation prompt
            context_str = json.dumps(snapshot, indent=2)
            prompt = (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"<|user|>\n"
                + GENERATION_PROMPT_TEMPLATE.format(context=context_str)
                + "\n<|assistant|>\n"
            )

            # Call local LLM
            raw_output = call_llama(prompt)
            if raw_output is None:
                continue

            # Parse JSON from LLM output
            qa = extract_json(raw_output)
            if qa is None or "question" not in qa or "answer" not in qa:
                continue

            # Write training example
            example = {
                "context":  context_str,
                "question": qa["question"].strip(),
                "answer":   qa["answer"].strip(),
                "source":   "synthetic",
                "timestamp": ts.isoformat(),
            }
            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
            generated += 1

            if generated % 10 == 0:
                print(f"  {generated}/{args.count} generated "
                      f"({attempts} attempts)")

    client.close()
    print(f"\nDone. {generated} pairs written to {output}")
    print(f"Attempts: {attempts} | Success rate: "
          f"{generated/attempts*100:.1f}%")


if __name__ == "__main__":
    main()
```

Run the script — it will take several hours on CPU, so start it in a `tmux` or `screen` session:

```bash
pip install influxdb-client

tmux new-session -d -s generate \
  'python3 generate_qa_pairs.py \
    --count 500 \
    --output /data/fast/datasets/exports/synthetic_v1.jsonl'

# Monitor progress
tmux attach -t generate
```

A run of 500 pairs typically takes 4–8 hours on a 6-core CPU. Let it complete overnight.

---

### Step 3: Upload synthetic data to MinIO and queue for annotation

Once generation completes, upload the JSONL to MinIO and convert it into Label Studio tasks:

```bash
# Upload raw synthetic output to MinIO
mc cp /data/fast/datasets/exports/synthetic_v1.jsonl \
  local/exports/synthetic/v1/synthetic_v1.jsonl
```

Convert each JSONL line into a Label Studio task JSON and upload as individual task files — Label Studio's sync process works best with one task per file:

```python
# split_to_tasks.py
# Converts a JSONL file into individual Label Studio task JSON files

import json
from pathlib import Path
import subprocess

INPUT  = "/data/fast/datasets/exports/synthetic_v1.jsonl"
OUTDIR = "/tmp/ls_tasks/"
Path(OUTDIR).mkdir(exist_ok=True)

with open(INPUT) as f:
    for i, line in enumerate(f):
        example = json.loads(line)
        task = {
            "context":  example["context"],
            "question": example["question"],
            "answer":   example["answer"],
        }
        task_file = f"{OUTDIR}/task_{i:05d}.json"
        with open(task_file, "w") as out:
            json.dump([task], out, ensure_ascii=False, indent=2)

print(f"Written {i+1} task files to {OUTDIR}")

# Upload all tasks to MinIO exports bucket
subprocess.run([
    "mc", "mirror", "--overwrite",
    OUTDIR, "local/exports/synthetic/v1/tasks/"
])
print("Uploaded to MinIO")
```

```bash
python3 split_to_tasks.py
```

In Label Studio, go to **Settings → Cloud Storage**, set the source prefix to `synthetic/v1/tasks/`, and click **Sync Storage**. All 500 tasks will appear in the annotation queue.

---

### Step 4: The annotation workflow

With 500 synthetic tasks queued, the annotation workflow is straightforward but requires genuine attention — the quality of this step determines the quality of the trained model.

**What to look for when reviewing synthetic QA pairs:**

The local llama.cpp model produces answers that are usually directionally correct but often imprecise. Common failure patterns to watch for:

- **Incorrect numerical reasoning** — the model calculates a percentage or compares values incorrectly. Always check arithmetic against the context snapshot.
- **Seasonal overconfidence** — the model asserts something is "unusually high" without accounting for the season. A pellet consumption of 6kg/day is unremarkable in January but worth investigating in April.
- **Hallucinated thresholds** — the model invents specific temperature thresholds ("the buffer should never exceed 75°C") that are not grounded in the installation's actual configuration.
- **Missing cross-subsystem reasoning** — a good answer about whether to run the dishwasher references both PV production and thermal buffer state. A mediocre answer addresses only one.

For each task, select a verdict, write a corrected answer if needed, and categorise the failure reason. Aim for consistent annotation — if you are unsure whether something is "Correct" or "Partially correct", err toward "Partially correct" and write the missing detail in the correction field. Partial corrections become valuable training signal.

**A realistic annotation rate:** expect to spend 2–4 minutes per task once you develop a rhythm. Five hundred tasks therefore represents roughly 20–30 hours of annotation work. This does not need to happen in one sitting — Label Studio saves progress automatically and you can return to the queue over several days.

**Prioritising which tasks to annotate first:** not all synthetic tasks are equally valuable. Tasks generated from unusual timestamps — early morning, mid-winter, or periods of anomalous consumption — are more valuable than tasks from typical midday summer conditions, which the model will encounter frequently and handle adequately with minimal fine-tuning. The `timestamp` field in the task JSON lets you identify these if you sort by month or hour of day.

---

### Step 5: Export annotations from Label Studio

Once you have annotated at least 200 tasks (500 is better), export the completed annotations and convert them to the training format.

**Export from Label Studio:**

In your project, go to **Export** and select **JSON** format. Save the file to the annotation server:

```bash
# Or use the API directly
curl -X GET \
  "http://annotation-server-ip:8080/api/projects/1/export?exportType=JSON" \
  -H "Authorization: Token your-label-studio-api-token" \
  -o /data/fast/datasets/exports/annotations_v1.json
```

**Convert to JSONL training format:**

```python
# export_annotations.py
# Converts Label Studio JSON export to Unsloth-compatible JSONL training format

import json
from pathlib import Path

INPUT  = "/data/fast/datasets/exports/annotations_v1.json"
OUTPUT = "/data/fast/datasets/training/energy_qa_v1.jsonl"

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are an expert home energy advisor with deep knowledge
of photovoltaic solar systems, solar thermal collectors, and pellet heating.
You are given a real-time sensor snapshot from a residential installation and
a question from the homeowner. Provide accurate, practical guidance grounded
in the data. Be specific — reference actual values from the snapshot rather
than speaking in generalities."""

with open(INPUT) as f:
    annotations = json.load(f)

written  = 0
skipped  = 0

with open(OUTPUT, "w") as out_f:
    for task in annotations:
        # Skip tasks without a completed annotation
        if not task.get("annotations"):
            skipped += 1
            continue

        annotation = task["annotations"][0]   # take first annotator's result
        results    = annotation.get("result", [])

        # Extract verdict
        verdict = next(
            (r["value"]["choices"][0]
             for r in results if r.get("from_name") == "verdict"),
            None
        )
        if verdict is None:
            skipped += 1
            continue

        # Skip tasks marked as Wrong with no correction provided
        correction = next(
            (r["value"]["text"][0]
             for r in results
             if r.get("from_name") == "correction"
             and r["value"].get("text")),
            None
        )
        if verdict == "Wrong" and correction is None:
            skipped += 1
            continue

        # Build the final answer:
        # - Correct → use the original model answer
        # - Partially correct or Wrong → use the human correction
        data   = task["data"]
        answer = (
            correction
            if verdict in ("Wrong", "Partially correct") and correction
            else data["answer"]
        )

        # Format as Alpaca-style instruction example for Unsloth
        example = {
            "instruction": SYSTEM_PROMPT,
            "input": (
                f"Sensor context:\n{data['context']}\n\n"
                f"Question: {data['question']}"
            ),
            "output": answer,
        }

        out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
        written += 1

print(f"Exported {written} training examples ({skipped} skipped)")
print(f"Output: {OUTPUT}")
```

```bash
python3 export_annotations.py
# Exported 423 training examples (77 skipped)
```

Upload the training dataset to MinIO:

```bash
mc cp /data/fast/datasets/training/energy_qa_v1.jsonl \
  local/datasets/training/energy_qa_v1.jsonl

# Verify
mc ls local/datasets/training/
```

---

### Step 6: Prepare the cloud GPU instance

The fine-tuning run requires a machine with a capable GPU. The author uses an RTX 4090 instance on a GPU cloud provider at approximately €0.50/hr. The total cost of a training run is €1–2.

**Starting the instance:**

On your GPU cloud provider, start an Ubuntu 22.04 or 24.04 instance with:
- GPU: RTX 4090 (24GB VRAM) or A100 40/80GB
- Disk: at least 100GB NVMe scratch space
- Python 3.10+ pre-installed

**Install dependencies on the cloud instance:**

```bash
# On the cloud GPU instance
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install influxdb-client datasets

# Verify GPU is visible
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
# NVIDIA GeForce RTX 4090
```

**Rsync the training data from the annotation server:**

```bash
# On the annotation server — sync dataset to cloud GPU
rsync -avz --progress \
  /data/fast/datasets/training/energy_qa_v1.jsonl \
  user@cloud-gpu-ip:/workspace/data/energy_qa_v1.jsonl

# Also sync the base model if you have it cached locally
# Otherwise Unsloth will download it from Hugging Face directly
```

---

### Step 7: The fine-tuning script

Save this as `train_energy_adapter.py` on the cloud GPU instance:

```python
#!/usr/bin/env python3
"""
train_energy_adapter.py

Fine-tunes Qwen2.5-14B-Instruct on the energy QA dataset using
QLoRA via Unsloth. Outputs a LoRA adapter to ./output/lora_adapter/

Runtime on RTX 4090:
  ~2 hours for 2,000 examples · 3 epochs
  ~4 hours for 5,000 examples · 3 epochs
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# ── Model configuration ─────────────────────────────────────────────────────────

MAX_SEQ_LENGTH = 2048   # Covers context + question + answer comfortably
DTYPE          = None   # Auto-detect (bfloat16 on Ampere/Hopper)
LOAD_IN_4BIT   = True   # QLoRA — reduces VRAM from ~28GB to ~10GB

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = "Qwen/Qwen2.5-14B-Instruct",
    max_seq_length  = MAX_SEQ_LENGTH,
    dtype           = DTYPE,
    load_in_4bit    = LOAD_IN_4BIT,
)

# ── LoRA configuration ──────────────────────────────────────────────────────────
# Targets the attention and feed-forward projection layers.
# r=16 is a good balance between adapter expressiveness and size.

model = FastLanguageModel.get_peft_model(
    model,
    r                   = 16,
    target_modules      = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha          = 16,
    lora_dropout        = 0,       # Optimised at 0
    bias                = "none",
    use_gradient_checkpointing = "unsloth",   # Reduces VRAM further
    random_state        = 42,
    use_rslora          = False,
    loftq_config        = None,
)

# ── Dataset ─────────────────────────────────────────────────────────────────────

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with \
an input that provides further context. Write a response that appropriately \
completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token


def format_prompts(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, inp, out in zip(instructions, inputs, outputs):
        text = ALPACA_PROMPT.format(instruction, inp, out) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


dataset = load_dataset(
    "json",
    data_files = "/workspace/data/energy_qa_v1.jsonl",
    split      = "train",
)

# 90/10 train/validation split
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"].map(format_prompts, batched=True)
eval_dataset  = dataset["test"].map(format_prompts, batched=True)

print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(eval_dataset)}")

# ── Training arguments ──────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir             = "./output/checkpoints",
    num_train_epochs       = 3,
    per_device_train_batch_size  = 2,
    per_device_eval_batch_size   = 2,
    gradient_accumulation_steps  = 4,   # effective batch size = 8
    warmup_steps           = 20,
    learning_rate          = 2e-4,
    fp16                   = not torch.cuda.is_bf16_supported(),
    bf16                   = torch.cuda.is_bf16_supported(),
    logging_steps          = 10,
    eval_strategy          = "steps",
    eval_steps             = 50,
    save_strategy          = "steps",
    save_steps             = 100,
    load_best_model_at_end = True,
    metric_for_best_model  = "eval_loss",
    optim                  = "adamw_8bit",
    weight_decay           = 0.01,
    lr_scheduler_type      = "cosine",
    seed                   = 42,
    report_to              = "none",   # set to "wandb" if you want tracking
)

# ── Trainer ─────────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model            = model,
    tokenizer        = tokenizer,
    train_dataset    = train_dataset,
    eval_dataset     = eval_dataset,
    dataset_text_field = "text",
    max_seq_length   = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing          = False,
    args             = training_args,
)

# ── Run ─────────────────────────────────────────────────────────────────────────

print("Starting training...")
trainer_stats = trainer.train()

print(f"\nTraining complete.")
print(f"  Total steps:   {trainer_stats.global_step}")
print(f"  Training loss: {trainer_stats.training_loss:.4f}")
print(f"  Runtime:       {trainer_stats.metrics['train_runtime']:.0f}s")

# ── Save adapter ────────────────────────────────────────────────────────────────

model.save_pretrained("./output/lora_adapter")
tokenizer.save_pretrained("./output/lora_adapter")

print("\nLoRA adapter saved to ./output/lora_adapter")
print("Run `ls -lh ./output/lora_adapter/` to confirm output files")
```

Run the training:

```bash
cd /workspace
python3 train_energy_adapter.py 2>&1 | tee training_log.txt
```

Training progress will show loss values every 10 steps. On an RTX 4090 with 423 training examples and 3 epochs, expect a runtime of approximately 90–120 minutes. Watch for the validation loss to decrease and plateau — if it begins rising, the model is overfitting and you should stop early.

A successful run produces output similar to:

```
Training complete.
  Total steps:   159
  Training loss: 0.8234
  Runtime:       5847s

LoRA adapter saved to ./output/lora_adapter
```

---

### Step 8: Rsync the adapter back to the homelab

Once training completes, transfer the adapter back to the annotation server and then into MinIO:

```bash
# On the cloud GPU instance — archive the adapter
tar -czf lora_adapter_energy_v1.tar.gz ./output/lora_adapter/

# Check the size — should be approximately 260–270MB
ls -lh lora_adapter_energy_v1.tar.gz

# Rsync back to the annotation server
rsync -avz --progress \
  lora_adapter_energy_v1.tar.gz \
  user@annotation-server-ip:/data/fast/scratch/
```

On the annotation server, upload the adapter to MinIO:

```bash
# Upload adapter archive
mc cp /data/fast/scratch/lora_adapter_energy_v1.tar.gz \
  local/adapters/energy-assistant/v1/lora_adapter_energy_v1.tar.gz

# Tag with metadata
mc tag set \
  local/adapters/energy-assistant/v1/lora_adapter_energy_v1.tar.gz \
  "base-model=Qwen2.5-14B-Instruct" \
  "training-examples=423" \
  "epochs=3" \
  "final-eval-loss=0.7891"

# Verify
mc ls local/adapters/energy-assistant/v1/
```

**Terminate the cloud GPU instance now.** Do not leave it running — at €0.50/hr a forgotten instance costs €12/day. Confirm the adapter is safely in MinIO before terminating.

```bash
# Final check before terminating
mc stat local/adapters/energy-assistant/v1/lora_adapter_energy_v1.tar.gz
# Confirm the file size matches what you rsynced
```

---

### Verifying the complete training run

Before moving to Article 5, confirm these three things:

**1. The adapter is in MinIO.** Run `mc ls local/adapters/` and confirm the archive is present with the expected file size (~260–270MB compressed).

**2. The training log shows converging loss.** Open `training_log.txt` and confirm that both training loss and validation loss decreased over the course of training. A final training loss below 1.0 and eval loss below 1.2 indicates the model has learned something meaningful from your dataset.

**3. The training dataset is preserved.** Run `mc ls local/datasets/training/` and confirm the JSONL file is present. You will need it for subsequent training runs to add new examples to the existing dataset rather than starting from scratch.

```bash
# Quick sanity check — count lines in the training file
mc cat local/datasets/training/energy_qa_v1.jsonl | wc -l
# Should match the number reported by export_annotations.py
```

---

### A note on training set size and iteration cadence

423 examples is a modest starting point. The model will show measurable improvement on energy advisory tasks — better use of specific sensor values in responses, more accurate seasonal reasoning, fewer hallucinated thresholds — but it will also have gaps, particularly for edge cases and unusual seasonal patterns that are underrepresented in the synthetic data.

This is by design. The flywheel is not intended to produce a perfect model from a single training run. It is intended to produce a model that is better than the base model, deployed quickly, and improved continuously as real user feedback accumulates. Each subsequent training run incorporates both the original synthetic data and new corrected real interactions. After three or four cycles the model's behaviour will be noticeably more consistent and domain-specific than anything you could achieve from a single large batch of synthetic data alone.

The practical cadence for a home installation: run a new training cycle when you have accumulated 100–200 new annotated real interactions since the last run. At typical usage of a few queries per day, this means roughly one training run every 2–3 months in the early stages, compressing to monthly or more frequent as usage increases.

---

### What comes next

In Article 5 we merge the LoRA adapter into the base model, convert to GGUF format, quantise to q4_k_m, load the result into Ollama, and wire OpenWebUI to send user feedback back to Label Studio. By the end of Article 5 the flywheel is fully operational: the model answers questions, users rate responses, corrections accumulate, and the next training run begins automatically when the threshold is reached.

*The companion GitHub repository contains the complete `generate_qa_pairs.py`, `export_annotations.py`, and `train_energy_adapter.py` scripts referenced in this article, along with a `requirements.txt` for the cloud GPU environment.*

---

*Series: Building an AI Data Flywheel for Home Energy Management*
*Article 4 of 5 — Dataset Generation and Fine-Tuning*
