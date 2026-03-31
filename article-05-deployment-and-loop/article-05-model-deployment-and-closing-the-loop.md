# Building an AI Data Flywheel for Home Energy Management
## Part 5 of 5: Model Deployment and Closing the Loop

*This is the fifth and final article in a five-part click-along series. Articles 1–4 established the concept, built the metrics server and data foundation, set up the object store and annotation infrastructure, and produced a trained LoRA adapter. This article deploys that adapter, wires the feedback loop, and completes the flywheel.*

---

### What we are building in this article

By the end of this article you will have:

- The LoRA adapter merged into the Qwen2.5-14B base model and converted to fp16 GGUF format on the inference server
- A q4_k_m quantised model (~8.4GB) loaded into Ollama and serving queries
- OpenWebUI running as the chat interface, connected to the model via Ollama
- A Node-RED webhook that captures OpenWebUI feedback (thumbs up/down, free-text corrections) and routes them back to Label Studio as new annotation tasks
- The complete flywheel operational: query → response → feedback → annotation → training → deployment → query

If you already have Ollama running on your inference server, skip to **Step 3: Loading the model into Ollama** — the merge and quantisation steps are only needed the first time or after each new training run.

---

### The final mile

Four articles of infrastructure work converge in this one. The LoRA adapter sitting in your MinIO `adapters` bucket is the output of the training pipeline — a 263MB file that represents everything the model learned from your annotated energy data. On its own it is not usable: it is a set of weight differences relative to the Qwen2.5-14B base, not a standalone model.

To serve it with Ollama, three transformations are needed. First, the adapter is merged back into the base model weights using PEFT, producing a full-precision model in HuggingFace format. Second, that merged model is converted to GGUF format using llama.cpp's conversion tooling. Third, the GGUF is quantised to 4-bit precision (q4_k_m), reducing the 28GB fp16 model to approximately 8.4GB — small enough to fit comfortably in the 12GB VRAM of the inference server's RTX A2000.

All three steps happen on the inference server, which is the right machine for them: its 128GB of system RAM gives the merge process enough headroom to load the full base model without swapping, even before the GPU becomes involved.

---

### Step 1: Prepare the inference server

The inference server needs Python, the PEFT and Transformers libraries for the merge step, and llama.cpp for the conversion and quantisation steps. Ollama should already be installed if you have been following along — if not, install it now.

```bash
# On the inference server (Ubuntu 24.04 LTS)
sudo apt update && sudo apt install -y \
  python3-pip python3-venv git build-essential cmake

# Create a virtual environment for the merge/conversion work
python3 -m venv ~/merge-env
source ~/merge-env/bin/activate

# Install merge dependencies
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft accelerate bitsandbytes

# Install Ollama if not already present
curl -fsSL https://ollama.com/install.sh | sh

# Verify Ollama is running
ollama list
```

Clone llama.cpp for the GGUF conversion and quantisation tools:

```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DLLAMA_CUDA=ON   # enable CUDA for the quantisation step
cmake --build build --config Release -j$(nproc)
pip install -r requirements.txt
```

The `-DLLAMA_CUDA=ON` flag enables GPU-accelerated quantisation. With 12GB VRAM the quantisation step takes about 10 minutes rather than an hour on CPU alone.

---

### Step 2: Merge the adapter and convert to GGUF

Pull the adapter from MinIO to the inference server's scratch space:

```bash
# Configure mc on the inference server
curl -O https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc && sudo mv mc /usr/local/bin/

mc alias set store http://object-store-ip:9000 \
  admin your-minio-password

# Pull the adapter archive
mkdir -p ~/scratch
mc cp \
  store/adapters/energy-assistant/v1/lora_adapter_energy_v1.tar.gz \
  ~/scratch/

# Extract
cd ~/scratch
tar -xzf lora_adapter_energy_v1.tar.gz
ls lora_adapter/
# adapter_config.json  adapter_model.safetensors  tokenizer files ...
```

Run the merge script. This loads the base model, applies the adapter weights, and saves the result as a standard HuggingFace model directory:

```python
#!/usr/bin/env python3
# merge_adapter.py
# Merges a LoRA adapter into the base model and saves a full-weight checkpoint.
# Run on the inference server inside the merge-env virtual environment.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE_MODEL   = "Qwen/Qwen2.5-14B-Instruct"   # downloaded from HF on first run
ADAPTER_PATH = os.path.expanduser("~/scratch/lora_adapter")
OUTPUT_PATH  = os.path.expanduser("~/scratch/merged_model")

print(f"Loading base model: {BASE_MODEL}")
print("This will download ~28GB on first run — subsequent runs use cache.")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype    = torch.float16,
    device_map     = "auto",   # uses GPU if available, CPU otherwise
    low_cpu_mem_usage = True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print(f"Loading adapter from: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging adapter weights into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to: {OUTPUT_PATH}")
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("\nMerge complete.")
print(f"Merged model size: ", end="")
total = sum(
    os.path.getsize(os.path.join(OUTPUT_PATH, f))
    for f in os.listdir(OUTPUT_PATH)
    if os.path.isfile(os.path.join(OUTPUT_PATH, f))
)
print(f"{total / 1e9:.1f} GB")
```

```bash
source ~/merge-env/bin/activate
python3 merge_adapter.py
```

On the first run, the script downloads the Qwen2.5-14B base model from Hugging Face (~28GB). This takes 20–40 minutes depending on your connection speed. The merged model directory will occupy roughly 28GB on disk.

The merge itself — combining the adapter weights with the base — takes 5–10 minutes and requires approximately 60GB of system RAM. The 128GB on the inference server handles this comfortably. If you are working with a machine that has less RAM, add `load_in_8bit=True` to `from_pretrained()` to reduce memory pressure at the cost of some precision.

---

### Step 3: Convert to GGUF and quantise to q4_k_m

With the merged model saved, convert it to GGUF format and then quantise:

```bash
cd ~/llama.cpp

# Step 1: Convert HuggingFace model to fp16 GGUF
python3 convert_hf_to_gguf.py \
  ~/scratch/merged_model \
  --outfile ~/scratch/energy-assistant-v1-f16.gguf \
  --outtype f16

# Verify the output — expect approximately 28GB
ls -lh ~/scratch/energy-assistant-v1-f16.gguf
```

```bash
# Step 2: Quantise to q4_k_m (~8.4GB)
./build/bin/llama-quantize \
  ~/scratch/energy-assistant-v1-f16.gguf \
  ~/scratch/energy-assistant-v1-q4_k_m.gguf \
  q4_k_m

# Verify the quantised output — expect approximately 8.4GB
ls -lh ~/scratch/energy-assistant-v1-q4_k_m.gguf
```

The quantisation step takes 10–15 minutes with CUDA enabled on the RTX A2000. The output is the model file that Ollama will serve.

**A note on the q4_k_m format:** this is a 4-bit quantisation scheme that groups weights into blocks and applies different quantisation levels within each block (the "k" and "m" refer to the grouping strategy). For a 14B parameter model it provides an excellent balance — the quality degradation relative to the fp16 original is small enough to be imperceptible in advisory conversations, while the size reduction (28GB → 8.4GB) makes the model practical for a 12GB VRAM GPU.

Store the quantised model back in MinIO before loading into Ollama, so you have a durable copy independent of the inference server's local disk:

```bash
mc cp ~/scratch/energy-assistant-v1-q4_k_m.gguf \
  store/models/energy-assistant/v1/energy-assistant-v1-q4_k_m.gguf

mc tag set \
  store/models/energy-assistant/v1/energy-assistant-v1-q4_k_m.gguf \
  "base-model=Qwen2.5-14B-Instruct" \
  "quantisation=q4_k_m" \
  "size-gb=8.4"
```

---

### Step 4: Create and load the Ollama model

Ollama uses a Modelfile — a short configuration file — to define how the model should be served. Create one that sets the system prompt and inference parameters appropriate for the energy advisory task:

```bash
nano ~/scratch/Modelfile
```

```
FROM /root/scratch/energy-assistant-v1-q4_k_m.gguf

SYSTEM """
You are an expert home energy advisor with deep knowledge of photovoltaic
solar systems, solar thermal collectors, and pellet heating. You are given
a real-time sensor snapshot from a residential installation alongside a
question from the homeowner.

Always ground your answers in the specific values from the sensor snapshot.
Do not invent thresholds or reference values that are not in the data or
derivable from it. When you are uncertain, say so.

Be concise and practical. The homeowner wants actionable guidance, not
a lecture on energy physics.
"""

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER num_predict 512
```

```bash
# Register the model with Ollama
ollama create energy-assistant-v1 \
  -f ~/scratch/Modelfile

# Verify it appears in Ollama's model list
ollama list
```

Run a quick smoke test to confirm the model loads and responds:

```bash
ollama run energy-assistant-v1 \
  "The thermal buffer is at 42°C and it is 09:00 on a clear day in June. \
   PV is producing 2.1 kW. What do you recommend?"
```

If the model responds with coherent, sensor-grounded advice, it is working correctly. If it refuses to answer or produces garbled output, check that the GGUF file is not corrupted (`md5sum` it against the MinIO copy) and that the Modelfile path points to the correct file.

---

### Step 5: Install and configure OpenWebUI

OpenWebUI provides the browser-based chat interface that connects users to the Ollama-served model. It runs as a Docker container on the inference server.

```bash
# Pull and start OpenWebUI
docker run -d \
  --name openwebui \
  --restart unless-stopped \
  -p 3000:8080 \
  -v openwebui-data:/app/backend/data \
  -e OLLAMA_BASE_URL=http://host-gateway:11434 \
  --add-host=host-gateway:host-gateway \
  ghcr.io/open-webui/open-webui:main

# Watch startup
docker logs -f openwebui
```

Open `http://inference-server-ip:3000` in a browser. On first load, create an admin account. Once logged in:

1. Go to **Settings → Models** and confirm `energy-assistant-v1` appears in the model list
2. Set it as the default model
3. Go to **Settings → Interface** and enable **Response Rating** — this activates the thumbs up/down buttons on each response, which are the primary feedback signal for the flywheel

Test the full context flow by sending a question that requires real sensor data. The system prompt alone is not enough — the context block assembled by Node-RED in Article 2 needs to be prepended to the user's message. In the next step we wire that together with a system-level prompt injection.

---

### Step 6: Wire the context API into OpenWebUI

The context block assembled by Node-RED (the JSON snapshot of current sensor readings from Article 2) needs to reach the model with every query. The cleanest approach is a Node-RED function that intercepts the OpenWebUI request, fetches the context, and prepends it.

OpenWebUI supports a **pipelines** mechanism that allows preprocessing of messages before they reach the model. Create a pipeline on the inference server:

```bash
mkdir -p ~/openwebui-pipelines
nano ~/openwebui-pipelines/energy_context_pipeline.py
```

```python
"""
energy_context_pipeline.py

OpenWebUI pipeline that fetches the current sensor context from
the Node-RED context API and prepends it to every user message.

Install in OpenWebUI: Admin → Pipelines → Upload
"""

from typing import List, Optional
import httpx
import json


class Pipeline:
    class Valves:
        CONTEXT_API_URL: str = "http://metrics-server-ip:1880/api/context"
        CONTEXT_TIMEOUT: int = 5

    def __init__(self):
        self.name    = "Energy context injector"
        self.valves  = self.Valves()

    async def on_startup(self):
        print(f"Energy context pipeline started")
        print(f"Context API: {self.valves.CONTEXT_API_URL}")

    async def inlet(
        self,
        body: dict,
        user: Optional[dict] = None,
    ) -> dict:
        """Fetch sensor context and prepend to the latest user message."""

        # Fetch current sensor snapshot from Node-RED
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.valves.CONTEXT_API_URL,
                    timeout=self.valves.CONTEXT_TIMEOUT,
                )
                context = response.json()
        except Exception as e:
            print(f"Context API unavailable: {e}")
            return body   # fall through without context rather than failing

        context_block = (
            "Current sensor readings from the home energy system:\n"
            + json.dumps(context, indent=2)
            + "\n\n"
        )

        # Prepend context to the last user message
        messages: List[dict] = body.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                msg["content"] = context_block + msg["content"]
                break

        return body
```

Install the pipeline in OpenWebUI:

1. Go to **Admin Panel → Pipelines**
2. Click **Upload a pipeline**
3. Upload `energy_context_pipeline.py`
4. Enable it — it will now run on every message sent through OpenWebUI

Test with a question like "Should I run the washing machine now?" — the model should respond with specific reference to current PV output, thermal buffer state, and time of day, because those values are now injected automatically with every query.

---

### Step 7: The feedback webhook — closing the loop

This is the step that turns a deployed model into a flywheel. When a user rates a response in OpenWebUI, that rating needs to find its way back to Label Studio as a new annotation task. A Node-RED flow handles this.

**First, create a dedicated MinIO bucket prefix for live feedback:**

```bash
# On the object store server
# (The exports bucket already exists — we use a subfolder)
mc mb store/exports/feedback 2>/dev/null || true
```

**Create the feedback webhook flow in Node-RED:**

In Node-RED on the metrics server, import the following flow:

```json
[
  {
    "id": "feedback-webhook-in",
    "type": "http in",
    "method": "POST",
    "url": "/api/feedback",
    "name": "OpenWebUI feedback"
  },
  {
    "id": "parse-feedback",
    "type": "function",
    "name": "Build annotation task",
    "func": "const body = msg.payload;\n\n// OpenWebUI sends: message_id, rating (1=up/-1=down), comment\nconst task = {\n  context:  body.context  || '',\n  question: body.message  || '',\n  answer:   body.response || '',\n  rating:   body.rating,\n  comment:  body.comment  || '',\n  source:   'user_feedback',\n  timestamp: new Date().toISOString()\n};\n\n// Only queue negative or corrected responses\n// Thumbs-up with no comment is good signal but not worth annotating\nif (body.rating === 1 && !body.comment) {\n  msg.skip = true;\n  return msg;\n}\n\nmsg.task = task;\nmsg.filename = `feedback_${Date.now()}.json`;\nreturn msg;"
  },
  {
    "id": "skip-check",
    "type": "switch",
    "name": "Skip if no annotation needed",
    "property": "skip",
    "rules": [
      { "t": "true" },
      { "t": "else" }
    ]
  },
  {
    "id": "write-to-minio",
    "type": "exec",
    "name": "Upload task to MinIO",
    "command": "",
    "addpay": false,
    "func": "const task = JSON.stringify([msg.task], null, 2);\nconst filename = msg.filename;\nconst tmpFile = `/tmp/${filename}`;\n\n// Write task JSON to temp file, then upload to MinIO\nconst fs = require('fs');\nfs.writeFileSync(tmpFile, task);\n\nmsg.payload = `mc cp ${tmpFile} store/exports/feedback/${filename} && rm ${tmpFile}`;\nreturn msg;"
  },
  {
    "id": "exec-mc",
    "type": "exec",
    "name": "Run mc cp",
    "command": "bash -c",
    "addpay": true
  },
  {
    "id": "feedback-response",
    "type": "http response",
    "name": "Acknowledge feedback",
    "statusCode": "200"
  }
]
```

Wire: `feedback-webhook-in` → `parse-feedback` → `skip-check` → (else branch) → `write-to-minio` → `exec-mc` → `feedback-response`. The true branch of `skip-check` goes directly to `feedback-response` to acknowledge without writing.

**Configure OpenWebUI to send feedback to Node-RED:**

In OpenWebUI, go to **Admin Panel → Settings → General** and set the **Webhook URL** to:

```
http://metrics-server-ip:1880/api/feedback
```

OpenWebUI will now POST to this endpoint whenever a user submits a rating. The payload includes the conversation context, the rated message, and any free-text comment.

**Create a Label Studio sync in Node-RED** to run every hour and pull new feedback tasks into the annotation queue:

```json
[
  {
    "id": "feedback-sync-timer",
    "type": "inject",
    "repeat": "3600",
    "name": "Every hour"
  },
  {
    "id": "sync-feedback-bucket",
    "type": "exec",
    "command": "bash -c",
    "addpay": false,
    "func": "msg.payload = 'mc mirror --overwrite store/exports/feedback/ /tmp/ls-feedback-staging/ && echo done';\nreturn msg;"
  },
  {
    "id": "trigger-ls-sync",
    "type": "http request",
    "method": "POST",
    "url": "http://annotation-server-ip:8080/api/storages/1/sync",
    "headers": {
      "Authorization": "Token your-label-studio-api-token"
    },
    "name": "Sync Label Studio storage"
  }
]
```

Wire: `feedback-sync-timer` → `sync-feedback-bucket` → `trigger-ls-sync`.

Adjust the storage ID (`/api/storages/1/sync`) to match the source storage connection ID in your Label Studio project — you can find it in **Settings → Cloud Storage** where the connection URL is displayed.

---

### Step 8: Verify the complete flywheel

With everything in place, verify each stage of the loop works end to end.

**Test query → response:**

```bash
curl -s -X POST http://metrics-server-ip:1880/api/context \
  | python3 -m json.tool
# Confirm context block is populated

# Then send a question through OpenWebUI manually
# and confirm the response references specific sensor values
```

**Test feedback routing:**

```bash
# Simulate a negative feedback event from OpenWebUI
curl -s -X POST http://metrics-server-ip:1880/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "rating": -1,
    "message": "The thermal buffer is at 67°C. Should I run the dishwasher?",
    "response": "I cannot determine the best time without knowing your solar output.",
    "comment": "Wrong — PV data was in the context. Answer should reference 3.84kW production.",
    "context": "{\"solarpv\": {\"current_power_w\": 3840}}"
  }'

# Confirm the task appeared in MinIO
mc ls store/exports/feedback/

# Trigger Label Studio sync manually
curl -s -X POST \
  "http://annotation-server-ip:8080/api/storages/1/sync" \
  -H "Authorization: Token your-label-studio-api-token"

# Confirm the task appears in Label Studio
# Open http://annotation-server-ip:8080 → your project → task queue
```

**Test the full annotation → training → deploy cycle:**

This is the long-form verification — annotate the simulated feedback task in Label Studio, run the export script from Article 4, check that the JSONL includes the new example, and confirm the dataset in MinIO has been updated. You do not need to trigger a full training run for this check — confirming the data flows correctly through every stage is sufficient.

```bash
# After annotating the test task in Label Studio:
python3 export_annotations.py

mc ls local/datasets/training/
mc cat local/datasets/training/energy_qa_v1.jsonl | wc -l
# Line count should have increased by 1
```

---

### Step 9: Firewall rules on the inference server

```bash
# OpenWebUI — accessible from your LAN
sudo ufw allow from 192.168.0.0/16 to any port 3000 \
  comment 'OpenWebUI'

# Ollama API — accessible from LAN (for pipeline and testing)
sudo ufw allow from 192.168.0.0/16 to any port 11434 \
  comment 'Ollama API'

sudo ufw enable
sudo ufw status
```

---

### Running subsequent training cycles

Once the flywheel is turning, subsequent training cycles follow the same pattern as Article 4 but with an important difference: the new dataset should include both the original synthetic examples and the accumulated real feedback, not just the new examples in isolation.

A helper script to merge datasets before each training run:

```bash
#!/bin/bash
# merge_datasets.sh — combines all dataset versions into a single training file

OUTPUT="/data/fast/datasets/training/energy_qa_combined.jsonl"

# Pull all dataset versions from MinIO
mc mirror store/datasets/training/ /tmp/dataset-merge/

# Concatenate all JSONL files, deduplicating by content hash
python3 << 'EOF'
import hashlib, json, glob

seen   = set()
output = []

for path in sorted(glob.glob("/tmp/dataset-merge/*.jsonl")):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h = hashlib.md5(line.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                output.append(line)

with open("/data/fast/datasets/training/energy_qa_combined.jsonl", "w") as f:
    f.write("\n".join(output) + "\n")

print(f"Combined dataset: {len(output)} unique examples")
EOF

# Upload combined dataset to MinIO
mc cp "$OUTPUT" store/datasets/training/energy_qa_combined.jsonl

echo "Ready to rsync energy_qa_combined.jsonl to cloud GPU"
```

Use `energy_qa_combined.jsonl` as the training file in `train_energy_adapter.py` for all runs after the first. The deduplication step ensures that examples which appear in multiple export batches are not counted multiple times during training.

---

### Verifying the flywheel is fully operational

Before declaring the system complete, confirm all eight stages of the loop work:

**1. Sensor data flows continuously.** Open the InfluxDB Data Explorer and confirm recent timestamps in each bucket. Check Node-RED for any MQTT errors.

**2. Context assembly works on demand.** `curl -X POST http://metrics-server-ip:1880/api/context` returns a populated JSON block within two seconds.

**3. The model serves responses.** `ollama run energy-assistant-v1 "test"` responds with a coherent answer.

**4. OpenWebUI shows the model and accepts queries.** Log in, send a question, and confirm the response references current sensor values (meaning the context pipeline is injecting correctly).

**5. Feedback reaches Node-RED.** Rate a response in OpenWebUI and confirm the POST appears in the Node-RED debug panel within seconds.

**6. Feedback tasks appear in MinIO.** `mc ls store/exports/feedback/` shows a new file after each negative rating or correction.

**7. Label Studio picks up feedback tasks.** After the hourly sync, negative feedback appears in the annotation queue.

**8. Completed annotations flow to the datasets bucket.** Submit an annotation in Label Studio, sync target storage, and confirm the file appears in `mc ls store/datasets/`.

If all eight pass, the flywheel is spinning. Every query the model handles from this point forward is a potential training example for the next version.

---

### What the flywheel looks like in practice

In the first weeks after deployment, the model will occasionally give imprecise answers — seasonal reasoning that doesn't account for the current month, numerical comparisons that are directionally correct but off by a meaningful margin, or advice that ignores one subsystem while reasoning well about another. These are not failures; they are the inputs to the next training cycle.

Rate every response you are uncertain about. Write corrections for every answer that is wrong. Use the failure reason field in Label Studio consistently — over time, the distribution of failure reasons tells you exactly where the model's weakest points are, and you can weight synthetic data generation to oversample those scenarios in the next training run.

After two or three cycles — typically 2–6 months of casual daily use — the pattern of errors changes noticeably. The model stops hallucinating thresholds it was never told about. It starts referencing the 30-day consumption baseline when it is relevant and staying silent about it when it is not. It learns that a thermal buffer at 42°C at 14:00 on a sunny day is a different situation from the same reading at 08:00 on a cloudy one. These are not things that can be prompted into a base model reliably — they are the product of training on your specific installation's data.

That is the point of the flywheel. Not a perfect model on day one, but a model that gets measurably better at your specific home, in your specific climate, answering the specific questions you actually ask.

---

### Where to go from here

The system described across this series is deliberately minimal — every component was chosen for practical deployability on secondhand homelab hardware, not for maximum capability. Several natural extensions are worth considering once the basic flywheel is running:

**Seasonal evaluation sets.** Build a small fixed evaluation set — 50–100 questions with known correct answers, covering each season and common edge cases. Run this set against every new model version before deploying it. A regression on summer thermal buffer advice should be caught before it reaches the production model.

**Preference pairs for RLHF.** When you correct a wrong answer, the original answer and your correction form a preference pair (wrong vs right). These can be fed into a preference optimisation step (DPO or ORPO) in addition to the SFT fine-tuning in Article 4. This tends to produce models that are not just more accurate but more appropriately confident — they hedge less on questions where they have strong training signal and are more explicit about uncertainty where they do not.

**Additional sensor subsystems.** The three-bucket structure (solarpv, solarthermie, heizung) established in Article 2 can be extended to any additional measurement source — grid import/export, battery state of charge, hot water consumption, EV charging sessions. Each new subsystem added to the context block gives the model more signal to reason from, and synthetic data generation naturally covers the new combinations.

**Automated training triggers.** The current cadence (train when you have 100–200 new examples) requires manual judgment. A Node-RED flow that monitors the `datasets` bucket, counts new examples since the last training run, and sends you a notification — or triggers a cloud GPU instance automatically via API — closes the last manual step in the loop.

---

*The companion GitHub repository contains all scripts, Compose files, Modelfiles, Node-RED flow exports, and configuration templates referenced across this series. A `README.md` in the repository root maps each file to the article and step where it is introduced.*

---

*Series: Building an AI Data Flywheel for Home Energy Management*
*Article 5 of 5 — Model Deployment and Closing the Loop*
