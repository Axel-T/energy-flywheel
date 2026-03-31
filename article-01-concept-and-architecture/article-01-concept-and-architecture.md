# Building an AI Data Flywheel for Home Energy Management
## Part 1 of 5: The Concept, the Use Case, and the Architecture

*This is the first article in a five-part click-along series. By the end of the series you will have a self-improving AI recommendation assistant running entirely on your own hardware — no cloud subscriptions, no vendor lock-in, and no data leaving your home.*

---

### Why a "data flywheel" and why now?

The term *data flywheel* describes a self-reinforcing loop: a deployed model generates predictions, users interact with those predictions, that interaction data is collected and labelled, and the labelled data trains a better model — which gets deployed, and the loop repeats. Each revolution of the wheel produces a slightly smarter assistant.

The concept was popularised by Andrej Karpathy during his time at Tesla, where he described the Autopilot team's approach to continuous model improvement at scale. Their key insight was that the deployed fleet itself is the data collection infrastructure. Every edge case the model encounters in production becomes a training example for the next version.

For a home energy management system this idea translates surprisingly well. Your photovoltaic array, solar thermal collectors, and pellet heating system generate continuous sensor data. A language model can reason about that data and offer recommendations — when to run high-consumption appliances, whether the thermal buffer is sized correctly for the forecast, whether pellet consumption is tracking within expected bounds for the season. Over time, you can tell the model when it was right and when it was wrong. That feedback becomes training data. The model gets better at your specific installation, your climate, your usage patterns.

This series documents exactly how to build that system on secondhand homelab hardware, using open-source software throughout.

---

### The use case: an energy advisory assistant

The assistant we are building answers questions like:

- *"My PV yield was 38 kWh today but I only consumed 12 kWh. Where did the rest go and was it a good day?"*
- *"The thermal buffer is at 67°C at 14:00. Should I run the dishwasher now or wait?"*
- *"Pellet consumption this week is 18% above last week's same-period average. Is that expected given the temperature delta?"*

These are not simple threshold queries. They require reasoning across multiple data sources simultaneously — PV yield, thermal buffer state, outdoor temperature, forecast, historical consumption patterns, and appliance schedules. A well-prompted large language model with access to structured context from your sensor history handles this kind of multi-variable reasoning naturally.

The base model we use throughout this series is **Qwen2.5-14B-Instruct**, a 14-billion-parameter open-weights model from Alibaba that performs strongly on technical reasoning tasks and fits comfortably on consumer hardware with 4-bit quantisation (approximately 8.4GB at q4_k_m). It is fine-tuned on domain-specific question-answer pairs derived from 3.5 years of real sensor data from a single installation.

---

### The architecture at a glance

Before diving into individual components, here is the complete system as it will exist by the end of Article 5:

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                          │
│  Home Assistant · Node-RED · MQTT                           │
│  PV inverter · thermal sensors · pellet boiler · weather    │
└──────────────────────────┬──────────────────────────────────┘
                           │ time-series writes
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      METRICS SERVER                          │
│  InfluxDB v2 — 3.5 years of PV, thermal, heating data      │
│  Node-RED — Flux query orchestration · MQTT bridge          │
│  Measurements: solarpv · heizung · solarthermie             │
└──────────────────────────┬──────────────────────────────────┘
                           │ Flux queries → JSON context
                           ▼
┌──────────────────────────┬──────────────────────────────────┐
│     ANNOTATION SERVER    │        OBJECT STORE              │
│                          │                                  │
│  Label Studio (8080)     │  MinIO                          │
│  Dataset generation      │  RAID 5 · ~3.6TB usable        │
│  Annotation pipeline     │  Port 9000 API · 9001 UI        │
│  export_annotations.py   │                                  │
│                          │  /exports/training/              │
│  /data/fast/datasets/    │  /adapters/                     │
└──────────┬───────────────┴──────────────────────────────────┘
           │ rsync to cloud GPU
           ▼
┌─────────────────────────────────────────────────────────────┐
│                   CLOUD GPU (on demand)                      │
│  RTX 4090 · ~€0.50/hr · pay only when training             │
│  Unsloth + QLoRA · Qwen2.5-14B-Instruct base               │
│  Output: LoRA adapter (~263MB)                              │
└──────────────────────────┬──────────────────────────────────┘
                           │ adapter rsync back to homelab
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE SERVER                          │
│  Thinkstation P510 · Dual Xeon · 128GB RAM · RTX A2000     │
│  peft merge → fp16 GGUF → q4_k_m GGUF (8.4GB)            │
│  ollama create energy-assistant-v1                          │
│  OpenWebUI → daily queries → feedback loop                  │
└─────────────────────────────────────────────────────────────┘
```

What makes this architecture practical for a homelab is the separation of concerns. Training is the only compute-intensive phase, and it happens rarely (weekly or monthly), on rented cloud hardware, for a few hours at a time. Everything else — data collection, annotation, inference, feedback — runs on hardware you already own, consuming modest power continuously.

---

### Hardware bill of materials

The system uses four distinct roles. Here is what each role requires and what the author uses:

**Metrics server** — any machine capable of running Docker. The author uses a dedicated small-form-factor machine with an Intel processor, 16GB RAM, and a 1TB SSD. InfluxDB, Node-RED, and an MQTT broker run as Docker containers. This machine runs 24/7 and should be power-efficient: a used Intel NUC, a Raspberry Pi 5, or a small refurbished tower all work. Total power draw under normal operation: 10–25W.

**Object store** — a machine with substantial storage capacity and reliable network connectivity. The author uses a rack-mountable server with three 1.8TB drives in a RAID 5 configuration (~3.6TB usable), running MinIO as a bare-metal systemd service on Ubuntu 24.04 LTS. Any machine with a PCIe slot for a RAID card, or software RAID via mdadm, and at least 2–4TB of storage is suitable. RAM requirements are modest: 8–16GB is fine.

**Annotation server** — a general-purpose workstation or server. The author uses a mid-range tower with 32GB RAM and a 1TB NVMe SSD running Ubuntu 24.04 LTS. Label Studio and its PostgreSQL backend run via Docker Compose. llama.cpp is installed natively for dataset generation scripts. This machine does not need a GPU.

**Inference server** — the only machine in the stack that benefits significantly from a GPU. The author uses a dual-socket Xeon workstation with 128GB RAM and an NVIDIA RTX A2000 (12GB VRAM). Ollama serves the quantised model and OpenWebUI provides the chat interface. The 128GB system RAM allows the full fp16 model to be staged during the merge process even when the GPU handles quantisation. A machine with 32–64GB RAM and any NVIDIA GPU with 8GB+ VRAM works for inference; 12GB VRAM is comfortable for q4_k_m at 14B parameters.

**Cloud GPU (rented, on demand)** — an RTX 4090 rented from a GPU cloud provider at approximately €0.50/hr. Training a QLoRA adapter on 2,000–5,000 examples takes 2–4 hours, making the total cost per training run €1–2. This is not a permanent infrastructure cost — the instance is started, used, and terminated. Any provider offering an RTX 4090 or A100 with Ubuntu and sufficient NVMe scratch space works.

---

### Software stack overview

Every component in this series is open-source. Here is the complete software inventory before we begin installing anything:

| Component | Software | Role |
|---|---|---|
| Time-series database | InfluxDB v2 | Stores all sensor measurements |
| IoT orchestration | Node-RED | Flux queries, MQTT bridge, context assembly |
| Message broker | Mosquitto | MQTT broker for sensor data |
| Object store | MinIO | Stores datasets, adapters, exported annotations |
| Annotation tool | Label Studio | Human labeling of QA pairs and preference data |
| Dataset generation | Python + llama.cpp | Synthetic QA generation from sensor data |
| Fine-tuning | Unsloth + QLoRA | Efficient adapter training on cloud GPU |
| Model merging | PEFT + llama.cpp | Merge adapter → GGUF quantisation |
| Inference | Ollama | Serves quantised model |
| Chat interface | OpenWebUI | Daily queries, feedback collection |
| Feedback loop | Label Studio API | Routes user feedback back to annotation queue |

---

### The data flywheel loop — step by step

Understanding the loop before building it prevents architectural mistakes later. Here is exactly how data flows through the system once it is fully operational:

**Step 1 — Sensor data accumulates continuously.** Home Assistant and Node-RED write measurements to InfluxDB every 30–60 seconds. PV yield, battery state of charge (if present), thermal buffer temperature, outdoor temperature, pellet consumption, and grid import/export are the core measurements. After several years this becomes a rich historical record of how the installation behaves across seasons, weather conditions, and usage patterns.

**Step 2 — Context assembly at query time.** When a user sends a question via OpenWebUI, Node-RED executes a set of Flux queries against InfluxDB, assembling a structured JSON context block containing the last 24 hours of relevant measurements, the 7-day averages, and any anomalies. This context is prepended to the user's question before it reaches the model.

**Step 3 — Model generates a recommendation.** Ollama serves the quantised Qwen2.5-14B model. The assembled context plus the user's question are sent as a structured prompt. The model responds with a recommendation, explanation, or diagnostic assessment.

**Step 4 — User provides feedback.** OpenWebUI allows users to rate responses with a thumbs up or down, or leave free-text corrections. A lightweight webhook captures this feedback and routes it to Label Studio, where it appears as a task for review.

**Step 5 — Annotation and dataset curation.** In Label Studio, annotators (in a homelab context, usually just the owner) review flagged responses, correct wrong answers, and mark good examples. A dataset generation script also runs periodically, synthesising new QA pairs from recent sensor data using llama.cpp locally, diversifying the training set beyond just corrected failures.

**Step 6 — Training run.** When enough new labeled examples have accumulated (typically 200–500 new pairs), the dataset is exported from Label Studio, synced to a cloud GPU instance, and a new QLoRA adapter is trained on top of the Qwen2.5-14B base. Training takes 2–4 hours. The resulting adapter is approximately 263MB.

**Step 7 — Merge, quantise, deploy.** The adapter is synced back to the inference server, merged with the base model using PEFT, converted to GGUF format via llama.cpp, quantised to q4_k_m, and loaded into Ollama. The new model version is live within an hour of training completing.

**Step 8 — The loop repeats.** The improved model generates better responses, users interact with it, feedback accumulates, and the next training run begins when the threshold is reached.

---

### What this series covers — and what it does not

Each article in this series is a complete, executable guide. You should be able to follow it on your own hardware and arrive at a working system by the end. Here is the scope:

**Article 2 — The metrics server and data foundation** covers InfluxDB v2 setup on Docker, the Node-RED Flux query patterns for assembling model context, and the MQTT bridge configuration. By the end of Article 2 you have 3.5 years of sensor history queryable and a Node-RED flow that can assemble a structured context block on demand.

**Article 3 — Object store and annotation infrastructure** covers MinIO installation on Ubuntu 24.04 LTS, the bucket structure for a training pipeline, and Label Studio setup with full MinIO integration. By the end of Article 3 you can store datasets and serve annotation tasks backed by your object store.

**Article 4 — Dataset generation and fine-tuning** covers the synthetic QA generation script using llama.cpp, the annotation workflow in Label Studio, exporting datasets, syncing to a cloud GPU, and running QLoRA fine-tuning with Unsloth. By the end of Article 4 you have a trained LoRA adapter.

**Article 5 — Model deployment and closing the loop** covers merging the adapter, GGUF quantisation, Ollama deployment, OpenWebUI setup, and the feedback collection webhook that routes user interactions back to Label Studio. By the end of Article 5 your flywheel is spinning.

What this series does not cover: Kubernetes, cloud-native infrastructure, managed ML platforms, or any paid services beyond the rented GPU hours. Everything runs on hardware you own.

---

### Before you begin: prerequisites

To follow this series you need:

**Hardware (minimum):** one machine capable of running Docker with 16GB RAM and 100GB SSD (metrics server); one machine with 2–4TB storage (object store); one machine with 16–32GB RAM and 500GB NVMe (annotation server); one machine with a GPU of 8GB+ VRAM and 32GB+ system RAM (inference server). These can be consolidated onto fewer machines if yours are well-specced.

**Operating system:** Ubuntu 24.04 LTS on all server roles. The series assumes a fresh install on each machine with a non-root user that has sudo privileges.

**Accounts:** a GPU cloud provider account (RunPod, Vast.ai, Lambda Labs, or similar). You will need a credit card on file but will not be charged until you start an instance.

**Domain knowledge:** basic Linux command-line comfort (navigating directories, editing files with nano or vim, running systemd commands). Python experience is helpful for Articles 3–5 but the scripts are provided in full and annotated.

**Time:** allow 2–3 hours per article for a first read-through and installation. Articles 4 and 5 involve waiting periods (training, quantisation) where the machine works while you do something else.

---

### A note on privacy and data sovereignty

Everything described in this series runs on hardware you physically control. Your energy consumption data, your home's thermal behaviour, your usage patterns — none of it leaves your network except during the training run, when anonymised question-answer pairs are sent to a rented GPU instance. Even that transfer can be eliminated if you have access to a local GPU with sufficient VRAM; the cloud GPU is a cost optimisation, not an architectural requirement.

The model weights, once trained, live on your inference server. Queries from OpenWebUI are processed locally. There is no API call to an external service, no usage logging, no subscription. This is the practical argument for running your own fine-tuned model rather than using a commercial assistant: your home's data stays in your home.

---

### What comes next

In Article 2 we install InfluxDB v2, configure Node-RED to bridge your MQTT sensors, and build the Flux queries that will assemble context blocks for the model. If you already have InfluxDB running with historical data, Article 2 will mostly be familiar — focus on the context assembly section, which is the piece that connects your sensor history to the language model.

If you are starting from scratch, Article 2 is where the foundation is laid. Take it slowly, verify each component before moving to the next, and make sure your sensor data is flowing before proceeding to Article 3.

*Article 2 will be published next week. The companion GitHub repository, containing all scripts, Compose files, and configuration templates referenced in this series, is available at [github.com/your-username/energy-flywheel] — star it to follow along.*

---

*The author runs this system in a private homelab. All IP addresses, hostnames, and identifying details in the architecture diagrams have been replaced with generic placeholders. The sensor data used for training is from a real installation but contains no personally identifiable information.*

*Series: Building an AI Data Flywheel for Home Energy Management*
*Article 1 of 5 — The Concept, the Use Case, and the Architecture*
