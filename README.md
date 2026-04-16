# Energy Flywheel — AI Data Flywheel for Home Energy Management

A self-improving AI recommendation assistant for PV solar, solar thermal, and pellet heating systems — built entirely on self-hosted hardware using open-source software.

This repository contains every script, configuration file, Docker Compose stack, and Node-RED flow referenced in the five-part click-along series **"Building an AI Data Flywheel for Home Energy Management"**.

---

## What this builds

A continuously improving language model assistant that answers questions like:

- *"My PV yield was 38 kWh today but I only consumed 12 kWh. Where did the rest go?"*
- *"The thermal buffer is at 67°C at 14:00. Should I run the dishwasher now or wait?"*
- *"Pellet consumption this week is 18% above last week. Is that expected?"*

The base model is **Qwen2.5-14B-Instruct**, fine-tuned on question-answer pairs derived from real sensor history using QLoRA. Every response the model gives is a potential training example for the next version. The loop never stops.

---

## Architecture

```
Home Assistant · Node-RED · MQTT
        │ time-series writes
        ▼
Metrics server — InfluxDB v2 · Node-RED · Mosquitto
        │ Flux queries → JSON context
        ▼
Object store ──────────────── Annotation server
MinIO (RAID 5 · ~3.6TB)       Label Studio · llama.cpp
exports/ datasets/             Synthetic QA generation
adapters/ models/              export_annotations.py
        │
        │ rsync to cloud GPU
        ▼
Cloud GPU (on demand · ~€0.50/hr)
Unsloth + QLoRA · Qwen2.5-14B-Instruct
LoRA adapter (~263MB)
        │ rsync back to homelab
        ▼
Inference server — Dual Xeon · 128GB RAM · RTX A2000
PEFT merge → fp16 GGUF → q4_k_m GGUF (8.4GB)
Ollama · OpenWebUI
        │ user feedback → Node-RED webhook
        └──────────────────────────────────▶ Label Studio (next cycle)
```

---

## Article index

| Article | Folder | What it builds |
|---|---|---|
| [Article 1 — Concept and architecture]([https://medium.com/](https://medium.com/@athobaben_56166/building-an-ai-data-flywheel-for-home-energy-management-9b7c1a0f63ae)) | `article-01-concept-and-architecture/` | Architecture overview, hardware BOM, software inventory |
| [Article 2 — Metrics server]([https://medium.com/](https://medium.com/@athobaben_56166/building-an-ai-data-flywheel-for-home-energy-management-49d2ed62acb2)) | `article-02-metrics-server/` | InfluxDB, Node-RED, Mosquitto, Flux queries, context API |
| [Article 3 — Object store and annotation]([https://medium.com/](https://medium.com/@athobaben_56166/building-an-ai-data-flywheel-for-home-energy-management-ae517b63ac8b)) | `article-03-object-store-annotation/` | MinIO, Label Studio, annotation project setup |
| [Article 4 — Dataset generation and fine-tuning](https://medium.com/) | `article-04-dataset-and-finetuning/` | Synthetic QA generation, export, QLoRA training |
| [Article 5 — Deployment and feedback loop](https://medium.com/) | `article-05-deployment-and-loop/` | PEFT merge, Ollama, OpenWebUI, feedback webhook |

*Update the links above once articles are published.*

---

## Hardware requirements

| Role | What it runs | Minimum spec |
|---|---|---|
| Metrics server | InfluxDB v2, Node-RED, Mosquitto | Docker-capable machine, 16GB RAM, 1TB SSD |
| Object store | MinIO on Ubuntu 24.04 | 2–4TB storage, 8GB RAM |
| Annotation server | Label Studio, PostgreSQL, llama.cpp | 16–32GB RAM, 500GB NVMe |
| Inference server | Ollama, OpenWebUI | GPU ≥8GB VRAM, 32GB+ system RAM |
| Cloud GPU | Unsloth QLoRA training | RTX 4090 or A100, rented on demand |

---

## Quick start

**Already have InfluxDB with sensor history?**
→ Start at [Article 3](https://medium.com/) — your data foundation is in place.

**Starting from scratch?**
→ Start at [Article 1](https://medium.com/) and follow in order. Each article ends with a verification checklist before the next one builds on top of it.

**Just want to run the fine-tuning script?**
→ See `article-04-dataset-and-finetuning/` — the `data/sample_tasks.jsonl` file lets you test the pipeline without real sensor data.

---

## Repository conventions

- **`.env.example`** files contain placeholder values — copy to `.env` and fill in your own credentials before running any Compose stack.
- **Node-RED flow JSON** files can be imported directly: Node-RED menu → Import → select the `.json` file.
- **Field names** in Flux queries and the context assembly function use generic semantic names. See `docs/field-naming.md` for how to map your sensor's field names to the names used in the scripts.
- **Python version:** 3.10 or later. Each article folder with Python scripts has its own `requirements.txt`.
- **Docker Compose version:** v2 (`docker compose` not `docker-compose`).

---

## CHANGELOG

| Date | Change |
|---|---|
| 2025-03-31 | Initial release — all five articles |

---

## Licence

MIT — see `LICENSE`.
