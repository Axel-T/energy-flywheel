# Hardware sourcing guide

This guide covers where to source the secondhand hardware used in this
series and what to look for when buying each component.

All prices are approximate European market rates as of early 2025.
eBay Kleinanzeigen (Germany), Ricardo (Switzerland), and eBay.de are
the most reliable sources for this hardware in the DACH region.
UK readers should check eBay.co.uk and CEX. US readers: eBay and
ServerMonkey.

---

## Metrics server

**What you need:** any low-power machine capable of running Docker with
16GB RAM and a 1TB SSD. This machine runs 24/7 so power efficiency matters
more than performance.

**Good options:**

| Machine | Cores | RAM (max) | TDP | Price |
|---|---|---|---|---|
| Intel NUC 10/11 | 4–6 | 64GB | 28W | €80–150 |
| Raspberry Pi 5 (8GB) | 4 (ARM) | 8GB | 12W | €80 new |
| HP ProDesk 400 G6 Mini | 4–6 | 64GB | 35W | €60–120 |
| Lenovo ThinkCentre M720q | 4–6 | 32GB | 35W | €60–100 |

**What to check:** confirm the machine has an M.2 NVMe slot (most do),
that the RAM is not soldered (NUCs sometimes have one soldered + one slot),
and that it boots from USB (needed to install Ubuntu).

---

## Object store

**What you need:** a machine with 3–4 drive bays, 8–16GB RAM, a PCIe slot
or onboard RAID support, and reliable 24/7 operation. NAS-rated drives
strongly preferred.

**Good options:**

| Machine | Drive bays | RAM (max) | Notes |
|---|---|---|---|
| HPE ProLiant ML30 Gen9 | 4 LFF | 64GB ECC | Good value, iLO management |
| HPE ProLiant ML30 Gen10 | 4 LFF | 64GB ECC | Newer, same form factor |
| Synology DS423+ | 4 | N/A (NAS OS) | If you prefer a NAS appliance |
| Custom ATX build | As many as case fits | Unlimited | Most flexible |

**What to check on the ML30:** confirm it has the 350W PSU (not 250W) and
that the drive backplane is present. Some stripped units are sold without
it. iLO is worth having — remote console access to a headless server is
genuinely useful.

**Drives:** use NAS-rated drives only in the RAID array.
Desktop drives (WD Blue, Seagate Barracuda) have TLER (Time-Limited Error
Recovery) settings that cause ZFS and mdadm to drop them from the array
during normal error recovery.

Recommended drives for this use case:
- WD Red Plus 4TB (CMR, not SMR — check the model suffix)
- Seagate IronWolf 4TB
- Avoid: WD Red (SMR versions), WD Red Pro (expensive, overkill)

Approximate cost for three 4TB NAS drives: €180–240.

---

## Annotation server

**What you need:** a general-purpose server or workstation with 32–48GB RAM,
a 500GB+ NVMe SSD, and reliable Ethernet. No GPU required. This machine
runs Label Studio, PostgreSQL, and llama.cpp for dataset generation (CPU only).

**Good options:**

| Machine | Cores | RAM (max) | Notes |
|---|---|---|---|
| HP ProLiant ML30 Gen9 (48GB config) | 4 | 64GB ECC | Ready to use, iLO |
| Dell PowerEdge T30 | 4 | 64GB ECC | Very quiet, good for home |
| Lenovo ThinkStation P310 | 4–6 | 64GB | SFF or tower, PCIe slots |
| Any Xeon E3-1200 v5/v6 workstation | 4 | 64GB ECC | Many options secondhand |

**What to check:** Elasticsearch (used by Argilla) requires at least 4GB
of JVM heap, so 16GB system RAM is the absolute floor. 32–48GB is comfortable.
Confirm the NVMe slot is available (some budget servers only have SATA).

**The ML30 Gen9 with 48GB (2×16GB + 2×8GB)** is a particularly good fit —
it can be sourced with RAM already installed at the right capacity, iLO
provides remote console access, and it fits neatly in a home office.

---

## Inference server

**What you need:** a machine with a GPU of at least 8GB VRAM (12GB
recommended for Qwen2.5-14B at q4_k_m), 32–128GB system RAM (more is better
for the merge step), and sufficient PCIe bandwidth for the GPU.

**Good options:**

| Machine | GPU options | System RAM | Notes |
|---|---|---|---|
| Lenovo ThinkStation P510 | Up to RTX 3090 | Up to 128GB DDR4 | Good value, proprietary PSU limits GPU choice |
| Dell Precision T7920 | Up to RTX 4090 | Up to 384GB DDR4 | 1400W PSU, best GPU compatibility |
| DIY ATX build | Any | Unlimited | Most flexibility, no OEM constraints |
| HP Z4 G4 | Up to RTX 3090 | Up to 128GB DDR4 | 1000W PSU, solid workstation |

**GPU selection for Qwen2.5-14B q4_k_m (8.4GB model):**

| GPU | VRAM | Fits model | Notes |
|---|---|---|---|
| RTX A2000 12GB | 12GB | Yes | Efficient, low-profile, quiet |
| RTX 3090 | 24GB | Yes, comfortably | Best secondhand value for VRAM |
| RTX 4090 | 24GB | Yes, comfortably | Fastest, needs 450W PSU |
| RTX 3080 | 10GB | Tight (8.4GB model) | Workable but little headroom |
| RTX 3070 | 8GB | No (insufficient) | Cannot run the 14B model |

**PSU considerations:** the RTX 4090 requires 450W at peak. Only the Dell
T7920 (1400W PSU) handles this comfortably among common workstations.
The P510 tops out safely at RTX 3090 (350W) with its 850W PSU.

---

## Cloud GPU (training only)

No hardware to buy — rented on demand and terminated after training.

**Recommended providers (EU-friendly):**

| Provider | RTX 4090 price | Notes |
|---|---|---|
| RunPod | ~€0.44/hr | Good EU availability, fast instance start |
| Vast.ai | ~€0.30–0.50/hr | Cheapest but variable quality |
| Lambda Labs | ~€0.60/hr | More reliable, US-focused |
| Hetzner Cloud GPU | ~€2.30/hr (A100) | EU data residency, pricier |

For a 2–4 hour training run, total cost is €1–2 on RunPod or Vast.ai.
Always terminate the instance immediately after training completes.

**What to select:** Ubuntu 22.04 or 24.04 template, 100GB+ NVMe scratch,
CUDA 12.x pre-installed. Confirm the instance has internet access for the
Hugging Face model download on the first run (~28GB for Qwen2.5-14B).

---

## Total approximate cost (secondhand, excluding drives)

| Role | Machine | Est. cost |
|---|---|---|
| Metrics server | Intel NUC 11 / HP ProDesk | €80–120 |
| Object store | HPE ML30 Gen9 | €80–150 |
| 3× NAS drives (4TB each) | WD Red Plus or IronWolf | €180–240 |
| Annotation server | HP ML30 Gen9 with 48GB | €120–200 |
| Inference server | Lenovo P510 + RTX 3090 | €250–500 |
| **Total** | | **€710–1,210** |

Cloud GPU training: €1–2 per run, estimated 6–10 runs per year = **€6–20/yr**.

This compares favourably with a commercial AI assistant subscription at
€20+/month, with the significant advantage that all data stays on your
own hardware and the model improves for your specific installation over time.
