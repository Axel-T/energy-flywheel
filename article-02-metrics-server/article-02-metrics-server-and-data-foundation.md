# Building an AI Data Flywheel for Home Energy Management
## Part 2 of 5: The Metrics Server and Data Foundation

*This is the second article in a five-part click-along series. If you have not read Article 1, start there — it establishes the architecture, the hardware bill of materials, and the overall flywheel concept this article builds on.*

---

### What we are building in this article

By the end of this article you will have:

- InfluxDB v2 running in Docker, storing measurements from your PV system, solar thermal collectors, and pellet heating
- Node-RED bridging your MQTT sensor data into InfluxDB continuously
- Three Flux queries — one per energy subsystem — that retrieve structured snapshots of your installation's current state
- A Node-RED flow that assembles those snapshots into a single JSON context block, ready to be prepended to a language model prompt

If you already have InfluxDB running with historical data, the first two sections will be a review. Skip to **Context assembly** — that is the piece that is specific to this series and is not covered in standard InfluxDB tutorials.

---

### Why InfluxDB and not a relational database?

The short answer is that time-series data has different access patterns than relational data, and InfluxDB is built for those patterns.

Your energy sensors generate measurements at regular intervals — a PV inverter might write yield and power every 10 seconds, a thermal sensor every 30 seconds, a pellet boiler every 60 seconds. Over three years that accumulates into tens of millions of rows. Querying that data in PostgreSQL with aggregations over time windows (mean power over the last hour, peak yield per day, rolling 7-day average consumption) requires careful indexing and query planning. InfluxDB handles these queries natively and efficiently because time is a first-class citizen of its data model.

The Flux query language, introduced in InfluxDB v2, makes the kind of context assembly we need for the language model straightforward to express. A query like "give me the mean, min, and max thermal buffer temperature for each hour of the past 24 hours, alongside the corresponding outdoor temperature" is a handful of lines of Flux and returns in milliseconds even against years of history.

---

### Step 1: Install Docker on the metrics server

The metrics server runs InfluxDB, Node-RED, and Mosquitto as Docker containers. This keeps dependencies isolated and makes upgrades straightforward.

```bash
# Update package index
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install -y ca-certificates curl gnupg lsb-release
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin

# Add your user to the docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
docker compose version
```

---

### Step 2: Create the Docker Compose stack

Create a directory for the metrics stack and write the Compose file:

```bash
mkdir -p ~/metrics-server
cd ~/metrics-server
```

```bash
nano docker-compose.yml
```

```yaml
services:

  influxdb:
    image: influxdb:2.7
    container_name: influxdb
    restart: unless-stopped
    ports:
      - "8086:8086"
    volumes:
      - ./influxdb-data:/var/lib/influxdb2
      - ./influxdb-config:/etc/influxdb2
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=${INFLUXDB_PASSWORD}
      - DOCKER_INFLUXDB_INIT_ORG=homelab
      - DOCKER_INFLUXDB_INIT_BUCKET=energy
      - DOCKER_INFLUXDB_INIT_RETENTION=0    # 0 = retain forever
      - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=${INFLUXDB_TOKEN}

  mosquitto:
    image: eclipse-mosquitto:2
    container_name: mosquitto
    restart: unless-stopped
    ports:
      - "1883:1883"
      - "9001:9001"
    volumes:
      - ./mosquitto/config:/mosquitto/config
      - ./mosquitto/data:/mosquitto/data
      - ./mosquitto/log:/mosquitto/log

  nodered:
    image: nodered/node-red:latest
    container_name: nodered
    restart: unless-stopped
    ports:
      - "1880:1880"
    volumes:
      - ./nodered-data:/data
    environment:
      - TZ=Europe/Berlin
    depends_on:
      - influxdb
      - mosquitto
```

Create the environment file:

```bash
nano .env
```

```bash
INFLUXDB_PASSWORD=choose-a-strong-password
INFLUXDB_TOKEN=choose-a-long-random-token-string
```

```bash
chmod 600 .env
```

Create the Mosquitto configuration:

```bash
mkdir -p mosquitto/config
nano mosquitto/config/mosquitto.conf
```

```
listener 1883
allow_anonymous true

listener 9001
protocol websockets

persistence true
persistence_location /mosquitto/data/
log_dest file /mosquitto/log/mosquitto.log
```

Start the stack:

```bash
docker compose up -d

# Watch startup — InfluxDB takes ~15 seconds to initialise on first run
docker compose logs -f influxdb
```

Once you see `InfluxDB is ready` in the logs, open the web UI at `http://metrics-server-ip:8086` and log in with the credentials you set in `.env`.

---

### Step 3: Create InfluxDB buckets for each subsystem

The author's installation uses three measurements, each in its own bucket for clean separation:

| Bucket | Measurement | Data |
|---|---|---|
| `solarpv` | `solarpv` | PV yield (kWh), AC power (W), DC power (W), inverter status |
| `solarthermie` | `solarthermie` | Collector temperature, buffer temperature (top/mid/bottom), flow rate, pump status |
| `heizung` | `heizung` | Pellet consumption (kg), boiler temperature, flow/return temps, burner runtime |

Create the buckets via the InfluxDB CLI inside the container:

```bash
# Enter the InfluxDB container
docker exec -it influxdb bash

# Create buckets (retention 0 = forever)
influx bucket create \
  --name solarpv \
  --org homelab \
  --retention 0 \
  --token $DOCKER_INFLUXDB_INIT_ADMIN_TOKEN

influx bucket create \
  --name solarthermie \
  --org homelab \
  --retention 0 \
  --token $DOCKER_INFLUXDB_INIT_ADMIN_TOKEN

influx bucket create \
  --name heizung \
  --org homelab \
  --retention 0 \
  --token $DOCKER_INFLUXDB_INIT_ADMIN_TOKEN

exit
```

You can also create buckets through the web UI: **Data → Buckets → Create Bucket**.

---

### Step 4: Configure Node-RED as the MQTT bridge

Node-RED reads sensor values arriving on MQTT and writes them to InfluxDB. Install the InfluxDB node:

Open Node-RED at `http://metrics-server-ip:1880`, then go to the hamburger menu → **Manage palette** → **Install** and search for:

- `node-red-contrib-influxdb` — InfluxDB v2 read/write nodes

Once installed, build the following flow for the PV subsystem. You will repeat the same pattern for `solarthermie` and `heizung`.

**PV MQTT → InfluxDB flow:**

In Node-RED, create this flow by importing the JSON below (**Menu → Import**):

```json
[
  {
    "id": "mqtt-pv-in",
    "type": "mqtt in",
    "topic": "homeassistant/sensor/pv/#",
    "broker": "mosquitto-broker",
    "name": "PV sensors (MQTT)"
  },
  {
    "id": "parse-pv",
    "type": "function",
    "name": "Parse PV payload",
    "func": "// Home Assistant MQTT payloads arrive as JSON strings\nconst payload = JSON.parse(msg.payload);\nconst topic = msg.topic;\n\n// Extract field name from topic\n// e.g. homeassistant/sensor/pv/yield_today → yield_today\nconst field = topic.split('/').pop();\n\nmsg.payload = [\n  {\n    measurement: 'solarpv',\n    tags: { source: 'homeassistant' },\n    fields: { [field]: parseFloat(payload.state) },\n    timestamp: new Date()\n  }\n];\nreturn msg;"
  },
  {
    "id": "influx-pv-out",
    "type": "influxdb out",
    "influxdb": "influxdb-v2-config",
    "bucket": "solarpv",
    "org": "homelab",
    "name": "Write to solarpv"
  }
]
```

Configure the InfluxDB connection node with:
- **Version:** 2.0
- **URL:** `http://influxdb:8086`
- **Token:** your admin token from `.env`
- **Organisation:** `homelab`

Repeat this flow pattern for `solarthermie` (subscribing to `homeassistant/sensor/solar_thermal/#`) and `heizung` (subscribing to `homeassistant/sensor/heating/#`). Adjust topic paths to match your Home Assistant MQTT discovery configuration.

**Verifying data is flowing:**

In InfluxDB's web UI, open **Data Explorer** and run a quick check:

```flux
from(bucket: "solarpv")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "solarpv")
  |> limit(n: 10)
```

If rows appear, your bridge is working. If nothing appears after a few minutes, check the Node-RED debug panel for MQTT connection errors or payload parse failures.

---

### Step 5: Importing historical data

If your sensors have been writing to a different system (a previous InfluxDB instance, a CSV export from Home Assistant, or another time-series database), now is the time to import that history. Three years of data is what gives the language model meaningful context — without it, the assistant can only reason about the present, not about seasonal patterns or year-on-year trends.

**From an existing InfluxDB v2 instance:**

```bash
# On the source machine — export a bucket
influx backup \
  --bucket solarpv \
  --host http://old-server:8086 \
  --token your-old-token \
  /tmp/solarpv-backup/

# Rsync to new metrics server
rsync -avz /tmp/solarpv-backup/ \
  user@metrics-server:/tmp/solarpv-backup/

# On the new metrics server — restore into the new bucket
docker exec -it influxdb \
  influx restore \
  --bucket solarpv \
  --org homelab \
  --token $DOCKER_INFLUXDB_INIT_ADMIN_TOKEN \
  /tmp/solarpv-backup/
```

**From CSV (Home Assistant recorder exports or third-party tools):**

```python
# import_csv_to_influx.py
# Reads a CSV with columns: timestamp, field_name, value
# Writes to InfluxDB v2 via the Python client

import csv
from datetime import datetime
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

INFLUX_URL   = "http://metrics-server-ip:8086"
INFLUX_TOKEN = "your-admin-token"
INFLUX_ORG   = "homelab"
INFLUX_BUCKET = "solarpv"

client = InfluxDBClient(
    url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG
)
write_api = client.write_api(write_options=SYNCHRONOUS)

with open("solarpv_history.csv") as f:
    reader = csv.DictReader(f)
    batch = []
    for i, row in enumerate(reader):
        point = (
            Point("solarpv")
            .tag("source", "import")
            .field(row["field_name"], float(row["value"]))
            .time(
                datetime.fromisoformat(row["timestamp"]),
                WritePrecision.SECONDS
            )
        )
        batch.append(point)
        # Write in batches of 5000 to avoid memory pressure
        if len(batch) >= 5000:
            write_api.write(bucket=INFLUX_BUCKET, record=batch)
            batch = []
            print(f"  wrote {i} rows...")
    if batch:
        write_api.write(bucket=INFLUX_BUCKET, record=batch)

print("Import complete")
client.close()
```

```bash
pip install influxdb-client
python3 import_csv_to_influx.py
```

---

### Step 6: The context assembly pattern

This is the section that distinguishes this series from a standard InfluxDB tutorial. The language model needs structured, interpretable context — not raw time-series data, but a summarised snapshot that tells it what the installation is doing right now and how that compares to recent history.

The context block we assemble looks like this:

```json
{
  "generated_at": "2024-03-15T14:32:00Z",
  "solarpv": {
    "current_power_w": 3840,
    "yield_today_kwh": 18.4,
    "yield_7d_avg_kwh": 21.2,
    "yield_30d_avg_kwh": 19.8,
    "peak_power_today_w": 5120,
    "status": "producing"
  },
  "solarthermie": {
    "collector_temp_c": 78.3,
    "buffer_top_c": 67.1,
    "buffer_mid_c": 54.8,
    "buffer_bottom_c": 41.2,
    "pump_running": true,
    "buffer_7d_high_c": 72.4,
    "buffer_7d_low_c": 38.6
  },
  "heizung": {
    "boiler_temp_c": 68.4,
    "flow_temp_c": 55.2,
    "return_temp_c": 42.8,
    "pellets_today_kg": 3.2,
    "pellets_7d_avg_kg": 4.1,
    "pellets_30d_avg_kg": 5.8,
    "burner_runtime_today_min": 94,
    "status": "standby"
  },
  "weather": {
    "outdoor_temp_c": 12.4,
    "outdoor_temp_7d_avg_c": 10.1
  }
}
```

This JSON block is roughly 400 tokens — small enough to fit in any model's context window alongside the user's question and the system prompt, but rich enough for the model to perform meaningful multi-variable reasoning.

---

### Step 7: Writing the Flux queries

Each subsystem gets its own Flux query. These run in Node-RED when a user submits a question, and their outputs are merged into the context block above.

**PV context query:**

```flux
// flux/context_solarpv.flux
// Returns current power, today's yield, and rolling averages

today_start = date.truncate(t: now(), unit: 1d)
now_time = now()
ago_7d = date.add(d: -7d, to: now_time)
ago_30d = date.add(d: -30d, to: now_time)

// Current power (most recent reading)
current = from(bucket: "solarpv")
  |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "solarpv"
       and r._field == "ac_power_w")
  |> last()
  |> findRecord(fn: (key) => true, idx: 0)

// Today's yield so far
yield_today = from(bucket: "solarpv")
  |> range(start: today_start)
  |> filter(fn: (r) => r._measurement == "solarpv"
       and r._field == "yield_kwh")
  |> last()
  |> findRecord(fn: (key) => true, idx: 0)

// 7-day average daily yield
yield_7d = from(bucket: "solarpv")
  |> range(start: ago_7d)
  |> filter(fn: (r) => r._measurement == "solarpv"
       and r._field == "yield_kwh")
  |> aggregateWindow(every: 1d, fn: max, createEmpty: false)
  |> mean()
  |> findRecord(fn: (key) => true, idx: 0)

// Return as a structured record
{
  current_power_w: current._value,
  yield_today_kwh: yield_today._value,
  yield_7d_avg_kwh: yield_7d._value
}
```

**Solar thermal context query:**

```flux
// flux/context_solarthermie.flux

// Current buffer temperatures (most recent readings per sensor)
buffer_top = from(bucket: "solarthermie")
  |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "solarthermie"
       and r._field == "buffer_temp_top_c")
  |> last()
  |> findRecord(fn: (key) => true, idx: 0)

buffer_mid = from(bucket: "solarthermie")
  |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "solarthermie"
       and r._field == "buffer_temp_mid_c")
  |> last()
  |> findRecord(fn: (key) => true, idx: 0)

buffer_bottom = from(bucket: "solarthermie")
  |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "solarthermie"
       and r._field == "buffer_temp_bottom_c")
  |> last()
  |> findRecord(fn: (key) => true, idx: 0)

collector = from(bucket: "solarthermie")
  |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "solarthermie"
       and r._field == "collector_temp_c")
  |> last()
  |> findRecord(fn: (key) => true, idx: 0)

// 7-day high/low for buffer top
ago_7d = date.add(d: -7d, to: now())

buffer_top_7d_high = from(bucket: "solarthermie")
  |> range(start: ago_7d)
  |> filter(fn: (r) => r._measurement == "solarthermie"
       and r._field == "buffer_temp_top_c")
  |> max()
  |> findRecord(fn: (key) => true, idx: 0)

buffer_top_7d_low = from(bucket: "solarthermie")
  |> range(start: ago_7d)
  |> filter(fn: (r) => r._measurement == "solarthermie"
       and r._field == "buffer_temp_top_c")
  |> min()
  |> findRecord(fn: (key) => true, idx: 0)

{
  collector_temp_c:  collector._value,
  buffer_top_c:      buffer_top._value,
  buffer_mid_c:      buffer_mid._value,
  buffer_bottom_c:   buffer_bottom._value,
  buffer_7d_high_c:  buffer_top_7d_high._value,
  buffer_7d_low_c:   buffer_top_7d_low._value
}
```

**Pellet heating context query:**

```flux
// flux/context_heizung.flux

today_start = date.truncate(t: now(), unit: 1d)
ago_7d  = date.add(d: -7d,  to: now())
ago_30d = date.add(d: -30d, to: now())

// Current temperatures
boiler = from(bucket: "heizung")
  |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "heizung"
       and r._field == "boiler_temp_c")
  |> last()
  |> findRecord(fn: (key) => true, idx: 0)

// Today's pellet consumption
pellets_today = from(bucket: "heizung")
  |> range(start: today_start)
  |> filter(fn: (r) => r._measurement == "heizung"
       and r._field == "pellets_consumed_kg")
  |> sum()
  |> findRecord(fn: (key) => true, idx: 0)

// 7-day average daily consumption
pellets_7d = from(bucket: "heizung")
  |> range(start: ago_7d)
  |> filter(fn: (r) => r._measurement == "heizung"
       and r._field == "pellets_consumed_kg")
  |> aggregateWindow(every: 1d, fn: sum, createEmpty: false)
  |> mean()
  |> findRecord(fn: (key) => true, idx: 0)

// 30-day average (seasonal baseline)
pellets_30d = from(bucket: "heizung")
  |> range(start: ago_30d)
  |> filter(fn: (r) => r._measurement == "heizung"
       and r._field == "pellets_consumed_kg")
  |> aggregateWindow(every: 1d, fn: sum, createEmpty: false)
  |> mean()
  |> findRecord(fn: (key) => true, idx: 0)

{
  boiler_temp_c:          boiler._value,
  pellets_today_kg:       pellets_today._value,
  pellets_7d_avg_kg:      pellets_7d._value,
  pellets_30d_avg_kg:     pellets_30d._value
}
```

Save these three files to `~/metrics-server/flux/` — they will be referenced by Node-RED in the next step.

---

### Step 8: The Node-RED context assembly flow

This is the flow that runs whenever a user submits a query via OpenWebUI. It executes all three Flux queries in parallel, merges the results, and returns a single JSON context block.

In Node-RED, create a new flow and import the following:

```json
[
  {
    "id": "http-context-in",
    "type": "http in",
    "method": "POST",
    "url": "/api/context",
    "name": "Context request"
  },
  {
    "id": "parallel-split",
    "type": "function",
    "name": "Trigger parallel queries",
    "func": "// Clone message for each subsystem query\nconst pv      = { ...msg, subsystem: 'solarpv' };\nconst thermal = { ...msg, subsystem: 'solarthermie' };\nconst heating  = { ...msg, subsystem: 'heizung' };\nreturn [pv, thermal, heating];",
    "outputs": 3
  },
  {
    "id": "query-pv",
    "type": "influxdb in",
    "influxdb": "influxdb-v2-config",
    "org": "homelab",
    "query": "// contents of flux/context_solarpv.flux",
    "name": "Query PV"
  },
  {
    "id": "query-thermal",
    "type": "influxdb in",
    "influxdb": "influxdb-v2-config",
    "org": "homelab",
    "query": "// contents of flux/context_solarthermie.flux",
    "name": "Query thermal"
  },
  {
    "id": "query-heating",
    "type": "influxdb in",
    "influxdb": "influxdb-v2-config",
    "org": "homelab",
    "query": "// contents of flux/context_heizung.flux",
    "name": "Query heating"
  },
  {
    "id": "join-results",
    "type": "join",
    "mode": "custom",
    "build": "object",
    "property": "subsystem",
    "count": 3,
    "name": "Join all subsystems"
  },
  {
    "id": "assemble-context",
    "type": "function",
    "name": "Assemble context block",
    "func": "const parts = msg.payload;\n\nconst context = {\n  generated_at: new Date().toISOString(),\n  solarpv:      parts.solarpv,\n  solarthermie: parts.solarthermie,\n  heizung:      parts.heizung\n};\n\nmsg.payload = context;\nreturn msg;"
  },
  {
    "id": "http-context-out",
    "type": "http response",
    "name": "Return context JSON"
  }
]
```

Wire the nodes in this order: `http-context-in` → `parallel-split` (three outputs) → `query-pv` / `query-thermal` / `query-heating` → `join-results` → `assemble-context` → `http-context-out`.

Test the endpoint from the command line:

```bash
curl -s -X POST http://metrics-server-ip:1880/api/context \
  | python3 -m json.tool
```

You should receive a fully populated JSON context block within a second or two. If any subsystem returns nulls, check that measurements are flowing into the corresponding InfluxDB bucket and that the field names in your Flux queries match what Node-RED is writing.

---

### Step 9: Firewall configuration

Open only the ports needed by other services in the stack:

```bash
# InfluxDB — accessible from annotation server and inference server
sudo ufw allow from 192.168.0.0/16 to any port 8086 \
  comment 'InfluxDB API'

# Node-RED — accessible from inference server (context API)
sudo ufw allow from 192.168.0.0/16 to any port 1880 \
  comment 'Node-RED context API'

# MQTT — accessible from Home Assistant and all sensors
sudo ufw allow from 192.168.0.0/16 to any port 1883 \
  comment 'Mosquitto MQTT'

sudo ufw enable
sudo ufw status
```

Do not expose InfluxDB or Node-RED to the internet. The context API endpoint returns raw sensor data — it should only be reachable within your LAN.

---

### Verifying the complete foundation

Before moving on to Article 3, confirm all four of these things are true:

**1. Data is flowing from Home Assistant into InfluxDB.** Open the InfluxDB Data Explorer and run a query against each bucket. Each query should return recent rows with timestamps within the last few minutes.

**2. Historical data is present.** Run a query with `range(start: -365d)` and confirm you get meaningful results. If you have three years of history, verify that the 30-day average pellet consumption query returns a sensible seasonal value — not zero and not a wildly incorrect number.

**3. The context API responds correctly.** Run the `curl` command from Step 8 and inspect the output. Every field should be populated with a real number, not `null`. Pay particular attention to the rolling averages — if they are returning zero or null, the aggregation windows in your Flux queries may not match the field names in your data.

**4. The data survives a reboot.** Restart the metrics server and confirm Docker Compose brings all three containers back up automatically, that data is still queryable in InfluxDB, and that the context API still responds within a minute of boot.

```bash
# Test reboot persistence
sudo reboot
# Wait for boot, then:
curl -s -X POST http://metrics-server-ip:1880/api/context \
  | python3 -m json.tool
```

---

### A note on field naming conventions

The Flux queries in this article use field names that match a specific Home Assistant + MQTT Discovery setup. Your field names will almost certainly differ. The important thing is consistency: whatever names your sensors write into InfluxDB, those same names must appear in your Flux queries, and the assembled context block must use human-readable keys that the language model can interpret without additional explanation.

If your pellet boiler writes `pelletverbrauch` rather than `pellets_consumed_kg`, use that field name in the Flux query but map it to `pellets_consumed_kg` in the context assembly function. The model does not know what `pelletverbrauch` means, but it does know what `pellets_consumed_kg` means. This translation step — from machine field names to semantically meaningful keys — is one of the most important things the context assembly flow does.

---

### What comes next

In Article 3 we set up the object store and annotation infrastructure: MinIO on a dedicated storage server, the bucket structure for the training pipeline, and Label Studio wired to read images and text directly from MinIO. By the end of Article 3 you will be able to store training datasets and serve annotation tasks to a web browser, all backed by your own hardware.

If you are following along and already have a MinIO instance running, Article 3's installation section will be a review — focus on the bucket structure and the Label Studio MinIO integration, which are specific to this pipeline.

*The companion GitHub repository contains the complete Flux query files, the Node-RED flow export JSON, the Docker Compose file, and the CSV import script referenced in this article. All field names in the repository use the generic naming convention described above.*

---

*Series: Building an AI Data Flywheel for Home Energy Management*
*Article 2 of 5 — The Metrics Server and Data Foundation*
