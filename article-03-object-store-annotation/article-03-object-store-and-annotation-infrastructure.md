# Building an AI Data Flywheel for Home Energy Management
## Part 3 of 5: Object Store and Annotation Infrastructure

*This is the third article in a five-part click-along series. Articles 1 and 2 established the concept, the hardware, and the metrics server. This article builds the storage and annotation layer that sits between your sensor data and your training pipeline.*

---

### What we are building in this article

By the end of this article you will have:

- MinIO running as a bare-metal systemd service on Ubuntu 24.04 LTS, backed by a RAID array
- A bucket structure designed specifically for a fine-tuning pipeline: raw exports, curated datasets, trained adapters, and deployed models each in their own bucket
- Label Studio running via Docker Compose, with PostgreSQL as its database backend
- Label Studio wired directly to MinIO as its storage backend, so annotation tasks reference objects in your object store rather than files copied into the Label Studio container
- A working annotation project configured for the energy QA labeling task this series uses

If you already have MinIO running, jump to **Bucket structure for the training pipeline** — the layout described there is specific to this series and differs from a generic MinIO setup. If you already have Label Studio running, jump to **Connecting Label Studio to MinIO**.

---

### Why a dedicated object store?

Before installing anything, it is worth understanding why the architecture uses a dedicated object store rather than a shared filesystem or NFS mount.

The training pipeline generates several distinct categories of artefact: raw question-answer exports from the context assembly flow, curated and labelled datasets ready for training, LoRA adapter checkpoints from completed training runs, and quantised GGUF models ready for deployment. Each of these has a different lifecycle, different access pattern, and different retention requirement. A shared filesystem treats all of them identically. An object store with a deliberate bucket structure lets you apply lifecycle rules, access controls, and retention policies per category — and the S3-compatible API means every tool in the pipeline (Label Studio, Python scripts, rsync-over-mc, the inference server) uses the same interface to read and write artefacts regardless of which machine it runs on.

MinIO is the natural choice for a homelab: it is a single binary, it speaks the S3 API natively, it runs as a systemd service without Docker, and its web console is clear enough to use without consulting documentation. For a RAID-backed storage server with 3–4TB of usable space, MinIO's single-node single-drive (or single-node multi-drive) mode is exactly the right deployment pattern.

---

### Step 1: Prepare storage on the object store server

The object store server in this series uses three 1.8TB drives in a RAID 5 configuration, giving approximately 3.6TB of usable space with single-drive fault tolerance. The steps below use `mdadm` software RAID, which works on any server without a hardware RAID card.

**Check your drive device names first:**

```bash
lsblk
# Look for your three storage drives — typically /dev/sdb, /dev/sdc, /dev/sdd
# Confirm they are the correct devices before proceeding
```

**Create the RAID 5 array:**

```bash
sudo apt update && sudo apt install -y mdadm

# Create the array (adjust device names to match your hardware)
sudo mdadm --create /dev/md0 \
  --level=5 \
  --raid-devices=3 \
  /dev/sdb /dev/sdc /dev/sdd

# Watch the sync progress — this takes 1-3 hours for 1.8TB drives
watch -n 10 cat /proc/mdstat
```

Do not proceed to the next step until the sync completes. The array is usable during sync but writing large amounts of data before it finishes degrades performance significantly.

**Format and mount:**

```bash
# Format as XFS — well-suited to MinIO's large sequential writes
sudo apt install -y xfsprogs
sudo mkfs.xfs -L minio-store /dev/md0

# Create mount point
sudo mkdir -p /mnt/data

# Add to fstab for automatic mounting on boot
echo "LABEL=minio-store /mnt/data xfs defaults,noatime 0 2" \
  | sudo tee -a /etc/fstab

sudo mount -a

# Verify
df -h /mnt/data
```

**Save the mdadm configuration so the array survives reboots:**

```bash
sudo mdadm --detail --scan | sudo tee -a /etc/mdadm/mdadm.conf
sudo update-initramfs -u
```

---

### Step 2: Install MinIO as a systemd service

MinIO is distributed as a single binary with a matching `.deb` package that installs a systemd unit automatically. This is the cleanest installation method on Ubuntu 24.04.

```bash
# Download and install the .deb package
curl -O https://dl.min.io/server/minio/release/linux-amd64/minio_latest_amd64.deb
sudo dpkg -i minio_latest_amd64.deb

# Verify
minio --version
```

**Create the MinIO user and data directory:**

```bash
sudo useradd --system --shell /sbin/nologin minio-user
sudo mkdir -p /mnt/data/minio
sudo chown -R minio-user:minio-user /mnt/data/minio
```

**Write the environment file:**

```bash
sudo mkdir -p /etc/minio
sudo nano /etc/minio/minio.env
```

```bash
# /etc/minio/minio.env

MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=choose-a-strong-password-here

# Point at the RAID array
MINIO_VOLUMES="/mnt/data/minio"

# API on 9000, web console on 9001
MINIO_OPTS="--console-address :9001"

MINIO_SITE_NAME="energy-flywheel"
```

```bash
sudo chmod 600 /etc/minio/minio.env
```

**Configure the systemd service** to read from that environment file. The `.deb` installation creates the unit at `/usr/lib/systemd/system/minio.service` — check it points at the right environment file:

```bash
sudo systemctl cat minio
```

If the `EnvironmentFile` line does not point at `/etc/minio/minio.env`, edit it:

```bash
sudo systemctl edit minio --full
```

The relevant section should read:

```ini
[Service]
User=minio-user
Group=minio-user
EnvironmentFile=/etc/minio/minio.env
ExecStart=/usr/local/bin/minio server $MINIO_OPTS $MINIO_VOLUMES
Restart=always
LimitNOFILE=65536
TasksMax=infinity
TimeoutStopSec=infinity
SendSIGKILL=no
```

**Start and enable:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable minio
sudo systemctl start minio

# Watch startup logs
sudo journalctl -u minio -f
```

You should see MinIO report its API and WebUI addresses within a few seconds. Open the web console at `http://object-store-ip:9001` and log in with the credentials from your environment file.

---

### Step 3: Install the MinIO client

`mc` is the command-line tool for administering MinIO — creating buckets, managing users, setting policies, and copying objects. Install it on the object store server and on any other machine in the stack that needs to interact with MinIO directly (the annotation server and inference server, at minimum).

```bash
curl -O https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/

# Register an alias pointing at your local MinIO instance
mc alias set local http://localhost:9000 \
  admin your-password-here

# Verify
mc admin info local
```

---

### Step 4: Bucket structure for the training pipeline

A flat bucket layout works for simple object storage, but the training pipeline has four distinct categories of data with different access patterns and retention requirements. Creating a dedicated bucket per category makes lifecycle management clean and access controls explicit.

```bash
# Raw question-answer exports from the context assembly flow
mc mb local/exports

# Curated, labelled datasets ready for training
mc mb local/datasets

# LoRA adapter checkpoints from completed training runs
mc mb local/adapters

# Quantised GGUF models ready for Ollama deployment
mc mb local/models
```

**Lifecycle rules** — raw exports are cheap to regenerate and accumulate quickly. Set them to expire after 90 days so the bucket does not grow unbounded:

```bash
mc ilm rule add \
  --expire-days 90 \
  local/exports
```

**Verify the layout:**

```bash
mc ls local
# Should show: exports  datasets  adapters  models
```

This is the layout the export, training, and deployment scripts in Articles 4 and 5 expect. If you already have a MinIO instance with a different layout, either adjust the scripts to match or rename your buckets now — consistency here saves debugging later.

---

### Step 5: Open the firewall

MinIO should be accessible from the annotation server and the inference server, but not from the internet:

```bash
# Allow MinIO API from LAN
sudo ufw allow from 192.168.0.0/16 to any port 9000 \
  comment 'MinIO API'

# Allow MinIO console from LAN
sudo ufw allow from 192.168.0.0/16 to any port 9001 \
  comment 'MinIO console'

sudo ufw enable
sudo ufw status
```

---

### Step 6: Install Label Studio on the annotation server

Label Studio runs on a separate machine from the object store. In this series that is a general-purpose server with at least 16GB RAM — the machine does not need a GPU but Elasticsearch (used by Argilla, covered in Article 4) needs at least 4GB of heap, so headroom matters.

**Install Docker** on the annotation server using the same steps as Article 2, Step 1. Then create the Label Studio directory:

```bash
mkdir -p ~/label-studio
cd ~/label-studio
```

**Write the Docker Compose file:**

```bash
nano docker-compose.yml
```

```yaml
services:

  app:
    image: heartexlabs/label-studio:latest
    container_name: label-studio
    restart: unless-stopped
    ports:
      - "8080:8080"
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      - DJANGO_DB=default
      - POSTGRE_HOST=postgres
      - POSTGRE_PORT=5432
      - POSTGRE_USER=labelstudio
      - POSTGRE_PASSWORD=${LS_DB_PASSWORD}
      - POSTGRE_NAME=labelstudio
      - LABEL_STUDIO_HOST=http://annotation-server-ip:8080
      - SECRET_KEY=${LS_SECRET_KEY}
      - SSRF_PROTECTION_ENABLED=true
    volumes:
      - ./label-studio-data:/label-studio/data
    networks:
      - ls-network

  postgres:
    image: postgres:16
    container_name: ls-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_USER=labelstudio
      - POSTGRES_PASSWORD=${LS_DB_PASSWORD}
      - POSTGRES_DB=labelstudio
    volumes:
      - ./postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U labelstudio"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - ls-network

networks:
  ls-network:
    driver: bridge
```

**Write the environment file:**

```bash
nano .env
```

```bash
# .env — do not commit this to version control
LS_DB_PASSWORD=choose-a-strong-database-password
LS_SECRET_KEY=choose-a-long-random-secret-key
```

```bash
chmod 600 .env
```

**Start the stack:**

```bash
docker compose up -d

# Watch startup — PostgreSQL initialises first, then Label Studio
docker compose logs -f app
```

After about 30 seconds you should see Label Studio report that it is listening on port 8080. Open `http://annotation-server-ip:8080` in a browser and create your admin account on first login.

---

### Step 7: Create a dedicated MinIO user for Label Studio

Label Studio needs to read objects from the `exports` bucket (to load annotation tasks) and write annotation results back to `datasets`. Create a scoped user for this rather than using root credentials:

**On the object store server:**

```bash
cat > /tmp/labelstudio-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket",
        "s3:PutObject"
      ],
      "Resource": [
        "arn:aws:s3:::exports",
        "arn:aws:s3:::exports/*",
        "arn:aws:s3:::datasets",
        "arn:aws:s3:::datasets/*"
      ]
    }
  ]
}
EOF

mc admin policy create local labelstudio-policy \
  /tmp/labelstudio-policy.json

mc admin user add local labelstudio-user labelstudio-secret-key
mc admin policy attach local labelstudio-policy \
  --user labelstudio-user
```

---

### Step 8: Connect Label Studio to MinIO

Label Studio treats MinIO as an S3-compatible cloud storage backend. The connection is configured per project, not globally — each annotation project specifies which bucket it reads tasks from and which bucket it writes completed annotations to.

Before configuring the connection, set the CORS policy on the MinIO `exports` bucket so annotators' browsers can load task data directly:

**On the object store server:**

```bash
cat > /tmp/cors-exports.json << 'EOF'
{
  "CORSRules": [
    {
      "AllowedHeaders": ["*"],
      "AllowedMethods": ["GET", "HEAD"],
      "AllowedOrigins": ["http://annotation-server-ip:8080"],
      "ExposeHeaders": ["ETag"]
    }
  ]
}
EOF

mc anonymous set-json /tmp/cors-exports.json local/exports
```

Now configure the storage connection in Label Studio:

1. Log in to Label Studio at `http://annotation-server-ip:8080`
2. Create a new project (see Step 9 below for the labeling configuration)
3. Inside the project, go to **Settings → Cloud Storage → Add Source Storage**
4. Select **Amazon S3** (MinIO is S3-compatible)
5. Fill in the following fields:

| Field | Value |
|---|---|
| Storage title | Energy QA exports |
| Bucket name | `exports` |
| Bucket prefix | *(leave blank to see all objects)* |
| Region name | `us-east-1` *(required but ignored by MinIO)* |
| S3 endpoint | `http://object-store-ip:9000` |
| Access key ID | `labelstudio-user` |
| Secret access key | `labelstudio-secret-key` |
| Use pre-signed URLs | Yes |
| Pre-signed URL expiry | `3600` |

6. Click **Check Connection** — it should return green
7. Click **Save**, then **Sync Storage**

Label Studio will scan the `exports` bucket and import any JSON task files it finds as annotation tasks.

For the **target storage** (where completed annotations are written), add a second storage connection under **Settings → Cloud Storage → Add Target Storage**, pointing at the `datasets` bucket with the same credentials.

---

### Step 9: Create the energy QA annotation project

The labeling task for this series is reviewing model-generated answers to energy questions and marking them as correct, partially correct, or wrong — with a free-text correction field for wrong answers. This is the feedback data that drives the fine-tuning loop.

In Label Studio, after creating your project, go to **Settings → Labeling Interface** and paste the following configuration:

```xml
<View>
  <Style>
    .context-box {
      background: #f8f9fa;
      border: 1px solid #dee2e6;
      border-radius: 6px;
      padding: 14px 16px;
      margin-bottom: 16px;
      font-family: monospace;
      font-size: 13px;
      white-space: pre-wrap;
    }
    .question-box {
      background: #e8f4f8;
      border-left: 4px solid #2196f3;
      padding: 12px 16px;
      margin-bottom: 12px;
      border-radius: 0 6px 6px 0;
    }
    .answer-box {
      background: #fff;
      border: 1px solid #ccc;
      padding: 12px 16px;
      border-radius: 6px;
      margin-bottom: 16px;
    }
  </Style>

  <!-- Sensor context snapshot -->
  <View className="context-box">
    <Header value="Sensor context (at time of query)"/>
    <Text name="context" value="$context"/>
  </View>

  <!-- User question -->
  <View className="question-box">
    <Header value="Question"/>
    <Text name="question" value="$question"/>
  </View>

  <!-- Model answer -->
  <View className="answer-box">
    <Header value="Model answer"/>
    <Text name="answer" value="$answer"/>
  </View>

  <!-- Verdict -->
  <Header value="Is this answer correct?"/>
  <Choices name="verdict" toName="question"
    choice="single" showInLine="true">
    <Choice value="Correct" background="#28a745"/>
    <Choice value="Partially correct" background="#ffc107"/>
    <Choice value="Wrong" background="#dc3545"/>
  </Choices>

  <!-- Free-text correction — shown for wrong/partial answers -->
  <Header value="Correction (required if Wrong or Partially correct)"/>
  <TextArea name="correction" toName="question"
    placeholder="Write the correct answer here..."
    rows="4"
    editable="true"/>

  <!-- Optional: note on why the model was wrong -->
  <Header value="Failure reason (optional)"/>
  <Choices name="failure_reason" toName="question"
    choice="single" showInLine="true">
    <Choice value="Missing context"/>
    <Choice value="Misread sensor value"/>
    <Choice value="Wrong seasonal reasoning"/>
    <Choice value="Calculation error"/>
    <Choice value="Other"/>
  </Choices>

</View>
```

This interface gives annotators three things to do per task: rate the answer, write a correction if it is wrong, and categorise the failure reason. The failure reason field is optional but valuable — over time it reveals which categories of error the model makes most often, which informs how you weight synthetic data generation in Article 4.

---

### Step 10: Upload a test task and verify the pipeline

Before you have real model outputs to annotate, verify the pipeline end to end using a manually crafted test task.

**Create a test task file:**

```bash
cat > /tmp/test-task.json << 'EOF'
[
  {
    "context": "{\n  \"generated_at\": \"2024-03-15T14:32:00Z\",\n  \"solarpv\": {\n    \"current_power_w\": 3840,\n    \"yield_today_kwh\": 18.4,\n    \"yield_7d_avg_kwh\": 21.2\n  },\n  \"solarthermie\": {\n    \"buffer_top_c\": 67.1,\n    \"buffer_mid_c\": 54.8,\n    \"buffer_bottom_c\": 41.2\n  },\n  \"heizung\": {\n    \"pellets_today_kg\": 3.2,\n    \"pellets_7d_avg_kg\": 4.1\n  },\n  \"weather\": {\n    \"outdoor_temp_c\": 12.4\n  }\n}",
    "question": "The thermal buffer is at 67°C at 14:00. Should I run the dishwasher now or wait?",
    "answer": "Yes, run the dishwasher now. The buffer top is at 67°C which is well above the minimum usable temperature of ~55°C. PV is currently producing 3.84 kW so you are covering the dishwasher load from solar rather than drawing from the grid. The buffer will recharge this afternoon while irradiation is still strong. Waiting until evening would mean running on grid power."
  }
]
EOF
```

**Upload to MinIO:**

```bash
mc cp /tmp/test-task.json \
  local/exports/test/task-001.json
```

**Sync in Label Studio:**

In your project, go to **Settings → Cloud Storage**, find your source storage connection, and click **Sync Storage**. The test task should appear as a new annotation task in the project queue.

Open the task, confirm the context, question, and answer fields all render correctly, submit a verdict, and check that the completed annotation appears in the `datasets` bucket target storage after syncing.

If the task does not appear after syncing, check that the JSON file is valid, that the bucket CORS policy allows requests from the Label Studio hostname, and that the pre-signed URL expiry is long enough for your browser to load the task before it expires.

---

### Step 11: Firewall on the annotation server

```bash
# Label Studio — accessible from your LAN
sudo ufw allow from 192.168.0.0/16 to any port 8080 \
  comment 'Label Studio'

sudo ufw enable
sudo ufw status
```

---

### Verifying the complete layer

Before moving on to Article 4, confirm these four things:

**1. MinIO is serving objects correctly.** Run `mc ls local` and confirm all four buckets exist. Upload a small test file and retrieve it via the web console to confirm read/write works.

**2. The RAID array is healthy.** Run `sudo mdadm --detail /dev/md0` and confirm the state is `clean`. Note the device names of the individual drives — you will need them if you ever have to replace a failed drive.

**3. Label Studio can read from MinIO.** The test task uploaded in Step 10 should be visible and renderable in the annotation interface. All three text fields (context, question, answer) should display their content without errors.

**4. Completed annotations reach the datasets bucket.** Submit the test annotation, sync target storage, and confirm the annotation JSON appears in `local/datasets/`.

```bash
# Quick check — list everything in datasets after submitting the test annotation
mc ls local/datasets/
```

---

### A note on task format

The task format used in this series — a JSON object with `context`, `question`, and `answer` fields — maps directly to the training data format used in Article 4. The Label Studio export script produces a JSONL file where each line is a training example derived from one annotation task. Keeping the field names consistent between the annotation interface, the export script, and the training data format eliminates an entire class of preprocessing bugs.

If you add fields to your annotation interface (a confidence score from the inference server, a timestamp, a session ID), add them to the task JSON too. They will be carried through the export and can be used for filtering and weighting in the training script.

---

### What comes next

In Article 4 we generate synthetic training data from the InfluxDB history, run the annotation workflow at scale, export a curated dataset, sync it to a cloud GPU, and fine-tune Qwen2.5-14B-Instruct with QLoRA using Unsloth. By the end of Article 4 you will have a trained LoRA adapter sitting in the `adapters` bucket of your MinIO instance, ready for deployment.

The export, training, and evaluation scripts in Article 4 all assume the bucket layout and task format established in this article. If you have deviated from either, adjust the scripts accordingly before proceeding.

*The companion GitHub repository contains the Docker Compose file, the Label Studio labeling configuration XML, the MinIO policy JSON files, and the test task template referenced in this article.*

---

*Series: Building an AI Data Flywheel for Home Energy Management*
*Article 3 of 5 — Object Store and Annotation Infrastructure*
