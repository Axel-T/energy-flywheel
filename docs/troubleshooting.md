# Troubleshooting guide

Common errors encountered when following the series, with causes and fixes.
Organised by article. If your error is not listed here, open a GitHub issue
with the full error output and the article/step where it occurred.

---

## Article 2 — Metrics server

### InfluxDB container fails to start: "permission denied"

**Symptom:**
```
influxdb  | ts=... level=error msg="failed to open bolt file"
influxdb  | open /var/lib/influxdb2/influxd.bolt: permission denied
```

**Cause:** the `./influxdb-data` directory on the host was created by root
(e.g. by a previous `sudo docker compose up`), and the InfluxDB container
runs as a non-root user.

**Fix:**
```bash
sudo chown -R 1000:1000 ./influxdb-data
docker compose up -d
```

---

### Node-RED cannot connect to InfluxDB: "ECONNREFUSED"

**Symptom:** Node-RED debug panel shows connection refused when the
InfluxDB node tries to write.

**Cause:** Node-RED is trying to reach InfluxDB at `localhost:8086`, but
inside Docker each container has its own network namespace. Use the service
name instead.

**Fix:** in the Node-RED InfluxDB configuration node, set the host to
`influxdb` (the Docker Compose service name), not `localhost`.

---

### Context API returns null fields

**Symptom:**
```json
{
  "solarpv": {
    "current_power_w": null,
    "yield_today_kwh": null
  }
}
```

**Cause:** either the field name in the Flux query does not match what is
actually in InfluxDB, or there is no data within the query's time window.

**Fix:**
1. Open the InfluxDB Data Explorer and run:
   ```flux
   import "influxdata/influxdb/schema"
   schema.fieldKeys(bucket: "solarpv")
   ```
2. Compare the returned field names against those in `flux/context_solarpv.flux`.
3. Update the Flux queries to use your actual field names. See `docs/field-naming.md`.
4. If field names are correct, widen the time range in the query from `-5m` to `-30m`
   to confirm data exists before narrowing it back.

---

### MQTT messages not reaching InfluxDB

**Symptom:** sensors are publishing to MQTT (confirmed in MQTT Explorer or
`mosquitto_sub`) but no data appears in InfluxDB.

**Cause:** common causes are a misconfigured topic pattern in the MQTT-in
node, a JSON parse error in the function node, or the InfluxDB write node
using the wrong bucket name.

**Fix:**
1. Add a Debug node immediately after the MQTT-in node to inspect raw payloads.
2. Check that the topic pattern matches your sensor topics exactly —
   `homeassistant/sensor/pv/#` matches all subtopics, but
   `homeassistant/sensor/pv` only matches that exact topic.
3. In the function node, `JSON.parse(msg.payload)` will throw if the payload
   is already an object (Home Assistant sometimes sends objects, not strings).
   Use `const payload = typeof msg.payload === 'string' ? JSON.parse(msg.payload) : msg.payload;`
4. Confirm the bucket name in the InfluxDB-out node matches the bucket
   you created (`solarpv`, not `energy` or `default`).

---

## Article 3 — Object store and annotation

### Label Studio cannot reach MinIO: "connection refused" or "SSL error"

**Symptom:** the "Test Connection" button in Label Studio storage settings
returns an error.

**Cause:** either the MinIO endpoint URL is wrong, or MinIO is not
reachable from the Label Studio container.

**Fix:**
1. Confirm MinIO is running: `sudo systemctl status minio`
2. Confirm the port is open: `nc -zv object-store-ip 9000`
3. Use the full URL with protocol and port:
   `http://object-store-ip:9000` — not `object-store-ip:9000` alone.
4. The region field must be set to something (`us-east-1` works) — leaving
   it blank causes some S3 client errors even though MinIO ignores the value.
5. Check the UFW firewall on the MinIO server allows port 9000 from the
   annotation server's IP.

---

### Label Studio tasks not appearing after storage sync

**Symptom:** clicking "Sync Storage" completes without error but no tasks
appear in the project queue.

**Cause:** Label Studio expects task files to be valid JSON arrays at the
top level, and the bucket prefix may not match where the files were uploaded.

**Fix:**
1. Verify your task files are valid JSON arrays:
   ```bash
   mc cat local/exports/tasks/task_00000.json | python3 -m json.tool
   ```
   The output should start with `[` and end with `]`.
2. Check the **Bucket prefix** field in the storage settings matches the
   path where files were uploaded. If you uploaded to `exports/synthetic/v1/tasks/`,
   the prefix should be `synthetic/v1/tasks` (without the bucket name).
3. Label Studio only imports JSON files — confirm your files have a `.json`
   extension, not `.jsonl`.

---

### MinIO mdadm array shows "degraded" after reboot

**Symptom:** `sudo mdadm --detail /dev/md0` shows state `degraded` with
one drive listed as `removed`.

**Cause:** the array was not saved to `/etc/mdadm/mdadm.conf` before reboot,
so the system could not fully assemble it.

**Fix:**
```bash
# Reassemble manually
sudo mdadm --assemble --scan

# Save the config permanently
sudo mdadm --detail --scan | sudo tee -a /etc/mdadm/mdadm.conf
sudo update-initramfs -u

# Verify
sudo mdadm --detail /dev/md0   # state should be "clean" or "active"
```

---

## Article 4 — Dataset generation and fine-tuning

### llama.cpp produces empty output

**Symptom:** `call_llama()` returns an empty string or `None`.

**Cause:** the prompt format does not match the model's expected chat
template. Different models use different delimiters
(`<|system|>`, `[INST]`, `<s>`, etc.).

**Fix:**
1. Test the model manually first:
   ```bash
   ./build/bin/llama-cli -m ~/llama-models/model.gguf \
     -p "What is 2+2?" -n 64 --temp 0
   ```
2. If the manual test works but the script does not, the prompt template
   in `generate_qa_pairs.py` may not match your model.
   For Llama 3.x models, use `<|begin_of_text|><|start_header_id|>system<|end_header_id|>`.
   For Mistral/Mixtral, use `[INST]...[/INST]`.
   Check the model card on Hugging Face for the correct template.
3. Increase `--ctx-size` if the combined prompt + context exceeds 4096 tokens.

---

### Unsloth installation fails: CUDA version mismatch

**Symptom:**
```
ERROR: Could not find a version of torch that satisfies the requirement
```
or Unsloth errors about CUDA version.

**Cause:** the PyTorch CUDA wheel must match the CUDA version on the GPU instance.

**Fix:**
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Install matching PyTorch wheel — adjust cu118/cu121 to your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

---

### Training loss not decreasing / stuck at ~2.3

**Symptom:** training loss stays near 2.3 across all steps and does not drop.

**Cause:** the dataset format is incorrect — the model is seeing random
text rather than properly formatted instruction examples. A loss of ~2.3
corresponds to uniform random prediction over a vocabulary of ~10,000 tokens.

**Fix:**
1. Inspect the first formatted example:
   ```python
   from datasets import load_dataset
   ds = load_dataset("json", data_files="energy_qa_v1.jsonl", split="train")
   print(ds[0])
   ```
2. Confirm the `instruction`, `input`, and `output` fields are all populated.
3. Check `format_prompts()` is producing text that starts with the Alpaca
   prompt prefix and ends with the EOS token.
4. Print a sample formatted text before training starts to visually inspect it.

---

### Out of memory (OOM) during training

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**Cause:** the effective batch size or sequence length is too large for
the available VRAM.

**Fix (in order of impact):**
1. Reduce `per_device_train_batch_size` from 2 to 1.
2. Reduce `MAX_SEQ_LENGTH` from 2048 to 1024 — check your longest example
   first: `max(len(tokenizer.encode(ex['text'])) for ex in dataset)`.
3. Increase `gradient_accumulation_steps` to compensate for the smaller
   batch size (keeps effective batch size the same).
4. Set `load_in_4bit=True` if not already set.
5. Add `max_grad_norm=0.3` to `TrainingArguments`.

---

## Article 5 — Deployment and feedback loop

### Ollama model loads on CPU instead of GPU

**Symptom:** `ollama run energy-assistant-v1 "test"` is very slow (minutes
instead of seconds), and `ollama ps` shows `100% CPU`.

**Cause:** CUDA libraries are not visible to Ollama, or the GPU has
insufficient VRAM for the model.

**Fix:**
1. Confirm NVIDIA drivers are installed: `nvidia-smi`
2. Confirm CUDA is installed: `nvcc --version`
3. Check Ollama logs: `journalctl -u ollama -f`
4. If the model is 8.4GB and your GPU has exactly 8GB VRAM, Ollama may
   fall back to CPU because it cannot fit the model plus KV cache.
   Try a smaller quantisation: `q3_k_m` (~6.8GB) or set
   `OLLAMA_NUM_GPU_LAYERS` to offload only some layers.
5. On Linux, Ollama requires the NVIDIA Container Toolkit if running in Docker,
   but when running as a systemd service it uses the host GPU directly.
   Confirm with: `sudo systemctl status ollama` and look for CUDA in the startup log.

---

### OpenWebUI context pipeline not injecting sensor data

**Symptom:** model responses do not reference current sensor values —
the context block is not appearing in the prompt.

**Cause:** the pipeline is not enabled, the context API URL is wrong, or
the Node-RED context API is returning an error.

**Fix:**
1. In OpenWebUI Admin Panel → Pipelines, confirm the pipeline status is "Enabled".
2. Test the context API directly:
   ```bash
   curl -s -X POST http://metrics-server-ip:1880/api/context \
     | python3 -m json.tool
   ```
3. Check the pipeline logs in OpenWebUI (Admin Panel → Logs) for any
   `[energy-context]` warning messages.
4. Confirm `CONTEXT_API_URL` in the pipeline Valves matches the actual URL —
   `http://` not `https://`, correct IP, port 1880.
5. Check UFW on the metrics server allows port 1880 from the inference server IP.

---

### Feedback webhook not receiving events from OpenWebUI

**Symptom:** thumbs-down ratings in OpenWebUI do not appear in
`mc ls local/exports/feedback/`.

**Cause:** the webhook URL in OpenWebUI settings is incorrect, or
Node-RED is not listening on the expected endpoint.

**Fix:**
1. In OpenWebUI Admin Panel → Settings → General, confirm the Webhook URL
   is `http://metrics-server-ip:1880/api/feedback` with the correct IP.
2. In Node-RED, confirm the `http in` node shows method POST and URL `/api/feedback`,
   and that the flow is deployed (not just saved).
3. Test the webhook manually:
   ```bash
   curl -s -X POST http://metrics-server-ip:1880/api/feedback \
     -H "Content-Type: application/json" \
     -d '{"rating": -1, "message": "test question", "response": "test answer"}'
   ```
   This should return `{"status": "ok", "action": "queued"}`.
4. Check Node-RED debug output for any errors in the `parse-feedback` or
   `write-task-file` nodes.
