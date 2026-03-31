# Field naming guide

The Flux queries and context assembly scripts in this repository use generic,
semantically descriptive field names. Your InfluxDB installation almost
certainly uses different names — whatever your sensors or Home Assistant
integration writes.

This guide explains how to discover your field names and map them to the
names used in this series.

---

## Step 1: Discover your field names in InfluxDB

Open the InfluxDB Data Explorer at `http://metrics-server-ip:8086`.

Run this query to list all field keys in a bucket:

```flux
import "influxdata/influxdb/schema"

schema.fieldKeys(bucket: "solarpv")
```

This returns every field name that has been written to the `solarpv` bucket.
Repeat for `solarthermie` and `heizung`.

---

## Step 2: Map your field names to the series field names

The table below shows the field names used in the series Flux queries and
what each one represents. Find the equivalent in your InfluxDB and substitute.

### solarpv bucket

| Series field name    | Meaning                              | Common alternatives                          |
|----------------------|--------------------------------------|----------------------------------------------|
| `ac_power_w`         | Current AC output power in watts     | `power`, `pac`, `P_AC`, `grid_power`         |
| `yield_kwh`          | Cumulative energy yield in kWh       | `energy`, `e_day`, `yield_day`, `Eday`       |
| `outdoor_temp_c`     | Outdoor air temperature in °C        | `temperature`, `temp_out`, `outside_temp`    |

### solarthermie bucket

| Series field name    | Meaning                              | Common alternatives                          |
|----------------------|--------------------------------------|----------------------------------------------|
| `collector_temp_c`   | Solar collector temperature in °C    | `T_koll`, `collector`, `solar_temp`          |
| `buffer_temp_top_c`  | Thermal buffer top sensor in °C      | `T_oben`, `buffer_top`, `speicher_oben`      |
| `buffer_temp_mid_c`  | Thermal buffer mid sensor in °C      | `T_mitte`, `buffer_mid`, `speicher_mitte`    |
| `buffer_temp_bottom_c` | Thermal buffer bottom sensor in °C | `T_unten`, `buffer_bottom`, `speicher_unten` |

### heizung bucket

| Series field name      | Meaning                              | Common alternatives                        |
|------------------------|--------------------------------------|--------------------------------------------|
| `boiler_temp_c`        | Pellet boiler temperature in °C      | `kessel_temp`, `boiler`, `T_kessel`        |
| `flow_temp_c`          | Heating circuit flow temperature     | `vorlauf`, `flow`, `T_vorlauf`             |
| `return_temp_c`        | Heating circuit return temperature   | `ruecklauf`, `return`, `T_ruecklauf`       |
| `pellets_consumed_kg`  | Pellets consumed (kg, incremental)   | `pellet_kg`, `verbrauch`, `fuel_kg`        |
| `burner_runtime_min`   | Burner runtime in minutes            | `brenner_laufzeit`, `runtime`, `burn_min`  |

---

## Step 3: Apply the mapping

In each Flux query file (`flux/context_*.flux`), replace the series field
name with your actual field name. For example, if your PV inverter writes
`Pac` instead of `ac_power_w`:

```flux
// Before
|> filter(fn: (r) => r._field == "ac_power_w")

// After
|> filter(fn: (r) => r._field == "Pac")
```

In the context assembly Node-RED function (`nodered/context_assembly_flow.json`),
the output keys — `current_power_w`, `buffer_top_c`, etc. — should remain
unchanged. These are the keys the language model reads, and keeping them
consistent means you do not need to update prompts or training data formats
when your underlying field names differ.

The pattern is:
- **Flux query layer:** use your actual InfluxDB field names
- **Context assembly layer:** map to semantic, human-readable keys
- **LLM / training layer:** always use the semantic keys

---

## Step 4: Test the mapping

After updating the Flux queries, test the context API endpoint:

```bash
curl -s -X POST http://metrics-server-ip:1880/api/context \
  | python3 -m json.tool
```

Every field in the JSON output should have a non-null numeric value.
A `null` value means either the field name does not match, the bucket
has no recent data, or the time window in the query is too narrow.

To debug a null value, run the Flux query directly in the InfluxDB
Data Explorer with explicit time ranges to confirm data exists.
