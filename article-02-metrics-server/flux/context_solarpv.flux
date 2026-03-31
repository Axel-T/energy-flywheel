// context_solarpv.flux
// Returns current power, today's yield, and 7/30-day rolling averages.
// Adapt field names to match your InfluxDB measurement names.
// See docs/field-naming.md for guidance.

today_start = date.truncate(t: now(), unit: 1d)
now_time    = now()
ago_7d      = date.add(d: -7d,  to: now_time)
ago_30d     = date.add(d: -30d, to: now_time)

// Most recent AC power reading (within last 5 minutes)
current = from(bucket: "solarpv")
  |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "solarpv"
       and r._field == "ac_power_w")
  |> last()
  |> findRecord(fn: (key) => true, idx: 0)

// Cumulative yield since midnight
yield_today = from(bucket: "solarpv")
  |> range(start: today_start)
  |> filter(fn: (r) => r._measurement == "solarpv"
       and r._field == "yield_kwh")
  |> last()
  |> findRecord(fn: (key) => true, idx: 0)

// 7-day average of daily peak yield
yield_7d = from(bucket: "solarpv")
  |> range(start: ago_7d)
  |> filter(fn: (r) => r._measurement == "solarpv"
       and r._field == "yield_kwh")
  |> aggregateWindow(every: 1d, fn: max, createEmpty: false)
  |> mean()
  |> findRecord(fn: (key) => true, idx: 0)

// 30-day average of daily peak yield (seasonal baseline)
yield_30d = from(bucket: "solarpv")
  |> range(start: ago_30d)
  |> filter(fn: (r) => r._measurement == "solarpv"
       and r._field == "yield_kwh")
  |> aggregateWindow(every: 1d, fn: max, createEmpty: false)
  |> mean()
  |> findRecord(fn: (key) => true, idx: 0)

{
  current_power_w:   current._value,
  yield_today_kwh:   yield_today._value,
  yield_7d_avg_kwh:  yield_7d._value,
  yield_30d_avg_kwh: yield_30d._value,
}
