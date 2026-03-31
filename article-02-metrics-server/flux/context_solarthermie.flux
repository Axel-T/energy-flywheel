// context_solarthermie.flux
// Returns current buffer temperatures and 7-day high/low.
// Adapt field names to match your InfluxDB measurement names.
// See docs/field-naming.md for guidance.

ago_7d = date.add(d: -7d, to: now())

collector = from(bucket: "solarthermie")
  |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "solarthermie"
       and r._field == "collector_temp_c")
  |> last()
  |> findRecord(fn: (key) => true, idx: 0)

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
  buffer_7d_low_c:   buffer_top_7d_low._value,
}
