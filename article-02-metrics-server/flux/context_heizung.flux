// context_heizung.flux
// Returns boiler temperature and pellet consumption with rolling averages.
// Adapt field names to match your InfluxDB measurement names.
// See docs/field-naming.md for guidance.

today_start = date.truncate(t: now(), unit: 1d)
ago_7d      = date.add(d: -7d,  to: now())
ago_30d     = date.add(d: -30d, to: now())

boiler = from(bucket: "heizung")
  |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "heizung"
       and r._field == "boiler_temp_c")
  |> last()
  |> findRecord(fn: (key) => true, idx: 0)

pellets_today = from(bucket: "heizung")
  |> range(start: today_start)
  |> filter(fn: (r) => r._measurement == "heizung"
       and r._field == "pellets_consumed_kg")
  |> sum()
  |> findRecord(fn: (key) => true, idx: 0)

pellets_7d = from(bucket: "heizung")
  |> range(start: ago_7d)
  |> filter(fn: (r) => r._measurement == "heizung"
       and r._field == "pellets_consumed_kg")
  |> aggregateWindow(every: 1d, fn: sum, createEmpty: false)
  |> mean()
  |> findRecord(fn: (key) => true, idx: 0)

pellets_30d = from(bucket: "heizung")
  |> range(start: ago_30d)
  |> filter(fn: (r) => r._measurement == "heizung"
       and r._field == "pellets_consumed_kg")
  |> aggregateWindow(every: 1d, fn: sum, createEmpty: false)
  |> mean()
  |> findRecord(fn: (key) => true, idx: 0)

{
  boiler_temp_c:         boiler._value,
  pellets_today_kg:      pellets_today._value,
  pellets_7d_avg_kg:     pellets_7d._value,
  pellets_30d_avg_kg:    pellets_30d._value,
}
