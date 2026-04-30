# =============================================================================
#  Use Case 4 — Smart Traffic Management System
#  MapReduce Design Patterns — Full RDD Implementation
#
#  This file demonstrates all 4 required MapReduce patterns using RDDs:
#    Pattern 1 — Filtering
#    Pattern 2 — Aggregation
#    Pattern 3 — Sorting
#    Pattern 4 — Join
#
#  Each pattern maps directly to a Use Case 4 objective.
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

#  Spark Session 
spark = SparkSession.builder \
    .appName("TrafficMapReducePatterns") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 65)
print("  MapReduce Design Patterns — RDD Implementation")
print("  Use Case 4: Smart Traffic Management System")
print("=" * 65)



# LOAD DATASET

INPUT_PATH  = "hdfs://localhost:9000/user/hadoop/traffic/cleaned/traffic_collision_cleaned.csv"
OUTPUT_PATH = "hdfs://localhost:9000/user/hadoop/traffic/output/mapreduce"

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(INPUT_PATH)

# Convert to RDD — all patterns below use raw RDD operations
rdd = df.rdd

total = rdd.count()
print(f"\n  Dataset loaded: {total:,} rows")
print(f"  Partitions    : {rdd.getNumPartitions()}")
print(f"  RDD type      : {type(rdd)}")


# 
# PATTERN 1 — FILTERING
#
#  Purpose : Narrow dataset to only the records we care about
#  Use Case: Identify peak traffic hours — isolate rush hour collisions
#  RDD op  : .filter(lambda row: condition)
# 
print("\n" + "=" * 65)
print("  PATTERN 1 — FILTERING")
print("  Goal: Isolate rush hour and high-risk road collisions")
print("=" * 65)

# ── Filter: Morning Rush (06:00–09:00) ────────────────────────────────────
morning_rush_rdd = rdd.filter(
    lambda row: row["Hour"] is not None and 6 <= int(row["Hour"]) <= 9
)

morning_count = morning_rush_rdd.count()
print(f"\n  Morning Rush (06:00–09:00): {morning_count:,} collisions")

# ── Filter: Evening Rush (17:00–19:00) ────────────────────────────────────
evening_rush_rdd = rdd.filter(
    lambda row: row["Hour"] is not None and 17 <= int(row["Hour"]) <= 19
)

evening_count = evening_rush_rdd.count()
print(f"  Evening Rush (17:00–19:00): {evening_count:,} collisions")

# ── Filter: Road collisions only (exclude parking lots, alleys) ───────────
road_only_rdd = rdd.filter(
    lambda row: row["Road_Type"] in ["Road"]
)

road_count = road_only_rdd.count()
print(f"   Road collisions only      : {road_count:,} collisions")

# ── Filter: Hotspot locations only ────────────────────────────────────────
hotspot_rdd = rdd.filter(
    lambda row: str(row["Is_Hotspot"]).lower() == "true"
)

hotspot_count = hotspot_rdd.count()
print(f"  [Filter 1d] Hotspot locations only     : {hotspot_count:,} collisions")

# ──  Filter: Injury-involved collisions (exclude property damage) ──────────
injury_rdd = rdd.filter(
    lambda row: row["Collision_Severity"] == "Injury Involved"
)

injury_count = injury_rdd.count()
print(f"   Injury-involved only       : {injury_count:,} collisions")

# ── Convert results to DataFrame and show sample ─────────────────────────
print("\n  Sample evening rush collision records:")
evening_sample = spark.createDataFrame(
    evening_rush_rdd.map(lambda r: (
        str(r["DR_Number"]),
        str(r["Area_Name"]),
        int(r["Hour"]) if r["Hour"] is not None else 0,
        str(r["Address"]),
        str(r["Collision_Severity"])
    )),
    schema=["DR_Number", "Area_Name", "Hour", "Address", "Severity"]
)
evening_sample.show(8, truncate=False)

print("  Pattern 1 (Filtering) complete")


# =============================================================================
# PATTERN 2 — AGGREGATION
#
#  Purpose : Summarise data by grouping and counting
#  Use Case: Count collisions per hour → find peak hours
#            Count collisions per area → find most congested zones
#  RDD ops : .map() → .reduceByKey() → result
# =============================================================================

print("\n" + "=" * 65)
print("  PATTERN 2 — AGGREGATION")
print("  Goal: Count collisions per hour and per area")
print("=" * 65)

# ── Collision count per hour (Map → ReduceByKey) ──────────────────────────
#
#  MAP    : each row → (hour, 1)
#  REDUCE : sum all 1s per hour key
#
print("\n   Collisions per Hour")
print("  MAP: row → (hour, 1)  |  REDUCE: sum by hour key")

hourly_rdd = rdd \
    .filter(lambda row: row["Hour"] is not None) \
    .map(lambda row: (int(row["Hour"]), 1)) \
    .reduceByKey(lambda count_a, count_b: count_a + count_b)

hourly_result = hourly_rdd.sortBy(lambda x: x[0])   # sort by hour

hourly_df = spark.createDataFrame(
    hourly_result.map(lambda x: (x[0], x[1])),
    schema=["Hour", "Collision_Count"]
)

print("\n  Collisions by Hour (0–23):")
hourly_df.show(24, truncate=False)

# ──  Collision count per area (Map → ReduceByKey) ──────────────────────────
#
#  MAP    : each row → (area_name, 1)
#  REDUCE : sum all 1s per area key
#
print("   Collisions per District")
print("  MAP: row → (area, 1)  |  REDUCE: sum by area key")

area_rdd = rdd \
    .filter(lambda row: row["Area_Name"] is not None) \
    .map(lambda row: (str(row["Area_Name"]), 1)) \
    .reduceByKey(lambda a, b: a + b)

area_df = spark.createDataFrame(
    area_rdd.map(lambda x: (x[0], x[1])),
    schema=["Area_Name", "Collision_Count"]
)

print("\n  Collisions by District:")
area_df.orderBy(F.col("Collision_Count").desc()).show(21, truncate=False)

# ──  Collision count per (Area, Hour) pair ─────────────────────────────────
#
#  MAP    : each row → ((area, hour), 1)
#  REDUCE : sum by composite key
#
print("  Collisions per (District, Hour) pair")
print("  MAP: row → ((area, hour), 1)  |  REDUCE: sum by (area, hour) key")

area_hour_rdd = rdd \
    .filter(lambda row: row["Area_Name"] is not None
                     and row["Hour"] is not None) \
    .map(lambda row: ((str(row["Area_Name"]), int(row["Hour"])), 1)) \
    .reduceByKey(lambda a, b: a + b)

area_hour_df = spark.createDataFrame(
    area_hour_rdd.map(lambda kv: (kv[0][0], kv[0][1], kv[1])),
    schema=["Area_Name", "Hour", "Collision_Count"]
)

print("\n  Top 20 (District, Hour) collision counts:")
area_hour_df.orderBy(F.col("Collision_Count").desc()).show(20, truncate=False)

# ──  Year-over-year trend (Map → ReduceByKey) ──────────────────────────────
print("   Year-over-year collision trend")

yearly_rdd = rdd \
    .filter(lambda row: row["Year"] is not None) \
    .map(lambda row: (int(row["Year"]), 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .sortBy(lambda x: x[0])

yearly_df = spark.createDataFrame(
    yearly_rdd.map(lambda x: (x[0], x[1])),
    schema=["Year", "Collision_Count"]
)

print("\n  Year-over-Year Trend (note COVID drop in 2020):")
yearly_df.show(20, truncate=False)

print("  Pattern 2 (Aggregation) complete")


# =============================================================================
# PATTERN 3 — SORTING
#
#  Purpose : Rank results to find top/bottom N items
#  Use Case: Find the top 10 peak hours, top 5 most dangerous areas
#  RDD ops : .sortBy(key, ascending=False)
# =============================================================================

print("\n" + "=" * 65)
print("  PATTERN 3 — SORTING")
print("  Goal: Rank peak hours and most dangerous routes")
print("=" * 65)

# ──  Top 10 peak hours by collision count ──────────────────────────────────
print("\n  [Sort 3a] Top 10 peak hours — descending by collision count")

top_hours_rdd = hourly_rdd \
    .sortBy(lambda x: x[1], ascending=False)

top_10_hours = top_hours_rdd.take(10)

print("\n  Rank  Hour  Collisions")
print("  ----  ----  ----------")
for rank, (hour, count) in enumerate(top_10_hours, 1):
    bar = "█" * (count // 3000)
    print(f"  #{rank:<4} {hour:02d}:00  {count:,}  {bar}")

# ── Top 5 most dangerous areas ───────────────────────────────────────────
print("\n   Top 5 most dangerous districts")

top_areas_rdd = area_rdd \
    .sortBy(lambda x: x[1], ascending=False)

top_5_areas = top_areas_rdd.take(5)

print("\n  Rank  District            Collisions")
print("  ----  ------------------  ----------")
for rank, (area, count) in enumerate(top_5_areas, 1):
    print(f"  #{rank:<4} {area:<20} {count:,}")

# ── Sort by year ascending (time series order) ───────────────────────────
print("\n  [Sort 3c] Yearly trend — ascending time series")

sorted_years = yearly_rdd.sortBy(lambda x: x[0], ascending=True).collect()

print("\n  Year  Collisions  Trend")
print("  ----  ----------  -----")
prev = None
for year, count in sorted_years:
    if prev is None:
        trend = " ─"
    elif count > prev:
        trend = " ▲"
    elif count < prev:
        trend = " ▼"
    else:
        trend = " ─"
    print(f"  {year}  {count:>10,}  {trend}")
    prev = count

print("\n  Pattern 3 (Sorting) complete")


# =============================================================================
# PATTERN 4 — JOIN
#
#  Purpose : Combine two datasets by a shared key
#  Use Case: Merge hourly collision counts with area-level stats
#            so each (area, hour) row also shows the area's total
#  RDD ops : RDD_A.join(RDD_B) on matching key
# =============================================================================

print("\n" + "=" * 65)
print("  PATTERN 4 — JOIN")
print("  Goal: Enrich (Area, Hour) data with area-level totals")
print("=" * 65)

# ── Prepare RDD : area-level total collisions ────────────────────────────────
#    Key = Area_Name
#    Value = total_collisions for that area
rdd_A = area_rdd      # (area_name, total_collisions)

# ── Prepare RDD : peak hour per area ────────────────────────────────────────
#    Key = Area_Name
#    Value = (peak_hour, peak_hour_count) — the hour with max collisions
rdd_B = area_hour_rdd \
    .map(lambda kv: (kv[0][0], (kv[0][1], kv[1]))) \
    .reduceByKey(lambda a, b: a if a[1] >= b[1] else b)
    # keep whichever (hour, count) pair has the higher count

print("\n  RDD  (area totals) — first 3:")
for item in rdd_A.take(3):
    print(f"    {item}")

print("\n  RDD  (peak hour per area) — first 3:")
for item in rdd_B.take(3):
    print(f"    {item}")

# ── JOIN: rdd_A.join(rdd_B) → (area, (total, (peak_hour, peak_count))) ────────
print("\n  Joining area totals with peak hour data")

joined_rdd = rdd_A.join(rdd_B)
# Result structure: (area_name, (total_collisions, (peak_hour, peak_count)))

# Flatten for readability
flat_joined_rdd = joined_rdd.map(
    lambda kv: (
        kv[0],                  # Area_Name
        kv[1][0],               # Total_Collisions
        kv[1][1][0],            # Peak_Hour
        kv[1][1][1],            # Peak_Hour_Collisions
        round(kv[1][1][1] / kv[1][0] * 100, 1)  # % collisions in peak hour
    )
).sortBy(lambda x: -x[1])   # sort by total descending

joined_df = spark.createDataFrame(
    flat_joined_rdd,
    schema=["Area_Name", "Total_Collisions",
            "Peak_Hour", "Peak_Hour_Collisions", "Peak_Pct"]
)

print("\n  Joined Result — Area totals enriched with peak hour:")
joined_df.show(21, truncate=False)

# ── JOIN: hotspot addresses joined with area peak hour ─────────────────────────
print("  [Join 4b] Hotspot addresses joined with district peak hour")

# RDD C: hotspot address collision count — (area, (address, count))
rdd_C = rdd \
    .filter(lambda row: str(row["Is_Hotspot"]).lower() == "true"
                     and row["Area_Name"] is not None
                     and row["Address"] is not None) \
    .map(lambda row: ((str(row["Area_Name"]), str(row["Address"])), 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))  # (area, (address, count))

# RDD D: area peak hour — (area, peak_hour)
rdd_D = rdd_B.map(lambda kv: (kv[0], kv[1][0]))   # (area, peak_hour)

# Join: for each hotspot address, attach the area's peak hour
hotspot_joined_rdd = rdd_C.join(rdd_D)
# Result: (area, ((address, count), peak_hour))

flat_hotspot = hotspot_joined_rdd.map(
    lambda kv: (
        kv[0],              # Area_Name
        kv[1][0][0],        # Address
        kv[1][0][1],        # Collision_Count
        kv[1][1]            # Area_Peak_Hour
    )
).sortBy(lambda x: -x[2])   # sort by collision count descending

hotspot_joined_df = spark.createDataFrame(
    flat_hotspot,
    schema=["Area_Name", "Address", "Collision_Count", "Area_Peak_Hour"]
)

print("\n  Hotspot addresses with area peak hour attached:")
hotspot_joined_df.show(15, truncate=False)

print("  Pattern 4 (Join) complete")


# =============================================================================
# SAVE ALL MAPREDUCE OUTPUTS
# =============================================================================

print("\n" + "=" * 65)
print("  Saving MapReduce Pattern Outputs")
print("=" * 65)

def save(dataframe, name):
    path = f"{OUTPUT_PATH}/{name}"
    dataframe.coalesce(1).write.mode("overwrite") \
        .option("header", "true").csv(path)
    print(f"Saved → {path}/")

save(evening_sample,     "pattern1_filter_evening_rush")
save(hourly_df,          "pattern2_agg_hourly_count")
save(area_df,            "pattern2_agg_area_count")
save(area_hour_df,       "pattern2_agg_area_hour_count")
save(yearly_df,          "pattern2_agg_yearly_trend")
save(joined_df,          "pattern4_join_area_peak_hour")
save(hotspot_joined_df,  "pattern4_join_hotspot_area")

# Save top 10 peak hours as CSV manually from RDD
top_hours_df = spark.createDataFrame(
    spark.sparkContext.parallelize(
        [(int(h), int(c)) for h, c in top_10_hours]
    ),
    schema=["Hour", "Collision_Count"]
)
save(top_hours_df, "pattern3_sort_top_peak_hours")

top_areas_df = spark.createDataFrame(
    spark.sparkContext.parallelize(
        [(str(a), int(c)) for a, c in top_5_areas]
    ),
    schema=["Area_Name", "Collision_Count"]
)
save(top_areas_df, "pattern3_sort_top_areas")


# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 65)
print("  MAPREDUCE DESIGN PATTERNS — SUMMARY")
print("=" * 65)
print("""
  ┌─────────────┬────────────────────────────┬────────────────────────────┐
  │ Pattern     │ RDD Operations Used        │ Traffic Analysis Purpose   │
  ├─────────────┼────────────────────────────┼────────────────────────────┤
  │ Filtering   │ .filter(lambda row: cond)  │ Isolate rush hour records  │
  │             │                            │ Focus on road collisions   │
  │             │                            │ Extract hotspot rows       │
  ├─────────────┼────────────────────────────┼────────────────────────────┤
  │ Aggregation │ .map(row → (key, 1))       │ Count per hour             │
  │             │ .reduceByKey(a + b)        │ Count per district         │
  │             │                            │ Count per (area, hour)     │
  │             │                            │ Year-over-year trend       │
  ├─────────────┼────────────────────────────┼────────────────────────────┤
  │ Sorting     │ .sortBy(key, asc=False)    │ Rank top 10 peak hours     │
  │             │                            │ Rank most dangerous areas  │
  │             │                            │ Order yearly time series   │
  ├─────────────┼────────────────────────────┼────────────────────────────┤
  │ Join        │ rdd_A.join(rdd_B)          │ Merge area totals with     │
  │             │ on matching key            │ peak hour data             │
  │             │                            │ Attach district context    │
  │             │                            │ to hotspot addresses       │
  └─────────────┴────────────────────────────┴────────────────────────────┘
""")

spark.stop()
print("All MapReduce patterns complete. Spark session stopped.")
