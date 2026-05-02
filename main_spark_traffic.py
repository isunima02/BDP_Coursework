
#  Use Case 4 — Smart Traffic Management System
#  Main PySpark Implementation 
#  Objectives:
#    1. Identify peak traffic hours
#    2. Detect congested routes
#    3. Analyze vehicle flow patterns


from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# SECTION 1 — SPARK SESSION

spark = SparkSession.builder \
    .appName("SmartTrafficManagement") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "10") \
    .config("spark.executor.memory", "1g") \
    .config("spark.driver.memory", "1g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


print("  Use Case 4 — Smart Traffic Management System")

print(f"  Spark version : {spark.version}")


# SECTION 2 — LOAD DATASET FROM HDFS


# HDFS path 
INPUT_PATH  = "hdfs://localhost:9000/user/hadoop/traffic/cleaned/traffic_collision_cleaned.csv"
OUTPUT_PATH = "hdfs://localhost:9000/user/hadoop/traffic/output"

print(f"\n[1] Loading dataset: {INPUT_PATH}")

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(INPUT_PATH)

print(f"    Columns available: {df.columns}")
df.printSchema()



# SECTION 3 — VERIFY REQUIRED COLUMNS


print("\n[2] Verifying required columns...")

REQUIRED = [
    "DR_Number", "Date_Occurred", "Hour", "Area_Name",
    "Day_of_Week", "Time_of_Day", "Season", "Is_Weekend",
    "Year", "Month", "Collision_Severity", "Road_Type",
    "Is_Hotspot", "Congestion_Index", "Address",
    "Latitude", "Longitude"
]

missing = [c for c in REQUIRED if c not in df.columns]

if missing:
    print(f"    Missing columns: {missing}")
    print("    Run clean_traffic_collision.py first.")
    spark.stop()
    raise SystemExit(1)

print(f"    All {len(df.columns)} columns present")

df.cache()



# SECTION 4 — OBJECTIVE 1: IDENTIFY PEAK TRAFFIC HOURS

#  Patterns: Aggregation + Sorting + Filtering


print("  OBJECTIVE 1 — Peak Traffic Hours")
print("  Patterns: Aggregation + Sorting + Filtering")


# Collisions by Hour
print("\n[3] Collisions by Hour")

collisions_by_hour = df \
    .groupBy("Hour", "Time_of_Day") \
    .agg(F.count("DR_Number").alias("Total_Collisions")) \
    .orderBy(F.col("Total_Collisions").desc())

collisions_by_hour.show(10, truncate=False)

#Collisions by Time of Day 
print("[4] Collisions by Time of Day")

collisions_by_timeofday = df \
    .groupBy("Time_of_Day") \
    .agg(F.count("DR_Number").alias("Total_Collisions")) \
    .orderBy(F.col("Total_Collisions").desc())

collisions_by_timeofday.show(truncate=False)

# Collisions by Day of Week
print("[5] Collisions by Day of Week")

collisions_by_day = df \
    .groupBy("Day_of_Week") \
    .agg(F.count("DR_Number").alias("Total_Collisions")) \
    .orderBy(F.col("Total_Collisions").desc())

collisions_by_day.show(truncate=False)

# Weekend vs Weekday (Filtering) 
print("[6] Weekend vs Weekday — Filtering Pattern")

weekend_vs_weekday = df \
    .groupBy("Is_Weekend") \
    .agg(F.count("DR_Number").alias("Total_Collisions")) \
    .orderBy("Is_Weekend")

weekend_vs_weekday.show(truncate=False)

# Collisions by Season 
print("[7] Collisions by Season")

collisions_by_season = df \
    .groupBy("Season") \
    .agg(F.count("DR_Number").alias("Total_Collisions")) \
    .orderBy(F.col("Total_Collisions").desc())

collisions_by_season.show(truncate=False)



# SECTION 5 — OBJECTIVE 2: DETECT CONGESTED ROUTES

#  Patterns: Filtering + Aggregation + Sorting + Join


print("  OBJECTIVE 2 — Detect Congested Routes")
print("  Patterns: Filtering + Aggregation + Sorting + Join")


# Most dangerous districts
print("\n[8] Most Dangerous Districts — Aggregation + Sorting")

dangerous_areas = df \
    .groupBy("Area_Name") \
    .agg(F.count("DR_Number").alias("Total_Collisions")) \
    .orderBy(F.col("Total_Collisions").desc())

dangerous_areas.show(21, truncate=False)

# Road collisions only (Filtering) 
print("[9] Road-only Collisions — Filtering + Aggregation")

road_collisions = df \
    .filter(F.col("Road_Type") == "Road") \
    .groupBy("Area_Name", "Congestion_Index") \
    .agg(F.count("DR_Number").alias("Collision_Count")) \
    .orderBy(F.col("Collision_Count").desc())

road_collisions.show(20, truncate=False)

# Hotspot addresses (Filtering + Sorting) 
print("[10] Hotspot Addresses — Filtering + Aggregation + Sorting")

hotspot_addresses = df \
    .filter(F.col("Is_Hotspot") == True) \
    .groupBy("Address", "Area_Name") \
    .agg(F.count("DR_Number").alias("Collision_Count")) \
    .orderBy(F.col("Collision_Count").desc())

hotspot_addresses.show(15, truncate=False)

# Severity by district 
print("[11] Collision Severity by District — Aggregation")

severity_by_area = df \
    .groupBy("Area_Name", "Collision_Severity") \
    .agg(F.count("DR_Number").alias("Count")) \
    .orderBy("Area_Name", F.col("Count").desc())

severity_by_area.show(30, truncate=False)

# Peak hour per district — JOIN Pattern
print("[12] Peak Hour per District — Aggregation + Join")

# Step A: count per (area, hour)
area_hour_counts = df \
    .groupBy("Area_Name", "Hour") \
    .agg(F.count("DR_Number").alias("Hourly_Count"))

# Step B: max count per area
area_max = area_hour_counts \
    .groupBy("Area_Name") \
    .agg(F.max("Hourly_Count").alias("Max_Count"))

# Step C: Join to get the matching hour
peak_per_area = area_hour_counts \
    .join(area_max, on="Area_Name", how="inner") \
    .filter(F.col("Hourly_Count") == F.col("Max_Count")) \
    .select("Area_Name", "Hour", "Hourly_Count") \
    .withColumnRenamed("Hour", "Peak_Hour") \
    .withColumnRenamed("Hourly_Count", "Peak_Collisions") \
    .orderBy(F.col("Peak_Collisions").desc())

peak_per_area.show(21, truncate=False)


# SECTION 6 — OBJECTIVE 3: ANALYZE VEHICLE FLOW PATTERNS

#  Patterns: Aggregation + Sorting



print("  OBJECTIVE 3 — Analyze Vehicle Flow Patterns")
print("  Patterns: Aggregation + Sorting")


#  Year-over-year trend 
print("\n[13] Year-over-Year Trend — Aggregation + Sorting")

yearly_trend = df \
    .filter(F.col("Year").isNotNull()) \
    .groupBy("Year") \
    .agg(F.count("DR_Number").alias("Total_Collisions")) \
    .orderBy("Year")

yearly_trend.show(20, truncate=False)

# Monthly pattern 
print("[14] Monthly Flow Pattern — Aggregation")

monthly_trend = df \
    .groupBy("Month", "Season") \
    .agg(F.count("DR_Number").alias("Total_Collisions")) \
    .orderBy("Month")

monthly_trend.show(12, truncate=False)

# Hourly flow by district 
print("[15] Hourly Flow per District — Aggregation")

hourly_flow = df \
    .groupBy("Area_Name", "Hour") \
    .agg(F.count("DR_Number").alias("Collision_Count")) \
    .orderBy("Area_Name", "Hour")

hourly_flow.show(20, truncate=False)

#  Road type breakdown 
print("[16] Collisions by Road Type — Aggregation + Sorting")

by_road_type = df \
    .groupBy("Road_Type") \
    .agg(F.count("DR_Number").alias("Total_Collisions")) \
    .orderBy(F.col("Total_Collisions").desc())

by_road_type.show(truncate=False)



# SECTION 7 — SAVE RESULTS TO HDFS



print("  Saving Results")


def save_csv(dataframe, folder_name):
    path = f"{OUTPUT_PATH}/{folder_name}"
    dataframe.coalesce(1) \
        .write.mode("overwrite") \
        .option("header", "true") \
        .csv(path)
    print(f"  Saved -> {path}")

# Objective 1
save_csv(collisions_by_hour,      "obj1_peak_hours")
save_csv(collisions_by_timeofday, "obj1_time_of_day")
save_csv(collisions_by_day,       "obj1_day_of_week")
save_csv(weekend_vs_weekday,      "obj1_weekend_vs_weekday")
save_csv(collisions_by_season,    "obj1_seasonal")

# Objective 2
save_csv(dangerous_areas,         "obj2_dangerous_areas")
save_csv(road_collisions,         "obj2_road_congestion")
save_csv(hotspot_addresses,       "obj2_hotspot_addresses")
save_csv(severity_by_area,        "obj2_severity_by_area")
save_csv(peak_per_area,           "obj2_peak_hour_per_district")

# Objective 3
save_csv(yearly_trend,            "obj3_yearly_trend")
save_csv(monthly_trend,           "obj3_monthly_trend")
save_csv(hourly_flow,             "obj3_hourly_flow")
save_csv(by_road_type,            "obj3_road_type")

print("\n  All results saved.")

spark.stop()
print("Done.")
