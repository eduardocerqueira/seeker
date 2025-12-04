#date: 2025-12-04T16:59:39Z
#url: https://api.github.com/gists/963c70b8d78e2b3b593420f8046cc01f
#owner: https://api.github.com/users/davidwinalda

# Create Cleaned Inventory Table (inventory_fta_cleaned)

from google.cloud import bigquery

print("\n" + "="*100)
print("🧹 CREATING CLEANED INVENTORY TABLE")
print("="*100)
print("📌 Source: sample_recommendations.inventory_fta")
print("📌 Target: sample_recommendations.inventory_fta_cleaned")
print("="*100)

# ==================================================================================
# CREATE CLEANED TABLE WITH PARSED + EXISTING DATA
# ==================================================================================
print("\n" + "="*100)
print("1️⃣ CREATING CLEANED TABLE")
print("="*100)

create_cleaned_query = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.inventory_fta_cleaned` AS

WITH 
-- Part 1: Properly formatted existing data
existing_data AS (
  SELECT
    _file,
    _line,
    _modified,
    _fivetran_synced,
    
    -- Clean and standardize fields
    TRIM(channel_name) as channel_name,
    CAST(TRIM(tx_code) AS STRING) as tx_code,
    TRIM(tx_name) as tx_name,
    CAST(id_channel AS INT64) as id_channel,
    
    -- Program details
    TRIM(banner_name) as program_name,
    TRIM(film_poc) as film_code,
    TRIM(row_id_poc) as row_id,
    
    -- Timing
    TRIM(cbs_date) as broadcast_date,
    TRIM(st_time) as start_time,
    TRIM(ed_time) as end_time,
    TRIM(slot_ttime) as slot_time,
    CAST(slot_dur AS INT64) as slot_duration_sec,
    TRIM(cb_dur_avail) as cb_duration_avail,
    
    -- Pricing and category
    CASE 
      WHEN slot_amount IS NULL OR TRIM(slot_amount) = '' THEN NULL
      ELSE CAST(REPLACE(TRIM(slot_amount), ',', '') AS INT64)
    END as slot_price,
    TRIM(catslot_name) as slot_category,
    
    'existing_format' as data_source
    
  FROM `{PROJECT_ID}.{DATASET_ID}.inventory_fta`
  WHERE channel_name IS NOT NULL
),

-- Part 2: Hidden data (parsed from weird column)
hidden_data_parsed AS (
  SELECT
    _file,
    _line,
    _modified,
    _fivetran_synced,
    
    -- Parse from the long column name
    TRIM(SPLIT(channel_name_id_channel_tx_code_tx_name_cbs_date_slot_ttime_slot_dur_row_id_poc_film_poc_banner_name_cb_dur_avail_slot_amount_st_time_ed_time_catslot_name, ',')[SAFE_OFFSET(0)]) as channel_name,
    TRIM(SPLIT(channel_name_id_channel_tx_code_tx_name_cbs_date_slot_ttime_slot_dur_row_id_poc_film_poc_banner_name_cb_dur_avail_slot_amount_st_time_ed_time_catslot_name, ',')[SAFE_OFFSET(2)]) as tx_code,
    TRIM(SPLIT(channel_name_id_channel_tx_code_tx_name_cbs_date_slot_ttime_slot_dur_row_id_poc_film_poc_banner_name_cb_dur_avail_slot_amount_st_time_ed_time_catslot_name, ',')[SAFE_OFFSET(3)]) as tx_name,
    SAFE_CAST(TRIM(SPLIT(channel_name_id_channel_tx_code_tx_name_cbs_date_slot_ttime_slot_dur_row_id_poc_film_poc_banner_name_cb_dur_avail_slot_amount_st_time_ed_time_catslot_name, ',')[SAFE_OFFSET(1)]) AS INT64) as id_channel,
    
    -- Program details
    TRIM(SPLIT(channel_name_id_channel_tx_code_tx_name_cbs_date_slot_ttime_slot_dur_row_id_poc_film_poc_banner_name_cb_dur_avail_slot_amount_st_time_ed_time_catslot_name, ',')[SAFE_OFFSET(9)]) as program_name,
    TRIM(SPLIT(channel_name_id_channel_tx_code_tx_name_cbs_date_slot_ttime_slot_dur_row_id_poc_film_poc_banner_name_cb_dur_avail_slot_amount_st_time_ed_time_catslot_name, ',')[SAFE_OFFSET(8)]) as film_code,
    TRIM(SPLIT(channel_name_id_channel_tx_code_tx_name_cbs_date_slot_ttime_slot_dur_row_id_poc_film_poc_banner_name_cb_dur_avail_slot_amount_st_time_ed_time_catslot_name, ',')[SAFE_OFFSET(7)]) as row_id,
    
    -- Timing
    TRIM(SPLIT(channel_name_id_channel_tx_code_tx_name_cbs_date_slot_ttime_slot_dur_row_id_poc_film_poc_banner_name_cb_dur_avail_slot_amount_st_time_ed_time_catslot_name, ',')[SAFE_OFFSET(4)]) as broadcast_date,
    TRIM(SPLIT(channel_name_id_channel_tx_code_tx_name_cbs_date_slot_ttime_slot_dur_row_id_poc_film_poc_banner_name_cb_dur_avail_slot_amount_st_time_ed_time_catslot_name, ',')[SAFE_OFFSET(12)]) as start_time,
    TRIM(SPLIT(channel_name_id_channel_tx_code_tx_name_cbs_date_slot_ttime_slot_dur_row_id_poc_film_poc_banner_name_cb_dur_avail_slot_amount_st_time_ed_time_catslot_name, ',')[SAFE_OFFSET(13)]) as end_time,
    TRIM(SPLIT(channel_name_id_channel_tx_code_tx_name_cbs_date_slot_ttime_slot_dur_row_id_poc_film_poc_banner_name_cb_dur_avail_slot_amount_st_time_ed_time_catslot_name, ',')[SAFE_OFFSET(5)]) as slot_time,
    SAFE_CAST(TRIM(SPLIT(channel_name_id_channel_tx_code_tx_name_cbs_date_slot_ttime_slot_dur_row_id_poc_film_poc_banner_name_cb_dur_avail_slot_amount_st_time_ed_time_catslot_name, ',')[SAFE_OFFSET(6)]) AS INT64) as slot_duration_sec,
    TRIM(SPLIT(channel_name_id_channel_tx_code_tx_name_cbs_date_slot_ttime_slot_dur_row_id_poc_film_poc_banner_name_cb_dur_avail_slot_amount_st_time_ed_time_catslot_name, ',')[SAFE_OFFSET(10)]) as cb_duration_avail,
    
    -- Pricing and category
    SAFE_CAST(TRIM(SPLIT(channel_name_id_channel_tx_code_tx_name_cbs_date_slot_ttime_slot_dur_row_id_poc_film_poc_banner_name_cb_dur_avail_slot_amount_st_time_ed_time_catslot_name, ',')[SAFE_OFFSET(11)]) AS INT64) as slot_price,
    TRIM(SPLIT(channel_name_id_channel_tx_code_tx_name_cbs_date_slot_ttime_slot_dur_row_id_poc_film_poc_banner_name_cb_dur_avail_slot_amount_st_time_ed_time_catslot_name, ',')[SAFE_OFFSET(14)]) as slot_category,
    
    'hidden_recovered' as data_source
    
  FROM `{PROJECT_ID}.{DATASET_ID}.inventory_fta`
  WHERE channel_name IS NULL 
    AND channel_name_id_channel_tx_code_tx_name_cbs_date_slot_ttime_slot_dur_row_id_poc_film_poc_banner_name_cb_dur_avail_slot_amount_st_time_ed_time_catslot_name IS NOT NULL
),

-- Part 3: Combine all data
combined_data AS (
  SELECT * FROM existing_data
  UNION ALL
  SELECT * FROM hidden_data_parsed
)

-- Final output with additional calculated fields
SELECT
  *,
  
  -- Calculate daypart from start_time
  CASE
    WHEN SAFE_CAST(SPLIT(start_time, ':')[SAFE_OFFSET(0)] AS INT64) BETWEEN 6 AND 11 THEN 'Morning'
    WHEN SAFE_CAST(SPLIT(start_time, ':')[SAFE_OFFSET(0)] AS INT64) BETWEEN 12 AND 17 THEN 'Afternoon'
    WHEN SAFE_CAST(SPLIT(start_time, ':')[SAFE_OFFSET(0)] AS INT64) BETWEEN 18 AND 22 THEN 'Prime Time'
    WHEN SAFE_CAST(SPLIT(start_time, ':')[SAFE_OFFSET(0)] AS INT64) >= 23 
      OR SAFE_CAST(SPLIT(start_time, ':')[SAFE_OFFSET(0)] AS INT64) <= 5 THEN 'Late Night'
    ELSE 'Unknown'
  END as daypart,
  
  -- Standardize channel name (remove extra spaces, handle variants)
  CASE 
    WHEN UPPER(TRIM(channel_name)) IN ('RCTI', 'RCTI ') THEN 'RCTI'
    WHEN UPPER(TRIM(channel_name)) IN ('GTV', 'GTV ') THEN 'GTV'
    WHEN UPPER(TRIM(channel_name)) IN ('MNCTV', 'MNCTV ') THEN 'MNCTV'
    WHEN UPPER(TRIM(channel_name)) IN ('INEWS', 'INEWS ') THEN 'INEWS'
    ELSE UPPER(TRIM(channel_name))
  END as channel_name_std,
  
  -- Flag for data quality
  CASE 
    WHEN slot_price IS NULL THEN 'missing_price'
    WHEN slot_price < 1000 THEN 'invalid_price'
    WHEN slot_duration_sec IS NULL THEN 'missing_duration'
    ELSE 'valid'
  END as data_quality_flag

FROM combined_data
WHERE channel_name IS NOT NULL  -- Exclude any remaining nulls
  AND TRIM(channel_name) != ''   -- Exclude empty strings
"""

print("\n⏳ Creating cleaned table...")
print("   - Parsing hidden data from weird column")
print("   - Combining with existing data")
print("   - Cleaning and standardizing fields")
print("   - Adding calculated fields (daypart, etc.)")

job = bq_client.client.query(create_cleaned_query, location='US')
job.result()  # Wait for completion

print("✅ Table created successfully!")

# ==================================================================================
# VERIFY CLEANED TABLE
# ==================================================================================
print("\n" + "="*100)
print("2️⃣ VERIFYING CLEANED TABLE")
print("="*100)

verify_query = f"""
SELECT 
  COUNT(*) as total_records,
  COUNT(DISTINCT channel_name_std) as unique_channels,
  COUNT(DISTINCT program_name) as unique_programs,
  COUNT(DISTINCT broadcast_date) as unique_dates,
  COUNT(DISTINCT data_source) as data_sources,
  COUNTIF(slot_price IS NOT NULL AND slot_price > 1000) as records_with_valid_price,
  COUNTIF(data_quality_flag = 'valid') as high_quality_records,
  MIN(_modified) as earliest_modified,
  MAX(_modified) as latest_modified
FROM `{PROJECT_ID}.{DATASET_ID}.inventory_fta_cleaned`
"""

print("\n⏳ Querying cleaned table...")
verify_df = bq_client.query_to_dataframe(verify_query)
print("\n" + verify_df.to_string(index=False))

# ==================================================================================
# DATA SOURCE BREAKDOWN
# ==================================================================================
print("\n" + "="*100)
print("3️⃣ DATA SOURCE BREAKDOWN")
print("="*100)

source_query = f"""
SELECT 
  data_source,
  COUNT(*) as record_count,
  COUNT(DISTINCT channel_name_std) as channels,
  COUNT(DISTINCT program_name) as programs,
  COUNTIF(slot_price IS NOT NULL AND slot_price > 1000) as with_valid_price,
  ROUND(AVG(slot_price), 0) as avg_price
FROM `{PROJECT_ID}.{DATASET_ID}.inventory_fta_cleaned`
GROUP BY data_source
ORDER BY record_count DESC
"""

source_df = bq_client.query_to_dataframe(source_query)
print("\n" + source_df.to_string(index=False))

# ==================================================================================
# CHANNEL BREAKDOWN
# ==================================================================================
print("\n" + "="*100)
print("4️⃣ CHANNEL BREAKDOWN")
print("="*100)

channel_query = f"""
SELECT 
  channel_name_std as channel,
  COUNT(*) as total_slots,
  COUNT(DISTINCT program_name) as unique_programs,
  COUNTIF(slot_price IS NOT NULL AND slot_price > 1000) as slots_with_price,
  ROUND(COUNTIF(slot_price IS NOT NULL AND slot_price > 1000) * 100.0 / COUNT(*), 1) as pct_with_price,
  MIN(slot_price) as min_price,
  ROUND(AVG(slot_price), 0) as avg_price,
  MAX(slot_price) as max_price
FROM `{PROJECT_ID}.{DATASET_ID}.inventory_fta_cleaned`
GROUP BY channel_name_std
ORDER BY total_slots DESC
"""

channel_df = bq_client.query_to_dataframe(channel_query)
print("\n" + channel_df.to_string(index=False))

# ==================================================================================
# DAYPART BREAKDOWN
# ==================================================================================
print("\n" + "="*100)
print("5️⃣ DAYPART BREAKDOWN")
print("="*100)

daypart_query = f"""
SELECT 
  daypart,
  COUNT(*) as slot_count,
  COUNT(DISTINCT program_name) as unique_programs,
  ROUND(AVG(slot_price), 0) as avg_price,
  MIN(slot_price) as min_price,
  MAX(slot_price) as max_price
FROM `{PROJECT_ID}.{DATASET_ID}.inventory_fta_cleaned`
WHERE daypart != 'Unknown'
  AND slot_price IS NOT NULL
GROUP BY daypart
ORDER BY 
  CASE daypart
    WHEN 'Morning' THEN 1
    WHEN 'Afternoon' THEN 2
    WHEN 'Prime Time' THEN 3
    WHEN 'Late Night' THEN 4
  END
"""

daypart_df = bq_client.query_to_dataframe(daypart_query)
print("\n" + daypart_df.to_string(index=False))

# ==================================================================================
# DATA QUALITY SUMMARY
# ==================================================================================
print("\n" + "="*100)
print("6️⃣ DATA QUALITY SUMMARY")
print("="*100)

quality_query = f"""
SELECT 
  data_quality_flag,
  COUNT(*) as record_count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM `{PROJECT_ID}.{DATASET_ID}.inventory_fta_cleaned`
GROUP BY data_quality_flag
ORDER BY record_count DESC
"""

quality_df = bq_client.query_to_dataframe(quality_query)
print("\n" + quality_df.to_string(index=False))

# ==================================================================================
# SAMPLE CLEANED DATA
# ==================================================================================
print("\n" + "="*100)
print("7️⃣ SAMPLE CLEANED DATA")
print("="*100)

sample_query = f"""
SELECT 
  channel_name_std,
  program_name,
  broadcast_date,
  start_time,
  daypart,
  slot_duration_sec,
  slot_price,
  slot_category,
  data_source
FROM `{PROJECT_ID}.{DATASET_ID}.inventory_fta_cleaned`
WHERE slot_price IS NOT NULL
ORDER BY RAND()
LIMIT 10
"""

sample_df = bq_client.query_to_dataframe(sample_query)
print("\n📊 Random sample of cleaned records:")
print("\n" + sample_df.to_string(index=False))

# ==================================================================================
# FINAL SUMMARY
# ==================================================================================
print("\n" + "="*100)
print("💡 CLEANING SUMMARY")
print("="*100)

total_records = verify_df['total_records'].values[0]
valid_price = verify_df['records_with_valid_price'].values[0]
high_quality = verify_df['high_quality_records'].values[0]
channels = verify_df['unique_channels'].values[0]
programs = verify_df['unique_programs'].values[0]

existing_count = source_df[source_df['data_source'] == 'existing_format']['record_count'].values[0] if len(source_df[source_df['data_source'] == 'existing_format']) > 0 else 0
recovered_count = source_df[source_df['data_source'] == 'hidden_recovered']['record_count'].values[0] if len(source_df[source_df['data_source'] == 'hidden_recovered']) > 0 else 0

print(f"\n✅ Cleaned Table Created:")
print(f"   📊 Table: {PROJECT_ID}.{DATASET_ID}.inventory_fta_cleaned")
print(f"   📦 Total records: {total_records:,}")
print(f"   📺 Unique channels: {channels}")
print(f"   🎬 Unique programs: {programs}")
print(f"   📅 Date range: {verify_df['earliest_modified'].values[0]} to {verify_df['latest_modified'].values[0]}")

print(f"\n✅ Data Sources:")
print(f"   - Existing format: {existing_count:,} records")
print(f"   - Recovered (parsed): {recovered_count:,} records")
print(f"   - Total combined: {total_records:,} records")

print(f"\n✅ Data Quality:")
print(f"   - With valid price (>1000): {valid_price:,} ({valid_price/total_records*100:.1f}%)")
print(f"   - High quality (all fields): {high_quality:,} ({high_quality/total_records*100:.1f}%)")

print(f"\n✅ New Features Added:")
print(f"   ✓ daypart - Calculated from start_time")
print(f"   ✓ channel_name_std - Standardized channel names")
print(f"   ✓ data_quality_flag - Quality indicator")
print(f"   ✓ slot_price - Converted to INTEGER (no commas)")
print(f"   ✓ data_source - Track origin of each record")

print(f"\n✅ Ready for:")
print(f"   1. Matching with historical performance data")
print(f"   2. Building recommendation engine")
print(f"   3. Campaign planning and optimization")

print("\n✅ Cleaning complete!")