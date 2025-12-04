#date: 2025-12-04T16:57:36Z
#url: https://api.github.com/gists/8b4c8095889c2734a433c645a25fb36a
#owner: https://api.github.com/users/davidwinalda

# 5: Create Semantic Table - tv_spot_performance

# Create cleaned and structured tv_spot_performance table
print("\n" + "="*100)
print("🔨 CREATING SEMANTIC TABLE: tv_spot_performance")
print("="*100)

create_performance_table = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.tv_spot_performance` AS
WITH parsed_data AS (
  SELECT
    -- Basic info
    PARSE_DATE('%d/%m/%Y', Date) as date,
    Channel as channel,
    Description as program,
    Level_2 as program_type,
    
    -- Split program_type into 2 levels (Category:Subcategory)
    SPLIT(Level_2, ':')[SAFE_OFFSET(0)] as program_category,
    SPLIT(Level_2, ':')[SAFE_OFFSET(1)] as program_subcategory,
    
    -- Target audience parsing (FORMAT: STL Gender Age Occ Area SEC)
    Target_Variable as target,
    
    -- Extract Gender (position 1) - HANDLE NS/NS-*, ALL/ALL-*/GENDER-ALL/*-ALL, FM, M, F
    CASE
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(1)] = 'NS' THEN 'Not Defined'
      WHEN STARTS_WITH(SPLIT(Target_Variable, ' ')[SAFE_OFFSET(1)], 'NS-') THEN 'Not Defined'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(1)] = 'ALL' THEN 'All'
      WHEN STARTS_WITH(SPLIT(Target_Variable, ' ')[SAFE_OFFSET(1)], 'ALL-') THEN 'All'
      WHEN ENDS_WITH(SPLIT(Target_Variable, ' ')[SAFE_OFFSET(1)], '-ALL') THEN 'All'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(1)] = 'M' THEN 'Male'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(1)] = 'F' THEN 'Female'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(1)] = 'MF' THEN 'Male+Female'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(1)] = 'FM' THEN 'Male+Female'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(1)] IS NULL THEN 'Not Defined'
      ELSE 'Unknown'
    END as target_gender,
    
    -- Extract Age (position 2) - HANDLE NS/NS-*, ALL/ALL-*/AGE-ALL/*-ALL
    CASE
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(2)] = 'NS' THEN 'Not Defined'
      WHEN STARTS_WITH(SPLIT(Target_Variable, ' ')[SAFE_OFFSET(2)], 'NS-') THEN 'Not Defined'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(2)] = 'ALL' THEN 'All'
      WHEN STARTS_WITH(SPLIT(Target_Variable, ' ')[SAFE_OFFSET(2)], 'ALL-') THEN 'All'
      WHEN ENDS_WITH(SPLIT(Target_Variable, ' ')[SAFE_OFFSET(2)], '-ALL') THEN 'All'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(2)] IS NULL THEN 'Not Defined'
      ELSE SPLIT(Target_Variable, ' ')[SAFE_OFFSET(2)]
    END as target_age,
    
    -- Extract Occupation (position 3) - HANDLE NS/NS-OCC/NS-*, ALL/ALL-*/OCC-ALL/*-ALL + EXPANDED CODES
    CASE
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)] = 'NS' THEN 'Not Defined'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)] = 'NS-OCC' THEN 'Not Defined'
      WHEN STARTS_WITH(SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)], 'NS-') THEN 'Not Defined'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)] = 'ALL' THEN 'All'
      WHEN STARTS_WITH(SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)], 'ALL-') THEN 'All'
      WHEN ENDS_WITH(SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)], '-ALL') THEN 'All'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)] = 'STU' THEN 'Student'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)] = 'PROF' THEN 'Professional/Executive'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)] = 'SKIL' THEN 'Skill/Semi Skill'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)] = 'W' THEN 'Housewife'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)] = 'HW' THEN 'Housewife'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)] = 'E' THEN 'Entrepreneur'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)] = 'LAB' THEN 'Labourer'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)] = 'CLER' THEN 'Clerical Staff'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)] = 'RET' THEN 'Retired/Not Working'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)] = 'RTR' THEN 'Retired/Not Working'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)] IS NULL THEN 'Not Defined'
      ELSE SPLIT(Target_Variable, ' ')[SAFE_OFFSET(3)]
    END as target_occupation,
    
    -- Extract Area (position 4) - HANDLE NS/NS-LOC/NS-*, ALL/ALL-*/LOC-ALL/*-ALL + EXPANDED CITY CODES
    CASE
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'NS' THEN 'Not Defined'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'NS-LOC' THEN 'Not Defined'
      WHEN STARTS_WITH(SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)], 'NS-') THEN 'Not Defined'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'ALL' THEN 'All'
      WHEN STARTS_WITH(SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)], 'ALL-') THEN 'All'
      WHEN ENDS_WITH(SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)], '-ALL') THEN 'All'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'URB' THEN 'Urban'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'RUR' THEN 'Rural'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'JKT' THEN 'DKI Jakarta'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'BDG' THEN 'Bandung'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'SBY' THEN 'Surabaya'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'SMG' THEN 'Semarang'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'MDN' THEN 'Medan'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'MKS' THEN 'Makassar'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'DPR' THEN 'Denpasar'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'YGY' THEN 'Yogyakarta'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'PLG' THEN 'Palembang'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'BJM' THEN 'Banjarmasin'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'SKT' THEN 'Surakarta'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] = 'BTK' THEN 'Botabek Urban'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)] IS NULL THEN 'Not Defined'
      ELSE SPLIT(Target_Variable, ' ')[SAFE_OFFSET(4)]
    END as target_area,
    
    -- Extract SEC (position 5) - HANDLE NS/NS-*, ALL/ALL-SEC/ALL-*/SEC-ALL/*-ALL + EXPANDED CLASS CODES
    CASE
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)] = 'NS' THEN 'Not Defined'
      WHEN STARTS_WITH(SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)], 'NS-') THEN 'Not Defined'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)] = 'ALL' THEN 'All'
      WHEN STARTS_WITH(SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)], 'ALL-') THEN 'All'
      WHEN ENDS_WITH(SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)], '-ALL') THEN 'All'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)] = 'U' THEN 'Upper'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)] = 'U1' THEN 'Upper1'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)] = 'U2' THEN 'Upper2'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)] = 'M' THEN 'Middle'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)] = 'M1' THEN 'Middle1'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)] = 'M2' THEN 'Middle2'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)] = 'L' THEN 'Lower'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)] = 'ABC' THEN 'Upper Class (ABC)'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)] = 'D' THEN 'Middle Class (D)'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)] = 'E' THEN 'Lower Class (E)'
      WHEN SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)] IS NULL THEN 'Not Defined'
      ELSE SPLIT(Target_Variable, ' ')[SAFE_OFFSET(5)]
    END as target_sec,
    
    -- Time parsing
    Start_time as start_time,
    End_time as end_time,
    Duration as duration_seconds,
    
    -- Daypart classification
    CASE
      WHEN SAFE_CAST(SPLIT(Start_time, ':')[OFFSET(0)] AS INT64) BETWEEN 6 AND 11 THEN 'Morning'
      WHEN SAFE_CAST(SPLIT(Start_time, ':')[OFFSET(0)] AS INT64) BETWEEN 12 AND 17 THEN 'Afternoon'
      WHEN SAFE_CAST(SPLIT(Start_time, ':')[OFFSET(0)] AS INT64) BETWEEN 18 AND 22 THEN 'Prime Time'
      ELSE 'Late Night'
    END as daypart,
    
    -- Day classification
    FORMAT_DATE('%A', PARSE_DATE('%d/%m/%Y', Date)) as day_of_week,
    CASE 
      WHEN EXTRACT(DAYOFWEEK FROM PARSE_DATE('%d/%m/%Y', Date)) IN (1, 7) 
      THEN 1 ELSE 0 
    END as is_weekend,
    
    -- Performance metrics
    SAFE_CAST(_000s AS FLOAT64) as audience_000s,
    SAFE_CAST(TVR AS FLOAT64) as tvr,
    SAFE_CAST(Total_000s AS FLOAT64) as impressions,
    SAFE_CAST(Reach_000s_Not_cons_TH_0min AS FLOAT64) as reach_000s,
    SAFE_CAST(Share AS FLOAT64) as share,
    SAFE_CAST(`Index` AS FLOAT64) as target_index,
    
    -- Cost (remove currency formatting and convert)
    SAFE_CAST(
      REGEXP_REPLACE(REGEXP_REPLACE(Cost, r'Rp\\s*', ''), r'[,.]', '') 
      AS FLOAT64
    ) as cost,
    
    -- Additional metrics
    Average_Age_Class as avg_age_class,
    Median_Age_class as median_age_class
    
  FROM `{PROJECT_ID}.{DATASET_ID}.tv_raw_data`
  WHERE Date IS NOT NULL
    AND Channel IS NOT NULL
)
SELECT 
  *,
  
  -- Source type classification
  CASE 
    WHEN channel IN ('RCTI', 'GTV', 'MNCTV', 'INEWS', 'SCTV', 'INDOSIAR', 
                     'TRANS7', 'TRANSTV', 'ANTV', 'METRO', 'TVONE', 
                     'KOMPASTV', 'NET', 'RTV')
    THEN 'FTA'
    ELSE 'DTH'
  END as source_type,
  
  -- Cost per TVR (2 decimal places)
  CASE 
    WHEN tvr > 0 THEN ROUND(cost / tvr, 2)
    ELSE NULL 
  END as cost_per_tvr,
  
  -- Cost per Impression (2 decimal places)
  CASE 
    WHEN impressions > 0 THEN ROUND(cost / impressions, 2)
    ELSE NULL 
  END as cost_per_impression,
  
  -- Frequency (2 decimal places)
  CASE 
    WHEN reach_000s > 0 THEN ROUND(impressions / reach_000s, 2)
    ELSE NULL 
  END as frequency,
  
  -- GRP (same as TVR for individual spots)
  tvr as grp,
  
  -- Unique identifier
  GENERATE_UUID() as spot_id,
  CURRENT_TIMESTAMP() as created_at
  
FROM parsed_data
WHERE tvr IS NOT NULL
"""

print("\n⏳ Creating table (this may take 20-40 seconds)...")
job = bq_client.client.query(create_performance_table)
job.result()
print("✅ tv_spot_performance table created")
print("   📊 Program: 3 columns (type, category, subcategory)")
print("   👥 Demographics: 5 attributes (gender, age, occupation, area, sec)")
print("   📈 Metrics: audience_000s, tvr, impressions, reach_000s, frequency")

# Verify
verify = f"""
SELECT 
  COUNT(*) as total_records,
  COUNT(DISTINCT channel) as unique_channels,
  COUNT(DISTINCT program) as unique_programs,
  COUNT(DISTINCT program_category) as unique_categories,
  COUNT(DISTINCT program_subcategory) as unique_subcategories,
  COUNT(DISTINCT target) as unique_targets,
  MIN(date) as min_date,
  MAX(date) as max_date,
  ROUND(AVG(tvr), 2) as avg_tvr,
  ROUND(AVG(impressions), 1) as avg_impressions,
  ROUND(AVG(reach_000s), 1) as avg_reach_000s,
  ROUND(AVG(frequency), 2) as avg_frequency
FROM `{PROJECT_ID}.{DATASET_ID}.tv_spot_performance`
"""
result = bq_client.query_to_dataframe(verify)
print("\n📊 Table Summary:")
print(result.to_string(index=False))

# Check program category/subcategory distribution
program_dist = f"""
SELECT 
  program_category,
  program_subcategory,
  COUNT(*) as count,
  ROUND(AVG(tvr), 3) as avg_tvr,
  ROUND(AVG(frequency), 2) as avg_frequency
FROM `{PROJECT_ID}.{DATASET_ID}.tv_spot_performance`
GROUP BY program_category, program_subcategory
ORDER BY count DESC
LIMIT 15
"""
prog_result = bq_client.query_to_dataframe(program_dist)
print("\n📺 Program Category/Subcategory Distribution (Top 15):")
print(prog_result.to_string(index=False))

# Check 5-attribute distribution - CORRECTED PARSING
distribution_check = f"""
SELECT 
  target_gender,
  target_age,
  target_occupation,
  target_area,
  target_sec,
  COUNT(*) as count,
  ROUND(AVG(tvr), 3) as avg_tvr
FROM `{PROJECT_ID}.{DATASET_ID}.tv_spot_performance`
GROUP BY target_gender, target_age, target_occupation, target_area, target_sec
ORDER BY count DESC
LIMIT 25
"""
dist_result = bq_client.query_to_dataframe(distribution_check)
print("\n📊 Target Attribute Distribution (Top 25) - 5 attributes:")
print(dist_result.to_string(index=False))

# Sample Male 25-34 data (matches specs.md target)
sample_male_25_34 = f"""
SELECT 
  target, target_gender, target_age, target_occupation, target_area, target_sec,
  channel, program, program_category, program_subcategory, daypart, 
  ROUND(tvr, 2) as tvr, 
  impressions,
  reach_000s,
  ROUND(frequency, 2) as frequency,
  target_index,
  cost
FROM `{PROJECT_ID}.{DATASET_ID}.tv_spot_performance`
WHERE target_gender = 'Male'
  AND target_age = '25-34'
  AND tvr > 0
ORDER BY tvr DESC
LIMIT 10
"""
sample_df = bq_client.query_to_dataframe(sample_male_25_34)
print("\n👀 Top 10 Spots for Male 25-34 (from specs.md):")
print(sample_df.to_string(index=False))