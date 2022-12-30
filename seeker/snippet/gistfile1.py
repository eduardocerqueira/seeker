#date: 2022-12-30T16:53:32Z
#url: https://api.github.com/gists/e4d4238a466e25de8f7fb01bf9683298
#owner: https://api.github.com/users/jml1996

"""
# [result of this saved to distinct_rhs_uuids_updated_after_dec_15_v5.csv]

select distinct(uuid) as rhs_uuid_str
from mergedcompanymetadata join extractedorganization
on mergedcompanymetadata.extracted_org_id = extractedorganization.id
where extractedorganization.updated_dt > '2022-12-15'::timestamp
and mergedcompanymetadata.dre = 'SAIC_MOBILE'
and extractedorganization.created_dt < '2022-12-14'::timestamp; -- 371,841 on 2022-12-30
"""



# [ipy session]

from dmb_spark import pyspark_types
from diagnostics.ownerships import compute_ownership_diagnostics
z = pyspark_util.load_df('/Users/joshlovins/distinct_rhs_uuids_updated_after_dec_15_v5.csv', fmt='csv')
z = z.withColumn('rhs_uuid', pyspark_types.cast_uuid_to_binary('rhs_uuid_str'))

before = pyspark_util.load_df('s3://wirescreen-stage-data-pipeline/merging/2022-12-15T23:01:51.868228+00:00/merged_graph_relationships/')
before_filtered = before.join(z, ['rhs_uuid'], 'inner').filter((F.col('relationship_type') == 'has_shares') & (F.col('fraction').isNotNull()) & (F.col('end_date').isNull()))
after = pyspark_util.load_df('s3://wirescreen-stage-data-pipeline/merging/2022-12-29T23:01:53.066525+00:00//merged_graph_relationships/')
after_filtered = after.join(z, ['rhs_uuid'], 'inner').filter((F.col('relationship_type') == 'has_shares') & (F.col('fraction').isNotNull()) & (F.col('end_date').isNull()))

before_res = compute_ownership_diagnostics(before)
before_with_fract = before_res.withColumn('fraction', F.col('has_complete_ownership') / (sum(before_res[x] for x in before_res.columns)))
print("Before:")
before_with_fract.show()

after_res = compute_ownership_diagnostics(after)
after_with_fract = after_res.withColumn('fraction', F.col('has_complete_ownership') / (sum(after_res[x] for x in after_res.columns)))
print("After:")
after_with_fract.show()


"""
Before:
+----------------+---------------------+----------------------------+------------------+----------------------+----------------------+------------------+                                    
|has_no_ownership|has_unknown_ownership|has_less_than_half_ownership|has_most_ownership|has_complete_ownership|has_too_much_ownership|          fraction|
+----------------+---------------------+----------------------------+------------------+----------------------+----------------------+------------------+
|               0|               479154|                      174420|            165736|               3815510|               2119168|0.5649269735154993|
+----------------+---------------------+----------------------------+------------------+----------------------+----------------------+------------------+

After:
+----------------+---------------------+----------------------------+------------------+----------------------+----------------------+------------------+                                    
|has_no_ownership|has_unknown_ownership|has_less_than_half_ownership|has_most_ownership|has_complete_ownership|has_too_much_ownership|          fraction|
+----------------+---------------------+----------------------------+------------------+----------------------+----------------------+------------------+
|               0|               483677|                      166081|            159686|               3660952|               2305562|0.5402855212502793|
+----------------+---------------------+----------------------------+------------------+----------------------+----------------------+------------------+
"""

before_filtered.count()
"""
1405232
"""



after_filtered.count()
""""
2550348
"""







before_res_alt = before_filtered.filter((F.col('relationship_type') == 'has_shares') & (F.col('fraction').isNotNull())).groupBy('rhs_uuid').agg(F.sum('fraction'))
after_res_alt = after_filtered.filter((F.col('relationship_type') == 'has_shares') & (F.col('fraction').isNotNull())).groupBy('rhs_uuid').agg(F.sum('fraction'))

before_res_alt_avg = before_res_alt.agg(F.avg('sum(fraction)'))
after_res_alt_avg = after_res_alt.agg(F.avg('sum(fraction)'))

before_res_alt_avg.show()
after_res_alt_avg.show()

before_res_alt.show()
after_res_alt.show()


"""
+------------------+                                                                                                                                                                         
|avg(sum(fraction))|
+------------------+
| 4.308788415691441|
+------------------+

+------------------+                                                                                                                                                                         
|avg(sum(fraction))|
+------------------+
|2.6084438263178975|
+------------------+

+--------------------+-------------------+                                                                                                                                                   
|            rhs_uuid|      sum(fraction)|
+--------------------+-------------------+
|[00 02 65 9E 81 A...| 0.7857142857142857|
|[00 02 FE 28 0A 6...|                1.0|
|[00 04 53 BD 0E B...|                2.0|
|[00 06 09 8A B7 6...| 0.9957805907172996|
|[00 08 4D 89 6A 5...|                2.0|
|[00 0B 91 D5 E0 9...| 0.9999999751196743|
|[00 0C 65 FA 0D E...| 1.0307692307692307|
|[00 0C 91 2D 90 3...|                1.0|
|[00 0E B6 77 D9 C...|0.21739130434782608|
|[00 0E D1 53 07 A...| 1.9500237304224017|
|[00 10 3B A6 74 5...|                1.0|
|[00 10 DF BC FF 7...| 2.3333333333333335|
|[00 11 6A D8 98 1...|                2.0|
|[00 12 FC E4 64 6...|                1.0|
|[00 14 A7 A8 D7 C...|                1.0|
|[00 15 68 F9 22 2...|                2.0|
|[00 16 1E E0 92 C...|                1.0|
|[00 17 D1 EC 8D B...|                2.0|
|[00 19 CE D5 00 8...|                1.0|
|[00 1B 09 7B B5 B...|                2.0|
+--------------------+-------------------+
only showing top 20 rows

+--------------------+-------------------+                                                                                                                                                   
|            rhs_uuid|      sum(fraction)|
+--------------------+-------------------+
|[00 02 65 9E 81 A...| 1.7999999999999998|
|[00 02 FE 28 0A 6...|                2.0|
|[00 04 53 BD 0E B...|                2.0|
|[00 06 09 8A B7 6...|  2.147679324894515|
|[00 08 4D 89 6A 5...|                2.0|
|[00 0B 91 D5 E0 9...|  2.640095739493038|
|[00 0C 65 FA 0D E...| 2.0307692307692307|
|[00 0C 91 2D 90 3...| 2.2272727272727275|
|[00 0E B6 77 D9 C...|0.43478260869565216|
|[00 0E D1 53 07 A...| 3.5674893213099197|
|[00 10 3B A6 74 5...|                2.0|
|[00 10 DF BC FF 7...| 3.3333333333333335|
|[00 11 6A D8 98 1...|                3.0|
|[00 12 FC E4 64 6...|                2.0|
|[00 14 A7 A8 D7 C...|                2.0|
|[00 15 68 F9 22 2...|                2.0|
|[00 16 1E E0 92 C...|                2.0|
|[00 17 D1 EC 8D B...|                2.0|
|[00 19 CE D5 00 8...|                2.0|
|[00 1B 09 7B B5 B...|                3.0|
+--------------------+-------------------+
only showing top 20 rows
"""