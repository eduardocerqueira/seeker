#date: 2023-11-20T16:46:07Z
#url: https://api.github.com/gists/f9b356f7339421dc4612dc87ae2cd3ff
#owner: https://api.github.com/users/ireneisdoomed

mock_l2g_gs_df = session.spark.createDataFrame(
  [
      (1, "variant1", "gene1", "positive"),
      (
          2,
          "variant2",
          "gene1",
          "negative",
      ),  # in the same locus as 1 and pointing to same gene, has to be dropped
      (
          3,
          "variant3",
          "gene1",
          "positive",
      ),  # in diff locus as 1 and pointing to same gene, has to be kept
      (
          4,
          "variant4",
          "gene2",
          "positive",
      ),  # in same locus as 1 and pointing to diff gene, has to be kept
  ],
  "studyLocusId LONG, variantId STRING, geneId STRING, goldStandardSet STRING",
)
mock_l2g_gs = L2GGoldStandard(
        _df=mock_l2g_gs_df, _schema=L2GGoldStandard.get_schema()
    )

"""
+------------+---------+------+---------------+
|studyLocusId|variantId|geneId|goldStandardSet|
+------------+---------+------+---------------+
|           1| variant1| gene1|       positive|
|           2| variant2| gene1|       negative|
|           3| variant3| gene1|       positive|
|           4| variant4| gene2|       positive|
+------------+---------+------+---------------+
"""

mock_sl_overlap_df = session.spark.createDataFrame(
        [(1, 2, "variant2"), (1, 4, "variant4")],
        "leftStudyLocusId LONG, rightStudyLocusId LONG, tagVariantId STRING",
    )
mock_sl_overlap = StudyLocusOverlap(
        _df=mock_sl_overlap_df, _schema=StudyLocusOverlap.get_schema()
    )
"""
+----------------+-----------------+------------+
|leftStudyLocusId|rightStudyLocusId|tagVariantId|
+----------------+-----------------+------------+
|               1|                2|    variant2|
|               1|                4|    variant4|
+----------------+-----------------+------------+
"""
square_overlaps = mock_sl_overlap.convert_to_square_matrix().df

(
     mock_l2g_gs_df.alias("left")
    # identify all the study loci that point to the same gene
    .withColumn("sl_same_gene", f.collect_set("studyLocusId").over(Window.partitionBy("geneId")))
    # identify all the study loci that have an overlapping variant
    .join(square_overlaps.alias("right"), (f.col("left.studyLocusId") == f.col("right.leftStudyLocusId")) & (f.col("left.variantId") == f.col("right.tagVariantId")), "left")
    .withColumn("overlaps", f.when(f.col("right.tagVariantId").isNotNull(), 1).otherwise(0))
    # drop redundant rows: where the variantid overlaps and the gene is "explained" by more than one study locus
    .filter(
        ~((f.size("sl_same_gene") > 1) & (f.col("overlaps") == 1))
    )
    .select(*cols_to_keep)
    .show()
)
"""
+------------+---------+------+---------------+                                 
|studyLocusId|variantId|geneId|goldStandardSet|
+------------+---------+------+---------------+
|           1| variant1| gene1|       positive|
|           3| variant3| gene1|       positive|
|           4| variant4| gene2|       positive|
+------------+---------+------+---------------+
"""
