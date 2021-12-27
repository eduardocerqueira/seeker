#date: 2021-12-27T17:06:48Z
#url: https://api.github.com/gists/6e2b6318b04cff685058a3a6d864e499
#owner: https://api.github.com/users/garystafford

export DATA_LAKE_BUCKET="<your_data_lake_bucket_name>"
spark-submit \
    --jars /usr/lib/spark/jars/spark-avro.jar,/usr/lib/hudi/hudi-utilities-bundle.jar \
    --conf spark.sql.catalogImplementation=hive \
    --class org.apache.hudi.utilities.deltastreamer.HoodieDeltaStreamer /usr/lib/hudi/hudi-utilities-bundle.jar \
    --table-type MERGE_ON_READ \
    --source-ordering-field __source_ts_ms \
    --props "s3://${DATA_LAKE_BUCKET}/hudi/deltastreamer_artworks_apicurio.properties" \
    --source-class org.apache.hudi.utilities.sources.AvroDFSSource \
    --target-base-path "s3://${DATA_LAKE_BUCKET}/moma/artworks/" \
    --target-table moma.artworks \
    --schemaprovider-class org.apache.hudi.utilities.schema.SchemaRegistryProvider \
    --enable-sync \
    --op BULK_INSERT \ # INSERT
    --filter-dupes