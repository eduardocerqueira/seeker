#date: 2021-12-27T17:04:04Z
#url: https://api.github.com/gists/0313ff529056b4f027568b85d79a33fa
#owner: https://api.github.com/users/garystafford

export DATA_LAKE_BUCKET="<your_data_lake_bucket_name>"
spark-submit \
    --jars /usr/lib/spark/jars/spark-avro.jar,/usr/lib/hudi/hudi-utilities-bundle.jar \
    --conf spark.sql.catalogImplementation=hive \
    --class org.apache.hudi.utilities.deltastreamer.HoodieDeltaStreamer /usr/lib/hudi/hudi-utilities-bundle.jar \
    --table-type MERGE_ON_READ \
    --source-ordering-field __source_ts_ms \
    --props "s3://${DATA_LAKE_BUCKET}/hudi/deltastreamer_artists_apicurio.properties" \
    --source-class org.apache.hudi.utilities.sources.AvroDFSSource \
    --target-base-path "s3://${DATA_LAKE_BUCKET}/moma/artists/" \
    --target-table moma.artists \
    --schemaprovider-class org.apache.hudi.utilities.schema.SchemaRegistryProvider \
    --enable-sync \
    --continuous \
    --op UPSERT