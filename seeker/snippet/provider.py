#date: 2024-06-06T17:04:45Z
#url: https://api.github.com/gists/cb919327f2866f9bbee90e54141ce032
#owner: https://api.github.com/users/tabrezm

            # Check for corrupt records
            corrupt_records_df = df.filter(F.col(CORRUPT_RECORD_COLUMN).isNotNull())
            if corrupt_records_df.head():
                # Remove invalid characters
                corrupt_records_df = corrupt_records_df.withColumn(
                    CORRUPT_RECORD_COLUMN,
                    F.regexp_replace(
                        CORRUPT_RECORD_COLUMN,
                        r"[^\u0009\u000A\u000D\u0020-\uD7FF\uE000-\uFFFD\u10000-\u10FFFF]",
                        "",
                    ),
                )

                # Reprocess corrupt records
                from_xml_options = {
                    "rowTag": row_tag,
                    "mode": "FAILFAST" if self._is_dev_env else "PERMISSIVE",
                    "columnNameOfCorruptRecord": CORRUPT_RECORD_COLUMN,
                }
                reprocessed_df = corrupt_records_df.withColumn(
                    CORRUPT_RECORD_COLUMN,
                    from_xml(
                        F.col(CORRUPT_RECORD_COLUMN),
                        self.schema,
                        options=from_xml_options,
                    ),
                )

                # Flatten the reprocessed data
                reprocessed_df = reprocessed_df.select(
                    *[f"{CORRUPT_RECORD_COLUMN}.{col}" for col in self.columns],
                    f"{CORRUPT_RECORD_COLUMN}.{CORRUPT_RECORD_COLUMN}",
                    input_file_path(),
                )