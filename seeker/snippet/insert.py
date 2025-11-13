#date: 2025-11-13T17:04:40Z
#url: https://api.github.com/gists/4bdad1f027b0ffaf7b2a4bcd2bb6926c
#owner: https://api.github.com/users/chrisdel101

def insert_rows_into_sma_or_lma_results_or_standard_table(
            self,
            table_name: str,
            field_names: list[str],
            results_table_tuples: list[SolidMultiAnalyteResultsTableRowEntry | SolidMultiAnalyteStandardizedResultsTableRowEntry | LiquidMultiAnalyteResultsTableRowEntry | LiquidMultiAnalyteStandardizedResultsTableRowEntry]) -> list[int] | None:
        """
        Inserts rows into an ArcGIS table using a dictionary-to-field mapping.
        @return: list[int]: returns list of obj IDs.
        Parameters:
            @table_name (str): Name of the table inside the SDE.
            @field_names (list of str): List of field names in the table to insert data into.
            @results_table_tuples (list of tuples): List of row entry objects next to test_date.
        """
        if not results_table_tuples or len(results_table_tuples) == 0:
            logging.debug(f"insert_rows_into_sma_or_lma_results_or_standard_table: entry is empty or null")
            return
        try:
            table_path = f"{self.sde_path}\\{table_name}"
            object_ids = []
            inserts = 0
            non_duplicate_records_list = self.select_non_duplicate_records_sma_and_lma(
                                    table_name=table_name,
                                    field_names=field_names,
                                    records_to_check=results_table_tuples)
            with arcpy.da.InsertCursor(table_path, field_names) as cursor:
                for i, (row_entry_obj, _) in enumerate(non_duplicate_records_list):
                    if self.allow_live_inserts:
                        object_id = cursor.insertRow(list(row_entry_obj.__dict__.values())) 
                        inserts += 1
                        # track added rows
                        object_ids.append(object_id)
                if inserts:
                    logging.info(f"insert_rows_into_sma_or_lma_results_or_standard_table: Successfully inserted {inserts} rows into  {table_name}")
                else:
                    logging.debug(f"No rows inserted into {table_name}. Live inserts is {self.allow_live_inserts}.")
          
            return object_ids
        except Exception as e:
            self._run_if_cannot_open_exception(e, table_path)
            logging.error(f"Error in insert_rows_into_sma_or_lma_results_or_standard_table {table_path}: {e}", exc_info=True)
            raise