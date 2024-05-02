#date: 2024-05-02T16:53:03Z
#url: https://api.github.com/gists/a282b01c2a1c218bfd86c77f051321cd
#owner: https://api.github.com/users/quinnspraut

clear

color_gray='\e[90m'
color_green='\e[32m'
color_teal='\e[96m'
color_none='\e[0m'

echo -e "Welcome to the ${color_teal}BigQuery Table Optimizer${color_none}!"
echo -e "This script can help you add partitioning and clustering to your BigQuery tables, which can help you save costs and improve query performance."
echo -e "Note: you must have the ${color_teal}gcloud${color_none} CLI installed and authenticated to use this script."
echo -e ""
echo -e "Press any key to continue..."
read -n 1 -s

table_confirmed=false
while [ "$table_confirmed" = false ]; do

  clear
  echo -e "${color_gray}Optimizing Table: \`?.?.?\`${color_none}"
  echo -e ""
  echo -e "Please enter the name of the ${color_teal}GCP project${color_none} where your BigQuery dataset is located ${color_green}(default: webfx-data)${color_none}:"
  read project
  if [ -z "$project" ]; then
    project="webfx-data"
  fi

  dataset=""
  while [ -z "$dataset" ]; do
    clear
    echo -e "${color_gray}Optimizing Table: \`${project}.?.?\`${color_none}"
    echo -e ""
    echo -e "Please enter the name of the ${color_teal}BigQuery dataset${color_none} where your table is located:"
    read dataset
  done

  table=""
  while [ -z "$table" ]; do
    clear
    echo -e "${color_gray}Optimizing Table: \`${project}.${dataset}.?\`${color_none}"
    echo -e ""
    echo -e "Please enter the name of the ${color_teal}BigQuery table${color_none} you would like to optimize:"
    read table
  done

  clear
  echo -e "${color_gray}Optimizing Table: \`${project}.${dataset}.${table}\`${color_none}"
  echo -e ""
  echo -e "Is ${color_teal}\`$project.$dataset.$table\`${color_none} correct? (y/N)"
  read confirm
  if [ "$confirm" = "y" ]; then
    table_confirmed=true
  fi

  echo -e ""
  echo -e "Confirming that table ${color_teal}\`$project.$dataset.$table\`${color_none} exists..."
  echo -e "${color_gray}> bq show $project:$dataset.$table${color_none}"
  bq show $project:$dataset.$table > /dev/null
  if [ $? -ne 0 ]; then
    echo -e ""
    echo -e "Table ${color_teal}\`$project.$dataset.$table\`${color_none} was not found. Please try again."
    table_confirmed=false
    echo -e "Press any key to start over..."
    read -n 1 -s
  fi

done

partition_column=""
partition_type=""
cluster_columns=""

clear
echo -e "${color_gray}Optimizing Table: \`${project}.${dataset}.${table}\`${color_none}"
echo -e ""
echo -e "Would you like to add ${color_teal}partitioning${color_none} to this table? (y/N)"
read partitioning
if [ "$partitioning" = "y" ]; then
  clear
  echo -e "${color_gray}Optimizing Table: \`${project}.${dataset}.${table}\`${color_none}"
  echo -e ""
  echo -e "Finding columns that can be used for partitioning..."
  echo -e "Note: only columns of type ${color_teal}TIMESTAMP${color_none}, ${color_teal}DATE${color_none}, and ${color_teal}DATETIME${color_none} can be used for partitioning at this time."
  echo -e "${color_gray}> bq show --format=prettyjson $project:$dataset.$table${color_none}"
  columns=$(bq show --format=prettyjson $project:$dataset.$table | jq '.schema.fields[] | select(.type == "TIMESTAMP" or .type == "DATE" or .type == "DATETIME") | .name' | tr -d '"')
  if [ -z "$columns" ]; then
    echo -e ""
    echo -e "No columns found that can be used for partitioning. Exiting!"
    exit 1
  else
    default_partition_column=""
    if [[ $columns == *"Metric_Date"* ]]; then
      default_partition_column="Metric_Date"
    fi
    while [[ -z "$partition_column" || $columns != *"$partition_column"* ]]; do
      echo -e ""
      echo -e "${color_teal}$columns${color_none}"
      echo -e ""
      if [ -z "$default_partition_column" ]; then
        echo -e "Please enter the name of the column you would like to use for partitioning:"
      else
        echo -e "Please enter the name of the column you would like to use for partitioning ${color_green}(default: Metric_Date)${color_none}:"
      fi
      read partition_column
      if [ -z "$partition_column" ] && [ ! -z "$default_partition_column" ]; then
        partition_column=$default_partition_column
      fi
    done

    while [[ -z "$partition_type" || "$partition_type" != "YEAR" && "$partition_type" != "MONTH" && "$partition_type" != "DAY" && "$partition_type" != "HOUR" ]]; do
      echo -e ""
      echo -e "${color_teal}YEAR${color_none}"
      echo -e "${color_teal}MONTH${color_none}"
      echo -e "${color_teal}DAY${color_none}"
      echo -e "${color_teal}HOUR${color_none}"
      echo -e ""
      echo -e "Please enter the type of partitioning you would like to use ${color_green}(default: MONTH)${color_none}:"
      read partition_type
      if [ -z "$partition_type" ]; then
        partition_type="MONTH"
      fi
    done
  fi
fi

clear
echo -e "${color_gray}Optimizing Table: \`${project}.${dataset}.${table}\`${color_none}"
echo -e ""
echo -e "Would you like to add ${color_teal}clustering${color_none} to this table? (y/N)"
read clustering
if [ "$clustering" = "y" ]; then
  clear
  echo -e "${color_gray}Optimizing Table: \`${project}.${dataset}.${table}\`${color_none}"
  echo -e ""
  echo -e "Finding columns that can be used for clustering..."
  echo -e "Note: only columns of type ${color_teal}STRING${color_none}, ${color_teal}INTEGER${color_none}, ${color_teal}FLOAT${color_none}, ${color_teal}BOOLEAN${color_none}, ${color_teal}TIMESTAMP${color_none}, ${color_teal}DATE${color_none}, and ${color_teal}DATETIME${color_none} can be used for clustering at this time."
  echo -e "${color_gray}> bq show --format=prettyjson $project:$dataset.$table${color_none}"
  columns=$(bq show --format=prettyjson $project:$dataset.$table | jq '.schema.fields[] | select(.type == "STRING" or .type == "INTEGER" or .type == "FLOAT" or .type == "BOOLEAN" or .type == "TIMESTAMP" or .type == "DATE" or .type == "DATETIME") | .name' | tr -d '"')
  if [ -z "$columns" ]; then
    echo -e ""
    echo -e "No columns found that can be used for clustering. Exiting!"
    exit 1
  else
    default_cluster_column=""
    if [[ $columns == *"External_ID"* ]]; then
      default_cluster_column="External_ID"
    fi
    all_columns_valid=false
    while [[ -z "$cluster_columns" || "$all_columns_valid" == "false" ]]; do
      echo -e ""
      echo -e "${color_teal}$columns${color_none}"
      echo -e ""
      if [ -z "$default_cluster_column" ]; then
        echo -e "Please enter a comma-separated list of columns you would like to use for clustering:"
      else
        echo -e "Please enter a comma-separated list of columns you would like to use for clustering ${color_green}(default: External_ID)${color_none}:"
      fi
      read cluster_columns
      if [ -z "$cluster_columns" ] && [ ! -z "$default_cluster_column" ]; then
        cluster_columns=$default_cluster_column
      fi
      IFS=',' read -ra ADDR <<< "$cluster_columns"
      all_columns_valid=true
      for i in "${ADDR[@]}"; do
        if [[ $columns != *"$i"* ]]; then
          echo -e "Column ${color_teal}$i${color_none} is not valid for clustering."
          all_columns_valid=false
          break
        fi
      done
    done
  fi
fi

clear
echo -e "${color_gray}Optimizing Table: \`${project}.${dataset}.${table}\`${color_none}"
echo -e ""
echo -e "Please review the following changes that will be made to the table:"
echo -e ""
if [ "$partitioning" = "y" ]; then
  echo -e "  - Add ${color_teal}${partition_type} partitioning${color_none} by column: ${color_teal}$partition_column${color_none}"
fi
if [ "$clustering" = "y" ]; then
  if [[ "$cluster_columns" == *,* ]]; then
    echo -e "  - Add ${color_teal}clustering${color_none} by columns: ${color_teal}$cluster_columns${color_none}"
  else
    echo -e "  - Add ${color_teal}clustering${color_none} by column: ${color_teal}$cluster_columns${color_none}"
  fi
fi

echo -e ""
echo -e "Would you like to proceed with these changes? (y/N)"
read apply_changes
if [ "$apply_changes" = "y" ]; then
  clear
  echo -e "${color_gray}Optimizing Table: \`${project}.${dataset}.${table}\`${color_none}"
  echo -e ""
  echo -e "Applying changes to table ${color_teal}\`$project.$dataset.$table\`${color_none}..."
  echo -e "Depending on the size of the table, this may take some time. Please be patient & do not interrupt the process."
  
  # Step 1: Create a temporary table to house the raw data
  command="bq query --use_legacy_sql=false --destination_table $project:$dataset.${table}___temp_opt 'SELECT * FROM \`$project.$dataset.$table\`;'"
  echo -e "${color_gray}> $command${color_none}"
  bash -c "$command" > /dev/null 2>&1 || exit 1

  # Step 2: Delete the original table
  command="bq rm -f $project:$dataset.$table"
  echo -e "${color_gray}> $command${color_none}"
  bash -c "$command" > /dev/null 2>&1 || exit 1

  # Step 3: Create the new table with partitioning and clustering
  command="bq query --use_legacy_sql=false --destination_table $project:$dataset.$table"
  if [ "$partitioning" = "y" ]; then
    command="$command --time_partitioning_field $partition_column --time_partitioning_type $partition_type"
  fi
  if [ "$clustering" = "y" ]; then
    command="$command --clustering_fields $cluster_columns"
  fi
  command="$command 'SELECT * FROM \`$project.$dataset.${table}___temp_opt\`;'"
  echo -e "${color_gray}> $command${color_none}"
  bash -c "$command" > /dev/null 2>&1 || exit 1

  # Step 4: Delete the temporary table
  command="bq rm -f $project:$dataset.${table}___temp_opt"
  echo -e "${color_gray}> $command${color_none}"
  bash -c "$command" > /dev/null 2>&1 || exit 1

  echo -e ""
  echo -e "Changes were successfully applied!"
  echo -e "Table ${color_teal}\`$project.$dataset.$table\`${color_none} has been optimized with the specified settings."
  echo -e "You can now run queries on this table with improved performance and cost savings."
else 
  echo -e ""
  echo -e "Changes were not applied. Exiting!"
  exit 1
fi

exit 0