#date: 2025-01-16T17:07:55Z
#url: https://api.github.com/gists/1da63852b2f1b4d56472cd03bb4b864e
#owner: https://api.github.com/users/pdpark-wex

export AWS_SDK_LOAD_CONFIG=1

aws_assume_athena_role() {
	aws_assume_role_profile=${1:-"_assume_athena_role"}
	aws_profile=${2:-"data"}

	while read key value
	do
	  $(aws configure --profile ${aws_assume_role_profile} set ${key} ${value})
	  unset key value
	done <<< $(aws sts assume-role \
	  --role-arn=$(aws configure get role_arn --profile ${aws_profile}) \
	  --role-session-name=athena-role-$USER-$(date +%Y-%m%d-%M%S) \
	  --query 'Credentials'  |
	jq -r '{"aws_access_key_id": "**********": .SecretAccessKey, "aws_session_token": .SessionToken} | to_entries[] | .key + " " + .value')
}

sqlwb() {
    aws_assume_athena_role &&
    # Run job in background, throw away any output, and disown it
    eval "java -jar /Applications/SQLWorkbenchJ.app/Contents/Java/sqlworkbench.jar" &>/dev/null & disown;
}

ducks3 () {
        (
                "$@"
                eval "$(aws configure export-credentials --profile ${AWS_PROFILE} --format env)"
                duckdb
        )
}

duckpg() {
    if [ $# -lt 4 ]
    then
      echo "Usage: "**********"
      return
    fi
    
    export PGHOST=$1
    export PGUSER=$2
    export PGDATABASE=$3
    export PGPASSWORD= "**********"

    duckdb
}

pip-upgrade-all-envs() {
  for VERSION in $(pyenv versions --bare)
  do
    pyenv shell ${VERSION}
    pip install --upgrade pip pip-tools
  done
  pyenv shell --unset
}

mkcd () {
     mkdir -p "$1" && cd $_
}

s3sup() {
    if [ $# -lt 2 ] || [ $# -gt 3 ]; then
        echo "Usage: $0 <env> <data_type> [format]"
        return 1
    fi

    local env=${1}
    local data_type=${2}
    local format=${3:-yyn}

    # Validate the format argument
    if [[ ${#format} -ne 3 ]] || [[ ! $format =~ ^[yn]{3}$ ]]; then
        echo "Error: format argument must be a three-character string containing only 'y' or 'n'."
        echo "The format argument must be a three-character string where each character can be 'y' or 'n'."
        echo "The first character indicates whether the formatted dts will be output."
        echo "The second character indicates whether the full path will be output."
        echo "The third character indicates whether the final command will be run."
        return 1
    fi

    # Mapping of data type abbreviations to actual data type names
    declare -A data_type_map=(
        ["sbc"]="supplier-business-prices/latest"
        ["ppa"]="product-price-account/latest"
        ["pim"]="pim-snapshots"
        ["abp"]="ahri-bundle-products"
        ["acbir"]="ahri-config-bundle-ir-search-docs"
        ["acbp"]="ahri-config-bundle-products"
        ["acb"]="ahri-config-bundles"
        ["ents"]="biz-entitlement-params-snapshot"
        ["cats"]="part-categories"
        ["pir"]="part-ir-search-docs"
    )

    # If data_type is "list", list available abbreviations and mapped values
    if [ "$data_type" = "list" ]; then
        echo "Available data type abbreviations and their mapped values:"
        for key in ${(k)data_type_map}; do
            echo "$key: ${data_type_map[$key]}"
        done
        return 0
    fi

    # Substitute the data type name for the abbreviation if it exists in the mapping
    if [[ -n "${data_type_map[$data_type]}" ]]; then
        data_type=${data_type_map[$data_type]}
    fi

    local s3_path="s3://payzer-rdf/${env}/${data_type}/"
    local dts=$(aws s3 ls ${s3_path} | tail -1 | sed 's/ //g' | sed 's/PRE//g' | sed 's:/*$::')

    # Format the dts string and replace trailing Z with UTC
    local formatted_dts=$(echo $dts | sed 's/\(....\)\(..\)\(..\)T\(..\)\(..\)\(..\)\(.*\)/\1-\2-\3 \4:\5:\6\7/' | sed 's/Z$/UTC/')

    local full_path="${s3_path}${dts}/"

    # Output based on format argument
    if [ "${format:0:1}" = "y" ]; then
        echo "Latest directory: ${formatted_dts}"
    fi

    if [ "${format:1:1}" = "y" ]; then
        echo $full_path
    fi

    # Run the command to list the contents of the full path if specified
    if [ "${format:2:1}" = "y" ]; then
        aws s3 ls ${full_path}
    fi
}
}" = "y" ]; then
        aws s3 ls ${full_path}
    fi
}
