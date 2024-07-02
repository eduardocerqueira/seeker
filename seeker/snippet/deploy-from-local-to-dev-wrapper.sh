#date: 2024-07-02T16:49:34Z
#url: https://api.github.com/gists/27367cadfff22a99a017be37f95c1bc7
#owner: https://api.github.com/users/lcatlett

#!/bin/bash
# Wrapper script to deploy code from local to dev.
# This script enhances standard "git push" deployment processes by adding:
#   - Parallel execution of code and deployment tasks on many sites
#   - Setup of SSH and git configuration to avoid password prompts
#   - Race condition handling for workflows, terminus commands, and git commands
#   - Configuration of environment variables commonly required in enterprise deployment scripts
#   - Automatic retries of failed site deployments
# To use this script, run:
# ./deploy-from-local-to-dev-wrapper.sh

current_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
ENV=dev
# Array of sites the script will run againt. Commonly the output of a terminus command. Eg: SITES=$(terminus org:sites [your-pantheon-org] --field name --tag deploy --format=list)
SITES=$(terminus org:sites [your-pantheon-org] --field name --tag deploy --format=list)

# export vars to be used in other scripts
export current_dir=$current_dir
export ENV=$ENV
export SITES=("${SITES[@]}")

# Create logs directory if it does not exist.
mkdir -p "${current_dir}/logs"

# Create file with a list of sites to retry in failed-sites-<env>.txt.
# A site will be added to this list if code push or deployment tasks fail.
touch "${current_dir}/logs/failed-sites-$ENV.txt"

# Run code deployment tasks to Pantheon dev environment in parallel.
# Adjust the number of jobs by modifying the --jobs flag. eg --jobs 30
function deploy_code_to_dev() {
    local SITES=("$@")
    echo "Running code push tasks to Pantheon $ENV environment"
    parallel --jobs 25 ./deploy-dev "${SITES[@]}" "$ENV" "$current_dir"
}

# Run standard site deployment tasks such as drush commands in the Pantheon dev environment in parallel.
# Adjust the number of jobs by modifying the --jobs flag. eg --jobs 30

function run_deployment_tasks() {
    local SITES=("$@")
    echo "Running deployment tasks on site $ENV environment"
    parallel --jobs 25 ./deploy-tasks "${SITES[@]}" $ENV "$current_dir"
}

# Retry failed sites for up to 2 hours until all sites are deployed successfully.
# A site can be in the failed-sites-dev.txt file if it fails to deploy code or run deployment tasks.
# deploy_code_to_dev is re-run for each failed site listed in failed-sites-dev.txt with the [code] tag.
# run_deployment tasks is re-run for each failed site listed in failed-sites-dev.txt with the [deploy] tag.
# Update timeout to change the maximum time to retry failed sites.
function retry_failed_deployments() {
    echo "Retrying failed sites in the Pantheon $ENV environment" >>"${current_dir}/logs/deploy-$ENV.log"
    local start_time=$(date +%s)
    local timeout=7200 # 2 hours
    while [ -s "${current_dir}/logs/failed-sites-$ENV.txt" ]; do
        local current_time=$(date +%s)
        local elapsed_time=$((current_time - start_time))
        if [ $elapsed_time -ge $timeout ]; then
            echo "Timeout reached without resolving all site failures." >>"${current_dir}/logs/deploy-$ENV.log"
            break
        fi
        if grep -q '\[code\]' "${current_dir}/logs/failed-sites-$ENV.txt"; then
            mapfile -t SITES < <(grep '\[code\]' "${current_dir}/logs/failed-sites-$ENV.txt")
            deploy_code_to_dev "${SITES[@]}"
        else
            mapfile -t SITES < <(grep '\[deploy\]' "${current_dir}/logs/failed-sites-$ENV.txt")
            run_deployment_tasks "${SITES[@]}"
        fi
    done

    echo "Finished retrying failed sites" >>"${current_dir}/logs/deploy-$ENV.log"
}

deploy_code_to_dev "${SITES[@]}"
run_deployment_tasks "${SITES[@]}"

retry_failed_deployments

TIMESTAMP=$(date "+%Y-%m-%d#%H:%M:%S")
LOG_FILENAME="~/release/logs/local-to-$ENV.$TIMESTAMP.log"
mv "${current_dir}/logs/deploy-$ENV.log" $LOG_FILENAME

echo "View log by running: cat $LOG_FILENAME"
