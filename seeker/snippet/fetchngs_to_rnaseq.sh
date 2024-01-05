#date: 2024-01-05T17:01:41Z
#url: https://api.github.com/gists/53de1d57ecfaa1aefc8d0cbae8769233
#owner: https://api.github.com/users/pinin4fjords

#!/bin/bash

set -e
set -o pipefail

# --------------------------
# User-set parameters
# --------------------------

# IDs and workspace details or default to 'scidev/testing'
upstream_workflow_id="$1"
child_workflow="$2"
workspace="${3:-scidev/testing}"
genome_fasta_path="$4"
gtf_path="$5"

echo "Setting up workflows with:"
echo "Upstream ID: $upstream_workflow_id"
echo "Child Workflow: $child_workflow"
echo "Workspace: $workspace"

# --------------------------------------------------------
# Core Function: build_params
# Purpose: Extract & assemble parameters from a manifest.
# Note: Input/Output interfaces consistent, but JSON 
# generation is customizable.
# --------------------------------------------------------

build_params() {
    local manifest_json="$1"
    local child_outdir="$2"
    local genome_fasta_path="$3"
    local gtf_path="$4"
    
    # Extract the sample sheet path
    local sample_sheet=$(jq -r '.io_domain.output_subdomain[] | 
        select(.uri.filename | contains("samplesheet.csv")).uri.uri' $manifest_json)
   
    echo "Building params" 1>&2 
    # Build parameters JSON
    jq -n --arg g "$gtf_path" --arg f "$genome_fasta_path" --arg o "$child_outdir" \
        --arg i "$sample_sheet" '{fasta:$f, gtf:$g, outdir:$o, 
        input: $i}'
}


# --------------------------------------------------------
# Utilty function: Fetch a remote file
# --------------------------------------------------------

fetch_remote_file() {

  local remote_path="$1"
  local local_dest=$(basename $remote_path)

  # Check if the remote path is an S3 path
  if [[ "$remote_path" == s3://* ]]; then
    # Use AWS CLI to copy the file from S3
    aws s3 cp "$remote_path" "$local_dest" 1>&2
    if [ $? -ne 0 ]; then
      echo "Error fetching from S3" >&2
      return 1
    fi
  else
    # Use curl to fetch the file
    curl -o "$local_dest" "$remote_path"
    if [ $? -ne 0 ]; then
      echo "Error fetching using curl" >&2
      return 1
    fi
  fi
}

# --------------------------------------------------------
# Utility Function: launch_child_workflow
# Purpose: Auto-launch of child workflows post parents.
# Note: This function remains static across use-cases.
# --------------------------------------------------------

launch_child_workflow() {
    local run_id="$1"
    local child_workflow="$2"
    local workspace="$3"
    local genome_fasta_path="$3"
    local gtf_path="$4"
    
    # Extract the outdir from upstream workflow
    local outdir=$(tw -o json runs view --workspace=$workspace --id=$run_id --params | 
        jq -r '.parameters.outdir')
    local child_outdir="$outdir/$(basename $child_workflow)"

    # Copy manifest.json from output directory
    fetch_remote_file ${outdir}/manifest.json

    # Build & modify parameters, then save to a JSON file
    build_params manifest.json $child_outdir $genome_fasta_path $gtf_path | \
        > params.json
   
    if [ $? -ne 0 ]; then
        echo "Params did not build successfully" >&2; return 1
    fi

    # Launch the child workflow with created parameters
    echo "Launching $child_workflow in $workspace"
    tw launch --params-file=params.json $child_workflow -w $workspace 
}

# Main Execution

# If previous workflow exited without errors, launch the child
if [ "$NXF_EXIT_STATUS" -eq 0 ]; then
    launch_child_workflow "$upstream_workflow_id" "$child_workflow" "$workspace" "$genome_fasta_path" "$gtf_path"
fi