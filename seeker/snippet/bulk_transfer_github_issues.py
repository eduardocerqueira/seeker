#date: 2025-03-19T17:08:42Z
#url: https://api.github.com/gists/c3885230d973c451de3a3b46ea6c1489
#owner: https://api.github.com/users/ParijatDas1991

import subprocess
import requests
import logging
import os
import time
import shlex

# Replace token with your actual value
zenhub_token = "**********"
pipeline_map = {}
failed_transfer_map = {}
failed_move_map = {}


# Generate pipeline map from destination workspace
def get_pipelines(destination_workspace_id, destination_repo_id):
    zenhub_api_url = f"https://your_domain_here/api/p2/workspaces/{destination_workspace_id}/repositories/{destination_repo_id}/board"
    headers = {"X-Authentication-Token": "**********"
    response_data = get_request(zenhub_api_url, headers)
    
    for pipeline in response_data["pipelines"]:
        pipeline_name = pipeline["name"]
        pipeline_id = pipeline["id"]
        pipeline_map[pipeline_name] = pipeline_id


def get_issue_pipeline(destination_repo_id, issue_number, pipeline_name):
    zenhub_api_url = f"https://your_domain_here/api/p1/repositories/{destination_repo_id}/issues/{issue_number}"
    headers = {"X-Authentication-Token": "**********"
    response_data = get_request(zenhub_api_url, headers)
        
    if "pipeline" in response_data:
        pipeline = response_data["pipeline"]
        if pipeline["name"] == pipeline_name:
            return True
    
    print(response_data)
    return False


# Get the deserialized JSON
def get_request(url, headers):
    try:
        response = requests.get(url=url, headers=headers, verify=False, timeout=60)
        data = response.json()
    except Exception as e:
        logging.error(f"Failed to get the deserialized JSON: {str(e)}")

    else:
        if 'error' in data:
            print(data['error']['message'])
            logging.error("Error in data: " + data)
            for detail in data['error']['details']:
                print(detail)
                logging.error("Details of Error in data: " + detail)
        else:
            return data

        
def transfer(issue_number, destination_repo_url):
    
    gh_command = f"gh issue transfer {issue_number} {destination_repo_url}"
    new_issue = 0

    try:
        # Run the command using subprocess
        gh_command_list = shlex.split(gh_command)
        new_issue_url = subprocess.run(gh_command_list, text=True, capture_output=True, check=True)
        new_issue = str(new_issue_url.stdout)
        if new_issue is None or len(new_issue) == 0:
            return 0

        new_issue = int((new_issue.split('/')[-1]).strip())

        time.sleep(2)

    except subprocess.CalledProcessError as e:
        print(f"Failed to transfer issue #{issue_number}: {e}")
        failed_transfer_map[issue_number] = e
        logging.info(f"Failed to transfer issue #{issue_number} error is {e}")

    return new_issue


# Function to get all issues from source ZenHub pipeline and then transfer to destination repo 
def transfer_issues(destination_workspace_id, destination_repo_id, source_workspace_id, source_repo_id, destination_repo_url):
    # Get the ZenHub board data from Source
    
    zenhub_api_url = f"https://your_domain_here/api/p2/workspaces/{source_workspace_id}/repositories/{source_repo_id}/board"
    headers = {"X-Authentication-Token": "**********"
    response_data = get_request(zenhub_api_url, headers)
    
    for pipeline in response_data["pipelines"]:
        current_pipeline = str(pipeline["name"])
        logging.info("Accessing data from " + current_pipeline)
        no_of_issues = len(pipeline["issues"])
        print(f"{current_pipeline} has {no_of_issues} issues")
        if no_of_issues != 0:
            for issues in pipeline["issues"]:
                # run transfer
                new_issue = transfer(issues["issue_number"], destination_repo_url)
                if new_issue is None or new_issue == 0:
                    continue
                old_issue_no = issues["issue_number"]
                if current_pipeline != "New Issues":
                    retry = 0
                    while get_issue_pipeline(destination_repo_id, new_issue, current_pipeline) is False:
                        if retry == 3:
                            print(f"FAILED TRANSFER {old_issue_no} to  {new_issue} and move to {current_pipeline}")
                            logging.info(f"FAILED TRANSFER {old_issue_no} to  {new_issue} and move to {current_pipeline}")
                            failed_transfer_map[new_issue] = current_pipeline
                            break
                        move_issue_pipelines(destination_workspace_id, destination_repo_id, new_issue, pipeline_map[current_pipeline])
                        retry = retry+1
                        time.sleep(2)

                        
# Move an issue to destination pipeline id
def move_issue_pipelines(workspace_id, repo_id, issue_number, destination_pipeline_id):
    # ZenHub API endpoints
    
    zenhub_api_url = f"https://your_domain_here/api/p2/workspaces/{workspace_id}/repositories/{repo_id}/issues/{issue_number}/moves"
    headers = {"X-Authentication-Token": "**********"
    payload_data = {
        "pipeline_id": destination_pipeline_id,
        "position": "top"
    }
    response = requests.post(zenhub_api_url, headers=headers, data=payload_data, verify=False)
    if response.status_code == 200:
        logging.info(f"POST request successful! {response.json}" + str(issue_number) + "moved to" + str(destination_pipeline_id))
    else:
        logging.info(f"POST request failed with {response.json}" + str(issue_number))


def main():
    # Example YOUR_SOURCE_WORKSPACE to YOUR_DESTINATION_WORKSPACE
    source_workspace_id = "YOUR_SOURCE_WORKSPACE_ID"
    source_repo_id = YOUR_SOURCE_REPO_ID
    destination_workspace_id = "YOUR_DESTINATION_WORKSPACE_ID"
    destination_repo_id = YOUR_DESTINATION_REPO_ID
    destination_repo_url = "https://your_domain_here/your_repository_here.git"
    
    log_file_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'BulkTransferIssuesActions.log')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_file_path,
                        filemode='w')

    get_pipelines(destination_workspace_id, destination_repo_id)
    
    print(pipeline_map)
    
    transfer_issues(destination_workspace_id, destination_repo_id, source_workspace_id, source_repo_id, destination_repo_url)

    print(failed_transfer_map)
    for key, value in failed_transfer_map.items():
        logging.info(f"{key} failed transfer with error as {value}")

    print(failed_move_map)
    for key, value in failed_move_map.items():
        logging.info(f"{key} failed move with error as {value}")


if __name__ == "__main__":
    main()_main__":
    main()