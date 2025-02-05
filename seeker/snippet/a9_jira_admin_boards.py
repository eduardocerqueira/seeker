#date: 2025-02-05T17:07:23Z
#url: https://api.github.com/gists/de117065df726b8efc72e388babafefa
#owner: https://api.github.com/users/wjkennedy

#!/usr/bin/env python3
import configparser
import os
import sys
import logging
from jira import JIRA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_file):
    """
    Load the configuration from the given file.
    Expects a [jira] section with keys: "**********"
    """
    if not os.path.exists(config_file):
        logger.error(f"Configuration file '{config_file}' not found.")
        sys.exit(1)
    config = configparser.ConfigParser()
    config.read(config_file)
    if 'jira' not in config:
        logger.error("Section [jira] not found in the configuration file.")
        sys.exit(1)
    return config['jira']

def get_all_boards(jira_client):
    """
    Retrieve all boards using pagination.
    """
    boards = []
    start_at = 0
    max_results = 50
    while True:
        result = jira_client.boards(startAt=start_at, maxResults=max_results)
        boards.extend(result.get('values', []))
        total = result.get('total', 0)
        if start_at + max_results >= total:
            break
        start_at += max_results
    return boards

def update_filter_owner(jira_client, filter_id, new_owner, server_url):
    """
    Update the owner of the filter (via Jira REST API PUT call).
    """
    update_url = f"{server_url}/rest/api/2/filter/{filter_id}"
    payload = {"owner": new_owner}
    response = jira_client._session.put(update_url, json=payload)
    return response

def main():
    # Determine the path to the configuration file
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(script_dir, "config", "config.properties")
    config = load_config(config_file)
    
    # Get Jira connection parameters
    server = config.get("server")
    username = config.get("username")
    password = "**********"
    
    logger.info("Connecting to Jira...")
    try:
        jira_client = "**********"=server, basic_auth=(username, password))
    except Exception as e:
        logger.error(f"Failed to connect to Jira: {e}")
        sys.exit(1)
    
    logger.info("Fetching all boards...")
    boards = get_all_boards(jira_client)
    logger.info(f"Found {len(boards)} boards.")
    
    boards_updated = 0
    for board in boards:
        board_id = board.get('id')
        board_name = board.get('name')
        try:
            config_data = jira_client.board_config(board_id)
        except Exception as e:
            logger.error(f"Failed to get configuration for board '{board_name}' (ID: {board_id}): {e}")
            continue

        board_filter = config_data.get('filter')
        if not board_filter:
            logger.warning(f"Board '{board_name}' (ID: {board_id}) does not have an associated filter.")
            continue

        filter_id = board_filter.get('id')
        filter_owner = board_filter.get('owner')
        # In Jira Data Center the filter owner is typically returned with a 'name' key.
        if filter_owner and filter_owner.get('name'):
            logger.info(f"Board '{board_name}' (ID: {board_id}) already has an admin: {filter_owner.get('name')}.")
            continue
        else:
            logger.info(f"Board '{board_name}' (ID: {board_id}) has no admin. Attempting to update filter {filter_id}.")
            response = update_filter_owner(jira_client, filter_id, username, server)
            if response.status_code == 200:
                logger.info(f"Successfully updated board '{board_name}' (ID: {board_id}) to have admin '{username}'.")
                boards_updated += 1
            else:
                logger.error(
                    f"Failed to update board '{board_name}' (ID: {board_id}). "
                    f"Response: {response.status_code} {response.text}"
                )
    
    logger.info(f"Total boards updated: {boards_updated}")

if __name__ == "__main__":
    main()
_ == "__main__":
    main()
