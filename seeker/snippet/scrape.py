#date: 2025-05-12T16:52:52Z
#url: https://api.github.com/gists/9affd76a765e56286fb2cadca3f0d730
#owner: https://api.github.com/users/AngeloGiacco

import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import sys
import argparse
import logging

# --- Configuration ---
# IMPORTANT: Replace these placeholders with your actual API key and Agent ID from .env
ELEVENLABS_API_KEY = "YOUR_ELEVENLABS_API_KEY"
AGENT_ID = "YOUR_AGENT_ID"

# Base URLs to start crawling from
BASE_URLS = [
    "https://www.abc.xyz/def/ghi/"
]

# API Endpoints
CREATE_KB_URL_ENDPOINT = "https://api.elevenlabs.io/v1/convai/knowledge-base/url"
UPDATE_AGENT_ENDPOINT = f"https://api.elevenlabs.io/v1/convai/agents/{AGENT_ID}"

# --- Crawler Settings ---
MAX_PAGES_PER_DOMAIN = 100 # Limit the number of pages to crawl per domain to prevent excessive requests
REQUEST_DELAY = 0.5 # Delay between HTTP requests to be polite
API_CALL_DELAY = 1.1 # Delay between ElevenLabs API calls to avoid rate limits
REQUEST_TIMEOUT = 15 # Timeout for HTTP requests in seconds
USER_AGENT = "KnowledgeBaseCrawler/1.0 (Python Script; +http://example.com/crawlerinfo)" # Optional: Set a user agent

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stderr)
# Suppress noisy logs from libraries
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# --- Helper Functions ---

def is_valid_url(url):
    """Checks if a URL is valid and has an http/https scheme."""
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except ValueError:
        return False

def crawl_site(start_url, visited_urls, all_urls_on_domain, max_pages):
    """
    Recursively crawls a website starting from start_url, staying within the same
    domain and initial path structure.

    Args:
        start_url (str): The URL to start crawling from.
        visited_urls (set): A set of URLs already visited during this crawl session.
        all_urls_on_domain (set): A set to store all valid URLs found for this domain.
        max_pages (int): The maximum number of pages to crawl for this domain.
    """
    if start_url in visited_urls or len(all_urls_on_domain) >= max_pages:
        return

    logging.info(f"Crawling: {start_url} (Found: {len(all_urls_on_domain)}/{max_pages})")
    visited_urls.add(start_url)

    headers = {'User-Agent': USER_AGENT}
    try:
        time.sleep(REQUEST_DELAY) # Be polite
        response = requests.get(start_url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Ensure content type is HTML before parsing
        if 'text/html' not in response.headers.get('Content-Type', ''):
            logging.warning(f"Skipping non-HTML content at: {start_url}")
            return

        soup = BeautifulSoup(response.content, 'html.parser')
        base_parsed = urlparse(start_url)
        base_domain = base_parsed.netloc
        # Define the allowed path prefix (e.g., '/fi/support/solutions/')
        # This helps stay within the relevant support section
        allowed_path_prefix = '/'.join(base_parsed.path.split('/')[:4]) # Adjust depth as needed

        # Add the current valid URL
        if start_url not in all_urls_on_domain:
             all_urls_on_domain.add(start_url)

        links_found = 0
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Construct absolute URL
            absolute_url = urljoin(start_url, href)

            # Validate and filter the URL
            if not is_valid_url(absolute_url):
                continue

            parsed_url = urlparse(absolute_url)

            # 1. Check if it's the same domain
            # 2. Check if it stays within the allowed path prefix
            # 3. Check if it hasn't been visited
            # 4. Check if we haven't hit the max pages limit
            if (parsed_url.netloc == base_domain and
                parsed_url.path.startswith(allowed_path_prefix) and
                absolute_url not in visited_urls and
                len(all_urls_on_domain) < max_pages):

                links_found += 1
                # Recursively crawl the new link
                crawl_site(absolute_url, visited_urls, all_urls_on_domain, max_pages)

        # print(f"  Found {links_found} new potential links on {start_url}")


    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {start_url}: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Error processing {start_url}: {e}", exc_info=True)


def create_kb_document(url, api_key):
    """
    Calls the ElevenLabs API to create a knowledge base document from a URL.

    Args:
        url (str): The URL to create the document from.
        api_key (str): Your ElevenLabs API key.

    Returns:
        dict: A dictionary containing 'id' and 'name' of the created document,
              or None if creation failed.
    """
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    payload = {
        "url": url
        # Optionally add a name: "name": f"Doc from {urlparse(url).path}"
    }
    logging.info(f"Attempting to create KB document for: {url}")
    try:
        time.sleep(API_CALL_DELAY) # Avoid rate limits
        response = requests.post(CREATE_KB_URL_ENDPOINT, headers=headers, json=payload, timeout=REQUEST_TIMEOUT * 2) # Longer timeout for API

        if response.status_code == 200:
            data = response.json()
            logging.info(f"    Success! KB Document ID: {data.get('id')}, Name: {data.get('name')}")
            return {"id": data.get("id"), "name": data.get("name")}
        else:
            logging.error(f"    Error creating KB document for {url}. Status: {response.status_code}, Response: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"    Network error creating KB document for {url}: {e}")
        return None
    except json.JSONDecodeError as e:
         logging.error(f"    Error decoding API response for {url}: {e}")
         return None
    except Exception as e:
        logging.error(f"    Unexpected error creating KB document for {url}: {e}")
        return None


def update_agent_knowledge_base(agent_id, created_docs, api_key):
    """
    Updates the specified agent's knowledge base with the provided document IDs.

    Args:
        agent_id (str): The ID of the agent to update.
        created_docs (list): A list of dictionaries, each containing 'id' and 'name'
                               of a successfully created KB document.
        api_key (str): Your ElevenLabs API key.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    if not created_docs:
        logging.info("No documents were successfully created. Skipping agent update.")
        return False

    headers = {
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }

    # Prepare the knowledge base list in the format expected by the API
    knowledge_base_payload = [
        {"id": doc['id'], "name": doc.get('name', f"Doc {doc['id']}"), "type": "url"} # Assuming type 'url' is correct
        for doc in created_docs if doc and 'id' in doc
    ]

    if not knowledge_base_payload:
        logging.info("No valid document IDs to add. Skipping agent update.")
        return False

    payload = {
        "conversation_config": {
            "agent": {
                "prompt": {
                    "knowledge_base": knowledge_base_payload
                }
            }
        }
    }

    logging.info(f"\nAttempting to update Agent {agent_id} with {len(knowledge_base_payload)} knowledge base documents...")

    try:
        time.sleep(API_CALL_DELAY)
        response = requests.patch(UPDATE_AGENT_ENDPOINT, headers=headers, json=payload, timeout=REQUEST_TIMEOUT * 2)

        if response.status_code == 200:
            logging.info("  Agent update successful!")
            return True
        else:
            logging.error(f"  Error updating agent {agent_id}. Status: {response.status_code}, Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        logging.error(f"  Network error updating agent {agent_id}: {e}")
        return False
    except Exception as e:
        logging.error(f"  Unexpected error updating agent {agent_id}: {e}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    if ELEVENLABS_API_KEY == "YOUR_ELEVENLABS_API_KEY" or AGENT_ID == "YOUR_AGENT_ID":
        logging.critical("ERROR: Please replace 'YOUR_ELEVENLABS_API_KEY' and 'YOUR_AGENT_ID' placeholders in the script.")
        sys.exit(1)

    logging.info("Starting Knowledge Base Update Process...")

    all_urls_found = set()
    for base_url in BASE_URLS:
        logging.info(f"\nCrawling domain starting from: {base_url}")
        visited_on_this_domain = set()
        urls_for_this_domain = set()
        try:
            crawl_site(base_url, visited_on_this_domain, urls_for_this_domain, MAX_PAGES_PER_DOMAIN)
            logging.info(f"Finished crawling for {urlparse(base_url).netloc}. Found {len(urls_for_this_domain)} unique pages.")
            all_urls_found.update(urls_for_this_domain)
        except Exception as e:
            logging.error(f"An error occurred during the crawl for {base_url}: {e}", exc_info=True)


    logging.info(f"\nTotal unique URLs found across all domains: {len(all_urls_found)}")

    if not all_urls_found:
        logging.info("No URLs found to process. Exiting.")
        sys.exit(0)

    logging.info("\nCreating Knowledge Base documents via ElevenLabs API...")
    created_kb_docs = []
    for i, url in enumerate(all_urls_found):
        logging.info(f"Processing URL {i+1}/{len(all_urls_found)}...")
        doc_info = create_kb_document(url, ELEVENLABS_API_KEY)
        if doc_info:
            created_kb_docs.append(doc_info)

    logging.info(f"\nSuccessfully created {len(created_kb_docs)} knowledge base documents.")

    # Update the agent
    update_successful = update_agent_knowledge_base(AGENT_ID, created_kb_docs, ELEVENLABS_API_KEY)

    if update_successful:
        logging.info("\nProcess finished successfully!")
    else:
        logging.error("\nProcess finished with errors during agent update.")
        sys.exit(1)
