#date: 2024-09-18T16:58:56Z
#url: https://api.github.com/gists/cc77d7c98cd4c0c02f40f0e777ec3c0c
#owner: https://api.github.com/users/philwhiteuk

from collections import defaultdict
from datetime import date, datetime, timedelta, UTC
import os
import textwrap

from dotenv import load_dotenv
from google.cloud import artifactregistry_v1 as artifactregistry, logging
import yaml

load_dotenv()  # take environment variables from .env.

project_id = os.getenv('PROJECT_ID')
location = os.getenv('LOCATION')

def get_log(log_date: date = date.today()):
    log_file_path = os.path.join(os.path.dirname(__file__), '../var/log', f'{log_date.isoformat()}.yaml')
    
    if not os.path.exists(log_file_path):
        print(f'Fetching log for {log_date.isoformat()}...')
        logging_client = logging.Client(project=project_id)
        log_query = f'''
            protoPayload.methodName="google.devtools.artifactregistry.v1.ArtifactRegistry.BatchDeleteVersions"
            AND protoPayload.authenticationInfo.principalEmail:"@gcp-sa-artifactregistry.iam.gserviceaccount.com"
            AND protoPayload.request.parent:"projects/{project_id}/locations/{location}" 
            AND protoPayload.request.validateOnly=true
            AND protoPayload.serviceName="artifactregistry.googleapis.com" 
            AND 
                (timestamp > {log_date.isoformat()} AND timestamp <= {(log_date + timedelta(days=1)).isoformat()})
        '''
        with open(log_file_path, 'a') as f:
            for entry in logging_client.list_entries(filter_=log_query):
                f.write('---\n')  # Separate documents with '---'
                yaml.dump(entry.to_api_repr(), f) # Write entry to file

    with open(log_file_path, 'r') as f:
        yaml_docs = yaml.safe_load_all(f)
        return list(yaml_docs)

def get_current_inventory(repository: str, package: str):
    inventory_path = os.path.join(os.path.dirname(__file__), '../var/inventory', f'{date.today().isoformat()}-{repository}-{package}.yaml')

    if not os.path.exists(inventory_path):
        print(f'Fetching most recent inventory for {repository}: {package}...')
        artifactregistry_client = artifactregistry.ArtifactRegistryClient()
        parent = f'projects/{project_id}/locations/{location}/repositories/{repository}/packages/{package}'
        request = artifactregistry.ListVersionsRequest(parent=parent)
        with open(inventory_path, 'a') as f:
            for entry in artifactregistry_client.list_versions(request=request):
                f.write('---\n')  # Separate documents with '---'
                yaml.dump({'name': entry.name, 'create_time': entry.create_time.isoformat(), 'update_time': entry.update_time.isoformat()}, f) # Write entry to file
        
    with open(inventory_path, 'r') as f:
        yaml_docs = yaml.safe_load_all(f)
        return list(yaml_docs)

def main():
    today = date.today()
    print(f'Checking log for {today.isoformat()}...')
    log_entries = get_log(log_date=today)

    versions_marked_for_deletion = set()
    for log_entry in log_entries:
        if log_entry['protoPayload']['request']['names']:
            versions_marked_for_deletion.update(log_entry['protoPayload']['request']['names'])

    versions_by_package = defaultdict(list)
    for tag in versions_marked_for_deletion:
        repository = tag.split('/')[5]
        package = tag.split('/')[7]
        versions_by_package[f'{repository}/{package}'].append(tag)

    now = datetime.now(UTC)
    for package, versions_marked_for_deletion in versions_by_package.items():
        repository = package.split('/')[0]
        package = package.split('/')[1]
        
        all_versions = get_current_inventory(repository=repository, package=package)
        assert len(versions_marked_for_deletion) <= len(all_versions) - 100, f'{repository}/{package} is keeping fewer than the minimum 100 versions!'
        summary = f'''
        Resource:                projects/{project_id}/locations/{location}/repositories/{repository}/package/{package}
        Total files:            {len(all_versions)}
        Marked for deletion:    {len(versions_marked_for_deletion)}
        '''
        print(textwrap.dedent(summary))

        inventory_lookup = dict()
        for item in all_versions:
            inventory_lookup.update([{ item['name'], datetime.fromisoformat(item['update_time']) }])

        tag_counter = 0
        for tag in versions_marked_for_deletion:
            if tag in inventory_lookup:
                timedelta = now - inventory_lookup[tag]
                assert timedelta.days >= 5, f'Version {tag} is newer than 5 days!'
                print(f'- ✅ {tag.split('/')[-1]} {timedelta.days} days old')
                tag_counter += 1
        
