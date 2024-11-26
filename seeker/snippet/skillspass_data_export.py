#date: 2024-11-26T17:02:31Z
#url: https://api.github.com/gists/af6c97bcb8546c1f391ccc1c76afbe47
#owner: https://api.github.com/users/donalmacanri

import requests
import time
from datetime import datetime, timezone, timedelta
import os

class SkillsPassExporter:
    def __init__(self, api_key, api_token, api_secret, base_url="https: "**********":
        self.base_url = base_url
        self.headers = {
            "Authorization": "**********"
        }

    def create_export_job(self, start_date, end_date, export_type, format="CSV"):
        """Create a new export data job"""
        url = f"{self.base_url}/tp-api/exports"

        payload = {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "exportType": export_type,
            "format": format
        }

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()["job"]["id"]

    def get_export_job_status(self, job_id):
        """Check the status of an export job"""
        url = f"{self.base_url}/tp-api/exports"
        params = {"id": job_id}

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def download_export(self, download_url, output_path):
        """Download the exported file"""
        response = requests.get(download_url)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

    def wait_for_export_completion(self, job_id, timeout_minutes=5, check_interval_seconds=10):
        """Wait for the export job to complete"""
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while True:
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Export job didn't complete within {timeout_minutes} minutes")

            job_status = self.get_export_job_status(job_id)
            status = job_status["job"]["status"]

            if status == "COMPLETED":
                return job_status
            elif status == "FAILED":
                errors = job_status.get("errors", [])
                raise Exception(f"Export job failed: {errors}")

            time.sleep(check_interval_seconds)

def main():
    # Replace with your actual credentials
    api_key = "your_api_key"
    api_token = "**********"
    api_secret = "**********"

    exporter = "**********"

    # Set up time range for the export
    end_date = datetime.now(timezone.utc)
    last_export_file = "last_export_time.txt"

    # Check for last export time, default to 2017-01-01 if file doesn't exist
    if os.path.exists(last_export_file):
        with open(last_export_file, 'r') as f:
            start_date = datetime.fromisoformat(f.read().strip())
    else:
        start_date = datetime(2017, 1, 1, tzinfo=timezone.utc)

    # Create export directory if it doesn't exist
    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)

    # List of export types to process
    export_types = [
        "ELEARNING_ENROLLMENTS",
        "LEARNING_ACTIVITIES",
        "NETWORKS",
        "USERS"
    ]

    for export_type in export_types:
        try:
            print(f"\nProcessing export for {export_type}")

            # Create export job
            job_id = exporter.create_export_job(
                start_date=start_date,
                end_date=end_date,
                export_type=export_type
            )
            print(f"Created export job: {job_id}")

            # Wait for job completion
            job_result = exporter.wait_for_export_completion(job_id)

            # Download the file if available
            download_url = job_result["exportData"]["downloadUrl"]
            if download_url:
                # Create filename with timestamp
                timestamp = end_date.strftime("%Y%m%d_%H%M%S")
                filename = f"{export_type.lower()}_{timestamp}.csv"
                output_path = os.path.join(export_dir, filename)

                # Download the file
                exporter.download_export(download_url, output_path)
                print(f"Downloaded export to: {output_path}")
            else:
                print("No data available for download")

        except Exception as e:
            print(f"Error processing {export_type}: {str(e)}")

    # Write the current end_date as the last export time
    with open(last_export_file, 'w') as f:
        f.write(end_date.isoformat())

if __name__ == "__main__":
    main()
 as the last export time
    with open(last_export_file, 'w') as f:
        f.write(end_date.isoformat())

if __name__ == "__main__":
    main()
