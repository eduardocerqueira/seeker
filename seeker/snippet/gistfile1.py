#date: 2025-11-14T16:51:01Z
#url: https://api.github.com/gists/a0313f814992016071cb20d61a528669
#owner: https://api.github.com/users/theosanderson

import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# Step 1: Create the 'images' folder if it doesn't exist
os.makedirs("images", exist_ok=True)

# Step 2: Define pathogens to process
pathogens = ['mpox', 'ebola-zaire', 'ebola-sudan', 'west-nile', 'cchf', 'hmpv', 'rsv-a', 'rsv-b']

# Step 3: Get date range from PPX launch (August 2024) until now
start_date = datetime(2024, 8, 1)
end_date = datetime.today()

# Calculate number of months
months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

# Function to fetch and count sequences for a given pathogen
def fetch_monthly_counts(pathogen, num_months):
    print(f"Fetching data for {pathogen}...")
    monthly_counts = []
    ncbi_monthly_counts = []
    
    for i in range(num_months):
        # Get first and last day of the month
        current_month = start_date.replace(day=1) + timedelta(days=32*i)
        month_start = current_month.replace(day=1)
        month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)

        # API request parameters for all submitters
        api_url = f"https://lapis.pathoplexus.org/{pathogen}/sample/aggregated"
        params = {
            "earliestReleaseDateFrom": month_start.strftime("%Y-%m-%d"),
            "earliestReleaseDateTo": month_end.strftime("%Y-%m-%d"),
            "versionStatus": "LATEST_VERSION",
        }

        # Make the API request for total counts
        response = requests.get(api_url, params=params)

        if response.status_code == 200:
            data = response.json()
            count = data['data'][0]['count'] if data.get('data') else 0
            monthly_counts.append(count)
        else:
            print(f"Error fetching total data for {pathogen} in {month_start.strftime('%Y-%m')}: {response.status_code}")
            monthly_counts.append(0)

        # Make calls for NCBI submitters
        params["submitter"] = "insdc_ingest_user"
        response = requests.get(api_url, params=params)

        if response.status_code == 200:
            data = response.json()
            count = data['data'][0]['count'] if data.get('data') else 0
            ncbi_monthly_counts.append(count)
        else:
            print(f"Error fetching NCBI data for {pathogen} in {month_start.strftime('%Y-%m')}: {response.status_code}")
            ncbi_monthly_counts.append(0)
    
    return monthly_counts, ncbi_monthly_counts

# Step 4: Collect data for all pathogens
all_direct_counts = {}
month_labels = []

# Get month labels from August 2024 to now
for i in range(months_diff):
    current_month = start_date.replace(day=1) + timedelta(days=32*i)
    month_start = current_month.replace(day=1)
    month_labels.append(month_start.strftime("%Y-%m"))

# Initialize combined counts
combined_direct_counts = [0] * months_diff

# Process each pathogen and aggregate
for pathogen in pathogens:
    print(f"Processing: {pathogen}")
    total_counts, ncbi_counts = fetch_monthly_counts(pathogen, months_diff)
    
    # Calculate direct counts for this pathogen
    direct_counts = [total - ncbi for total, ncbi in zip(total_counts, ncbi_counts)]
    all_direct_counts[pathogen] = direct_counts
    
    # Add to combined total
    combined_direct_counts = [combined + direct for combined, direct in zip(combined_direct_counts, direct_counts)]

# Step 5: Calculate cumulative counts
cumulative_counts = []
running_total = 0
for count in combined_direct_counts:
    running_total += count
    cumulative_counts.append(running_total)

# Step 6: Create the combined cumulative plot
plt.figure(figsize=(6, 4))

# Create a line plot with markers
plt.plot(month_labels, cumulative_counts, marker='o', linewidth=2, 
         markersize=6, color='darkorange', label='Cumulative Direct Submissions')

# Fill area under the line
plt.fill_between(range(len(month_labels)), cumulative_counts, alpha=0.3, color='darkorange')

# Title and labels
plt.title("Cumulative Direct Submissions (Aug 2024 - Present)", fontsize=11, fontweight='bold')
plt.xlabel("Month", fontsize=9)
plt.ylabel("Cumulative Sequences", fontsize=9)
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.legend(fontsize=8)

# Layout adjustment
plt.tight_layout()

# Save the chart with high DPI
image_path = "images/combined_direct_submissions.png"
plt.savefig(image_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved combined plot: {image_path}")
plt.close()

# Step 7: Print summary statistics
print("\n" + "="*60)
print("SUMMARY: Cumulative Direct Submissions by Month")
print("="*60)
for month, monthly, cumulative in zip(month_labels, combined_direct_counts, cumulative_counts):
    print(f"{month}: +{monthly:,} sequences (cumulative: {cumulative:,})")

total_direct = sum(combined_direct_counts)
print(f"\nTotal direct submissions: {total_direct:,}")
print(f"Average per month: {total_direct/months_diff:.1f}")

# Calculate trend
if combined_direct_counts[0] > 0 and len(combined_direct_counts) > 1:
    # Compare first and last month
    growth = ((combined_direct_counts[-1] - combined_direct_counts[0]) / combined_direct_counts[0]) * 100
    print(f"Growth from first to last month: {growth:+.1f}%")

print("\n✅ Analysis complete!")