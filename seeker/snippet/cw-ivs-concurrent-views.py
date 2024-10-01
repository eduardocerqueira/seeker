#date: 2024-10-01T17:03:55Z
#url: https://api.github.com/gists/91a220a7a7961ce5a15c9f338f39e034
#owner: https://api.github.com/users/mattwildes

import boto3
import json
from datetime import datetime, timedelta

def create_ivs_dashboard():
    # Create clients for IVS and CloudWatch
    ivs = boto3.client('ivs')
    cloudwatch = boto3.client('cloudwatch')

    # Get list of all IVS channels
    channels = []
    response = ivs.list_channels()
    channels.extend(response['channels'])
 "**********"  "**********"  "**********"  "**********"  "**********"w "**********"h "**********"i "**********"l "**********"e "**********"  "**********"' "**********"n "**********"e "**********"x "**********"t "**********"T "**********"o "**********"k "**********"e "**********"n "**********"' "**********"  "**********"i "**********"n "**********"  "**********"r "**********"e "**********"s "**********"p "**********"o "**********"n "**********"s "**********"e "**********": "**********"
        response = "**********"=response['nextToken'])
        channels.extend(response['channels'])

    # Prepare dashboard widgets
    widgets = []
    for channel in channels:
        channel_name = channel['name']
        channel_arn = channel['arn']
        channel_id = channel_arn.split(':')[-1].split('/')[-1]  # Extract channel ID from ARN and remove "channel/" prefix
        
        widget = {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 24,
            "height": 12,
            "properties": {
                "metrics": [
                    ["AWS/IVS", "ConcurrentViews", "Channel", channel_id]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": ivs.meta.region_name,  # Use the region from the IVS client
                "title": f"Avg Viewer Count (30s) - {channel_name}",
                "period": 60,  # Set to 60 seconds
                "stat": "Maximum",  
                "yAxis": {
                    "left": {
                        "min": 0,
                        "label": "Viewers"
                    }
                },
                "legend": {
                    "position": "bottom"
                },
                "liveData": True,
                "timeRange": {
                    "start": "-PT3H",  # Show last 3 hours
                    "end": "PT0H"
                }
            }
        }
        widgets.append(widget)

    # Create dashboard
    dashboard_name = "IVS-Avg-Viewer-Count-Dashboard"
    dashboard_body = {
        "widgets": widgets
    }

    try:
        response = cloudwatch.put_dashboard(
            DashboardName=dashboard_name,
            DashboardBody=json.dumps(dashboard_body)
        )
        print(f"Dashboard '{dashboard_name}' created successfully.")
    except Exception as e:
        print(f"Error creating dashboard: {str(e)}")

if __name__ == "__main__":
    create_ivs_dashboard()
