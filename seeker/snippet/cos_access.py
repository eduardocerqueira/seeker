#date: 2026-01-12T17:11:24Z
#url: https://api.github.com/gists/b6b608c28e615a201bf22798649c8715
#owner: https://api.github.com/users/greyhoundforty

#!/usr/bin/env python3
"""
IBM Cloud Object Storage - HMAC Credential Bucket Access Tester

This script tests HMAC credentials to determine which buckets you can actually
read from and download files. Useful when credentials may only work for certain
buckets due to regional restrictions or permissions.

Usage:
    python test_bucket_access.py
"""

import os 
import ibm_boto3
from ibm_botocore.client import Config
from ibm_botocore.exceptions import ClientError
import sys


def format_size(size_bytes):
    """
    Convert bytes to a human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 GB", "256 MB", "3.2 KB")
    """
    if size_bytes == 0:
        return "0 B (empty)"
    elif size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


 "**********"d "**********"e "**********"f "**********"  "**********"c "**********"r "**********"e "**********"a "**********"t "**********"e "**********"_ "**********"c "**********"o "**********"s "**********"_ "**********"c "**********"l "**********"i "**********"e "**********"n "**********"t "**********"( "**********"e "**********"n "**********"d "**********"p "**********"o "**********"i "**********"n "**********"t "**********", "**********"  "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"k "**********"e "**********"y "**********", "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"_ "**********"k "**********"e "**********"y "**********") "**********": "**********"
    """
    Create an IBM Cloud Object Storage client using HMAC credentials.
    
    HMAC (Hash-based Message Authentication Code) credentials are an alternative
    to API key authentication. They consist of an access key and secret key pair.
    
    Args:
        endpoint: The COS endpoint URL (e.g., "s3.us-south.cloud-object-storage.appdomain.cloud")
        access_key: "**********"
        secret_key: "**********"
    
    Returns:
        An IBM COS client object configured with the provided credentials
    """
    endpoint = os.getenv('COS_ENDPOINT')
    access_key = "**********"
    secret_key = "**********"
    # Ensure the endpoint has the https:// protocol
    if not endpoint.startswith("http"):
        endpoint = f"https://{endpoint}"
    
    # Create the client with S3-compatible signature version 4
    # This is required for IBM Cloud Object Storage HMAC authentication
    client = ibm_boto3.client(
        's3',
        ibm_api_key_id=None,  # We're using HMAC, not API key
        ibm_service_instance_id=None,  # Not needed for HMAC auth
        config=Config(signature_version='s3v4'),  # S3 signature version 4
        endpoint_url=endpoint,
        aws_access_key_id= "**********"
        aws_secret_access_key= "**********"
    )
    
    return client


def list_all_buckets(client):
    """
    List all buckets that are visible with the current credentials.
    
    Note: Just because a bucket is visible doesn't mean you can read from it.
    We'll test read access separately for each bucket.
    
    Args:
        client: IBM COS client object
    
    Returns:
        List of bucket names (strings)
    """
    try:
        # Call the list_buckets API
        response = client.list_buckets()
        
        # Extract bucket names from the response
        buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
        
        return buckets
    
    except ClientError as e:
        # If we can't even list buckets, the credentials are likely invalid
        error_code = e.response['Error']['Code']
        print(f"❌ Error listing buckets: {error_code}")
        print(f"   Message: {e.response['Error']['Message']}")
        return []
    
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return []


def test_bucket_read_access(client, bucket_name):
    """
    Test if we can actually read from a specific bucket.
    
    This attempts to list objects in the bucket. If this succeeds, we have
    read access. If it fails, we'll get an error code that tells us why.
    
    Common error codes:
    - AccessDenied: You don't have permission (often means wrong region)
    - NoSuchBucket: The bucket doesn't exist
    - 403: Forbidden - bucket exists but you can't access it
    
    Args:
        client: IBM COS client object
        bucket_name: Name of the bucket to test
    
    Returns:
        Tuple of (success: bool, message: str, object_count: int)
    """
    try:
        # Try to list objects in the bucket
        # We only request 1 object to make this test fast
        response = client.list_objects_v2(
            Bucket=bucket_name,
            MaxKeys=1  # Just need to know if we CAN list, not list everything
        )
        
        # If we got here without an exception, we have read access!
        object_count = response.get('KeyCount', 0)
        
        # Try to get a more accurate total count
        if 'Contents' in response:
            # There are objects in the bucket
            return True, "✅ READ ACCESS CONFIRMED", object_count
        else:
            # Bucket is empty, but we can read from it
            return True, "✅ READ ACCESS CONFIRMED (empty bucket)", 0
    
    except ClientError as e:
        # We got an error - parse it to understand what went wrong
        error_code = e.response['Error']['Code']
        error_msg = e.response['Error']['Message']
        
        if error_code == 'NoSuchBucket':
            return False, "❌ Bucket does not exist", 0
        
        elif error_code == 'AccessDenied':
            # This usually means the bucket is in a different region
            # or your credentials don't have permission
            return False, "❌ ACCESS DENIED (likely wrong region or insufficient permissions)", 0
        
        elif error_code == '403':
            return False, "❌ FORBIDDEN (bucket exists but credentials lack access)", 0
        
        else:
            return False, f"❌ ERROR: {error_code} - {error_msg}", 0
    
    except Exception as e:
        return False, f"❌ UNEXPECTED ERROR: {str(e)}", 0


def get_bucket_stats(client, bucket_name):
    """
    Get the total number of objects and total size for a bucket.
    
    This function iterates through ALL objects in the bucket using pagination
    to count them and sum up their sizes. This can take a while for large buckets
    but gives you accurate information about what you're about to download.
    
    Args:
        client: IBM COS client object
        bucket_name: Name of the bucket
    
    Returns:
        Tuple of (total_objects: int, total_size_bytes: int)
    """
    try:
        print(f"      📊 Counting objects and calculating size...")
        
        # Use a paginator to handle buckets with many objects
        # This automatically handles the continuation tokens for us
        paginator = client.get_paginator('list_objects_v2')
        
        total_objects = 0
        total_size = 0
        
        # Iterate through all pages
        for page in paginator.paginate(Bucket=bucket_name):
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Skip directory markers (objects ending with /)
                    if not obj['Key'].endswith('/'):
                        total_objects += 1
                        total_size += obj['Size']
        
        return total_objects, total_size
    
    except ClientError as e:
        # If we can't get stats, just return zeros
        # The bucket is still accessible, we just can't count everything
        return 0, 0
    
    except Exception as e:
        return 0, 0


def get_bucket_location(client, bucket_name):
    """
    Try to determine which region a bucket is in.
    
    This helps identify why you might not have access - the bucket could be
    in a different region than your credentials are configured for.
    
    Args:
        client: IBM COS client object
        bucket_name: Name of the bucket
    
    Returns:
        String with region name, or "unknown" if we can't determine it
    """
    try:
        response = client.get_bucket_location(Bucket=bucket_name)
        # LocationConstraint is the region
        region = response.get('LocationConstraint', 'us-east-1')
        return region if region else 'us-east-1'
    
    except:
        # If we can't get the location, it's often because we don't have access
        return "unknown (no access)"


def main():
    """
    Main function - prompts for credentials and tests bucket access.
    """
    print("=" * 70)
    print("IBM Cloud Object Storage - Bucket Access Tester")
    print("=" * 70)
    print()
    print("This script will test your HMAC credentials to see which buckets")
    print("you can actually read from and download files.")
    print()
    
    # Prompt for credentials
    print("Enter your IBM COS credentials:")
    print()
    
    endpoint = input("Endpoint (e.g., s3.us-south.cloud-object-storage.appdomain.cloud): ").strip()
    access_key = input("HMAC Access Key: "**********"
    secret_key = input("HMAC Secret Key: "**********"
    
    print()
    print("=" * 70)
    print("Testing credentials...")
    print("=" * 70)
    print()
    
    # Validate inputs
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"e "**********"n "**********"d "**********"p "**********"o "**********"i "**********"n "**********"t "**********"  "**********"o "**********"r "**********"  "**********"n "**********"o "**********"t "**********"  "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"k "**********"e "**********"y "**********"  "**********"o "**********"r "**********"  "**********"n "**********"o "**********"t "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"_ "**********"k "**********"e "**********"y "**********": "**********"
        print("❌ Error: All fields are required!")
        sys.exit(1)
    
    # Create the COS client
    try:
        client = "**********"
        print(f"✅ Created COS client for endpoint: {endpoint}")
    except Exception as e:
        print(f"❌ Failed to create COS client: {str(e)}")
        sys.exit(1)
    
    print()
    
    # Step 1: List all buckets
    print("STEP 1: Listing all visible buckets...")
    print("-" * 70)
    
    buckets = list_all_buckets(client)
    
    if not buckets:
        print("❌ No buckets found or unable to list buckets.")
        print()
        print("This could mean:")
        print("  - Your credentials are invalid")
        print("  - Your credentials don't have any bucket access")
        print("  - There's a network connectivity issue")
        sys.exit(1)
    
    print(f"✅ Found {len(buckets)} bucket(s)")
    print()
    
    # Step 2: Test read access for each bucket
    print("STEP 2: Testing read access for each bucket...")
    print("-" * 70)
    print()
    
    # Track results
    accessible_buckets = []
    inaccessible_buckets = []
    
    for i, bucket in enumerate(buckets, 1):
        print(f"[{i}/{len(buckets)}] Testing bucket: {bucket}")
        
        # Test if we can read from this bucket
        can_read, message, obj_count = test_bucket_read_access(client, bucket)
        
        print(f"      {message}")
        
        if can_read:
            # We have access! Try to get the region
            region = get_bucket_location(client, bucket)
            print(f"      📍 Region: {region}")
            
            # Get detailed statistics (object count and total size)
            total_objects, total_size = get_bucket_stats(client, bucket)
            
            print(f"      📦 Objects: {total_objects:,}")
            print(f"      💾 Total Size: {format_size(total_size)}")
            
            accessible_buckets.append({
                'name': bucket,
                'region': region,
                'has_objects': total_objects > 0,
                'object_count': total_objects,
                'total_size': total_size
            })
        else:
            inaccessible_buckets.append(bucket)
        
        print()
    
    # Step 3: Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Total buckets found:     {len(buckets)}")
    print(f"Accessible buckets:      {len(accessible_buckets)} ✅")
    print(f"Inaccessible buckets:    {len(inaccessible_buckets)} ❌")
    
    # Calculate totals across all accessible buckets
    if accessible_buckets:
        total_objects_all = sum(b['object_count'] for b in accessible_buckets)
        total_size_all = sum(b['total_size'] for b in accessible_buckets)
        
        print(f"Total objects:           {total_objects_all:,} 📦")
        print(f"Total size:              {format_size(total_size_all)} 💾")
    
    print()
    
    if accessible_buckets:
        print("🎉 ACCESSIBLE BUCKETS (You can download from these):")
        print("-" * 70)
        for bucket_info in accessible_buckets:
            print(f"  ✅ {bucket_info['name']}")
            print(f"     Region: {bucket_info['region']}")
            print(f"     Objects: {bucket_info['object_count']:,} | Size: {format_size(bucket_info['total_size'])}")
        print()
    
    if inaccessible_buckets:
        print("❌ INACCESSIBLE BUCKETS (Cannot download from these):")
        print("-" * 70)
        for bucket in inaccessible_buckets:
            print(f"  ❌ {bucket}")
        print()
        print("💡 TIP: Inaccessible buckets are often in different regions.")
        print("   You may need different credentials for those regions.")
        print()
    
    # Final recommendations
    if accessible_buckets:
        print("=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print()
        print("You can now download files from the accessible buckets using:")
        print()
        print("1. rclone (fastest for large syncs)")
        print("2. MinIO client (mc) (good for quick downloads)")
        print("3. boto3 (Python library, good for programmatic access)")
        print()
        print("Would you like me to help you set up a download script? 😊")
    
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
