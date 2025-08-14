#date: 2025-08-14T17:13:33Z
#url: https://api.github.com/gists/fa8c27f1e479ad94949857d268cd074f
#owner: https://api.github.com/users/zone559

import requests
import os
import re
import time
from urllib.parse import urlparse

def get_user_id_from_url(url):
    """Extract user ID from profile URL"""
    # Try to find ID in path
    match = re.search(r'/(\d+)/', url)
    if match:
        return match.group(1)
    raise ValueError("Could not extract user ID from URL")

def fetch_all_articles(session, user_id):
    """Fetch all articles with pagination"""
    base_url = f"https://api.onstove.com/postie/v1.0/interest/user/{user_id}/article/list"
    all_articles = []
    token = "**********"
    
    while True:
        params = {"user_id": user_id}
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            params["token"] = "**********"
        
        try:
            response = session.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data["code"] != 0:
                print(f"API error: {data['message']}")
                break
                
            articles = data["value"]["list"]
            all_articles.extend(articles)
            
            if not data["value"]["has_next"]:
                break
                
            token = "**********"
            
        except Exception as e:
            print(f"Error fetching articles: {str(e)}")
            break
            
    return all_articles

def download_images(session, media_urls, download_dir):
    """Download all images with progress tracking"""
    os.makedirs(download_dir, exist_ok=True)
    print(f"\nDownloading {len(media_urls)} images to '{download_dir}'...")
    
    for i, url in enumerate(media_urls, 1):
        try:
            if not url.startswith(('http://', 'https://')):
                url = f"https:{url}" if url.startswith('//') else f"https://{url}"
            
            filename = os.path.basename(urlparse(url).path)
            filepath = os.path.join(download_dir, filename)
            
            if os.path.exists(filepath):
                print(f"[{i}/{len(media_urls)}] Exists: {filename}")
                continue
                
            response = session.get(url, stream=True, timeout=10)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"[{i}/{len(media_urls)}] Downloaded: {filename}")
            
        except Exception as e:
            print(f"[{i}/{len(media_urls)}] Failed {filename}: {str(e)}")

def main():
    # Configure session
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    # Get user input
    profile_url = input("Enter profile URL (e.g. https://profile.onstove.com/en/132936468/list): ").strip()
    try:
        user_id = get_user_id_from_url(profile_url)
        print(f"Extracted user ID: {user_id}")
    except ValueError as e:
        print(f"Error: {str(e)}")
        return
    
    # Fetch all articles
    print("\nFetching articles...")
    articles = fetch_all_articles(session, user_id)
    
    # Extract media URLs
    media_urls = []
    for article in articles:
        if "attached" in article and "medias" in article["attached"]:
            media_urls.extend(
                media["media_url"] 
                for media in article["attached"]["medias"] 
                if media["media_type"] == "IMAGE"
            )
    
    if not media_urls:
        print("No images found to download")
        return
    
    # Download images
    download_dir = f"onstove_images_{user_id}"
    download_images(session, media_urls, download_dir)
    
    # Count downloaded files
    try:
        file_count = len([f for f in os.listdir(download_dir) if os.path.isfile(os.path.join(download_dir, f))])
        print(f"\nDone! Downloaded {file_count} files.")
    except Exception as e:
        print(f"\nError counting downloaded files: {str(e)}")

if __name__ == "__main__":
    main()