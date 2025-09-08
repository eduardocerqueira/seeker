#date: 2025-09-08T17:06:57Z
#url: https://api.github.com/gists/b73e443bdbad6292ab76c4badedd656a
#owner: https://api.github.com/users/joel-medicala-yral

#!/usr/bin/env python3
"""
TikTok Search with Proper Pagination
Based on how Apify and other scrapers handle TikTok pagination
"""

import json
import time
import random
import hashlib
from urllib.parse import quote
import requests

class TikTokPaginatedSearch:
    """
    TikTok search with working pagination
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.device_id = self._generate_device_id()
        
        # Headers that work for TikTok
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.tiktok.com/',
            'Origin': 'https://www.tiktok.com',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
        }
        
        # Initialize cookies by visiting TikTok
        self._init_session()
    
    def _generate_device_id(self):
        """Generate a device ID similar to TikTok's format"""
        return str(random.randint(7000000000000000000, 7999999999999999999))
    
    def _init_session(self):
        """Initialize session with TikTok cookies"""
        try:
            # Visit TikTok to get initial cookies
            self.session.get('https://www.tiktok.com/', headers=self.headers, timeout=10)
            time.sleep(1)
        except:
            pass
    
    def _generate_verifyFp(self):
        """Generate verifyFp token"""
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        return 'verify_' + ''.join(random.choices(chars, k=16))
    
    def _build_search_params(self, keyword, offset=0, count=20, search_id=None):
        """
        Build search parameters with proper pagination
        
        TikTok uses offset-based pagination:
        - offset: Starting position (0, 12, 24, 36, etc.)
        - count: Number of results per page (usually 12-20)
        - search_id: Maintains search session for pagination
        """
        
        # Generate search_id if not provided (for first request)
        if not search_id:
            # Create a unique search ID based on timestamp and random
            search_id = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()
        
        params = {
            # Core search parameters
            'keyword': keyword,
            'offset': str(offset),
            'count': str(count),
            'search_id': search_id,
            
            # Filter and sorting
            'type': '1',  # 1 = videos, 2 = users, 3 = hashtags
            'sort_type': '0',  # 0 = relevance, 1 = likes, 2 = date
            'publish_time': '0',  # 0 = all time, 1 = last 24h, 7 = last week, 30 = last month
            
            # Required parameters
            'aid': '1988',
            'app_language': 'en',
            'app_name': 'tiktok_web',
            'browser_language': 'en-US',
            'browser_name': 'Mozilla',
            'browser_online': 'true',
            'browser_platform': 'Win32',
            'browser_version': '5.0 (Windows NT 10.0; Win64; x64)',
            'channel': 'tiktok_web',
            'cookie_enabled': 'true',
            'device_id': self.device_id,
            'device_platform': 'web_pc',
            'focus_state': 'true',
            'from_page': 'search',
            'history_len': '3',
            'is_fullscreen': 'false',
            'is_page_visible': 'true',
            'language': 'en',
            'os': 'windows',
            'priority_region': 'US',
            'referer': '',
            'region': 'US',
            'screen_height': '1080',
            'screen_width': '1920',
            'tz_name': 'America/New_York',
            'verifyFp': self._generate_verifyFp(),
            'webcast_language': 'en',
            
            # Additional search parameters
            'search_source': 'normal_search',
            'query_source': 'default',
            'is_filter_search': '0',
            'min_cursor': str(offset),
            'max_cursor': str(offset + count),
            'enter_from': 'search_result',
        }
        
        return params
    
    def search_page(self, keyword, offset=0, count=20, search_id=None):
        """
        Search a single page of results
        
        Returns:
            tuple: (videos, has_more, next_offset, search_id)
        """
        
        # API endpoints to try
        endpoints = [
            'https://www.tiktok.com/api/search/general/full/',
            'https://www.tiktok.com/api/search/item/full/',
        ]
        
        for endpoint in endpoints:
            try:
                params = self._build_search_params(keyword, offset, count, search_id)
                
                response = self.session.get(
                    endpoint,
                    params=params,
                    headers=self.headers,
                    timeout=15
                )
                
                if response.status_code != 200:
                    continue
                
                data = response.json()
                
                # Extract videos from response
                videos = []
                items = data.get('data', [])
                
                for item in items:
                    if item.get('type') == 1:  # Video type
                        video_info = item.get('item', {})
                        if not video_info.get('id'):
                            continue
                        
                        author = video_info.get('author', {})
                        stats = video_info.get('stats', {})
                        
                        video = {
                            'id': video_info.get('id'),
                            'url': f"https://www.tiktok.com/@{author.get('uniqueId', '_')}/video/{video_info.get('id')}",
                            'title': video_info.get('desc', 'No title'),
                            'creator': author.get('nickname', 'Unknown'),
                            'username': author.get('uniqueId', 'unknown'),
                            'create_time': video_info.get('createTime', 0),
                            'duration': video_info.get('video', {}).get('duration', 0),
                            'views': stats.get('playCount', 0),
                            'likes': stats.get('diggCount', 0),
                            'shares': stats.get('shareCount', 0),
                            'comments': stats.get('commentCount', 0),
                        }
                        videos.append(video)
                
                # Check if there are more results
                has_more = data.get('has_more', 0) == 1
                
                # Calculate next offset
                next_offset = offset + len(videos)
                
                # Get or maintain search_id
                search_id = data.get('search_id') or params['search_id']
                
                return videos, has_more, next_offset, search_id
                
            except Exception as e:
                print(f"Error with {endpoint}: {e}")
                continue
        
        return [], False, offset, search_id
    
    def search(self, keyword, limit=100, page_size=20):
        """
        Search TikTok with pagination
        
        Args:
            keyword: Search term
            limit: Maximum number of videos to retrieve
            page_size: Number of videos per page (12-30 recommended)
        
        Returns:
            List of video dictionaries
        """
        
        print(f"\n🔍 Searching TikTok for '{keyword}' with pagination")
        print(f"Target: {limit} videos, Page size: {page_size}")
        print("=" * 60)
        
        all_videos = []
        seen_ids = set()
        offset = 0
        search_id = None
        page = 1
        consecutive_empty = 0
        
        while len(all_videos) < limit:
            print(f"\nPage {page}: Requesting offset {offset}...")
            
            # Get page of results
            videos, has_more, next_offset, search_id = self.search_page(
                keyword, offset, page_size, search_id
            )
            
            if not videos:
                consecutive_empty += 1
                print(f"  No videos returned (attempt {consecutive_empty}/3)")
                
                if consecutive_empty >= 3:
                    print("  No more results available")
                    break
                
                # Try different offset strategy
                offset += page_size
                time.sleep(2)
                continue
            
            consecutive_empty = 0
            new_videos = 0
            
            # Add unique videos
            for video in videos:
                video_id = video.get('id')
                if video_id and video_id not in seen_ids:
                    seen_ids.add(video_id)
                    all_videos.append(video)
                    new_videos += 1
                    
                    if len(all_videos) >= limit:
                        break
            
            print(f"  Found {len(videos)} videos, {new_videos} new (total: {len(all_videos)})")
            
            # Check if we have enough
            if len(all_videos) >= limit:
                print(f"\n✅ Reached target of {limit} videos")
                break
            
            # Check if there are more pages
            if not has_more:
                print("  No more pages available")
                break
            
            # Update offset for next page
            offset = next_offset
            page += 1
            
            # Rate limiting - important for pagination
            delay = random.uniform(1, 2)
            print(f"  Waiting {delay:.1f}s before next page...")
            time.sleep(delay)
            
            # Safety check for infinite loops
            if page > 50:
                print("  Reached maximum page limit")
                break
        
        return all_videos[:limit]

def search_tiktok_paginated(keyword, limit=100):
    """
    Main function to search TikTok with pagination
    
    Usage:
        videos = search_tiktok_paginated('funny cats', 100)
    """
    
    searcher = TikTokPaginatedSearch()
    videos = searcher.search(keyword, limit)
    
    if videos:
        print(f"\n✅ Successfully retrieved {len(videos)} videos!")
        print("\nFirst 10 results:")
        print("-" * 60)
        
        for i, video in enumerate(videos[:10], 1):
            title = video['title'][:70] + '...' if len(video['title']) > 70 else video['title']
            print(f"\n{i}. {title}")
            print(f"   Creator: @{video['username']} ({video['creator']})")
            print(f"   Stats: {video['views']:,} views, {video['likes']:,} likes")
            print(f"   URL: {video['url']}")
        
        if len(videos) > 10:
            print(f"\n... and {len(videos) - 10} more videos")
    else:
        print("\n❌ No videos found")
    
    return videos

if __name__ == '__main__':
    # Test with pagination
    keyword = 'funny cats'
    target = 1000
    
    print("TikTok Search with Pagination")
    print("=" * 60)
    
    videos = search_tiktok_paginated(keyword, target)
    
    # Save results
    if videos:
        with open('tiktok_paginated_results.json', 'w') as f:
            json.dump(videos, f, indent=2)
        
        print(f"\n💾 Results saved to tiktok_paginated_results.json")
        print(f"Total videos retrieved: {len(videos)}")
        
        # Show statistics
        total_views = sum(v.get('views', 0) for v in videos)
        total_likes = sum(v.get('likes', 0) for v in videos)
        
        print(f"\nStatistics:")
        print(f"  Total views: {total_views:,}")
        print(f"  Total likes: {total_likes:,}")
        print(f"  Average views: {total_views // len(videos):,}" if videos else "")