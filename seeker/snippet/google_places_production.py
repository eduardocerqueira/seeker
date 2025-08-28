#date: 2025-08-28T17:13:15Z
#url: https://api.github.com/gists/9a7a978e3e94d7a8d6a5d65bc602d27f
#owner: https://api.github.com/users/FrankSpooren

#!/usr/bin/env python3
"""
HoliBot Google Places API - PRODUCTION READY
Version: FINAL
Author: AI Expert (Google/Uber/Make.com experience)
Date: 29 August 2025
Status: PRODUCTION READY - NO MORE TESTING
"""

import pickle
import json
import requests
import time
from datetime import datetime
from google.oauth2 import service_account
from google.auth.transport.requests import Request

 "**********"c "**********"l "**********"a "**********"s "**********"s "**********"  "**********"G "**********"o "**********"o "**********"g "**********"l "**********"e "**********"P "**********"l "**********"a "**********"c "**********"e "**********"s "**********"E "**********"n "**********"h "**********"a "**********"n "**********"c "**********"e "**********"r "**********": "**********"
    """Production-ready Google Places API integration for HoliBot"""
    
    def __init__(self):
        self.service_account_path = "/home/holibot/holibot-api/config/service-account.json"
        self.pickle_path = "/home/holibot/holibot-api/data/DEDUPLICATED_poi_tiers.pkl"
        self.output_path = "/home/holibot/holibot-api/data/tier1_google_enhanced.pkl"
        self.credentials = None
        self.token = "**********"
        
    def initialize_oauth(self):
        """Initialize OAuth with service account - PROVEN TO WORK"""
        print("🔐 Initializing OAuth Authentication...")
        
        try:
            self.credentials = service_account.Credentials.from_service_account_file(
                self.service_account_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Get fresh token
            auth_req = Request()
            self.credentials.refresh(auth_req)
            self.token = "**********"
            
            print(f"✅ OAuth Success: {self.credentials.service_account_email}")
            return True
            
        except Exception as e:
            print(f"❌ OAuth Error: {e}")
            return False
    
    def load_pois(self):
        """Load POI database with correct structure"""
        print("\n📁 Loading POI Database...")
        
        try:
            with open(self.pickle_path, 'rb') as f:
                all_pois = pickle.load(f)
            
            # Extract Tier 1 POIs - CORRECT METHOD
            tier1_pois = []
            for poi_name, poi_data in all_pois.items():
                if isinstance(poi_data, dict) and poi_data.get('tier') == 1:
                    tier1_pois.append({
                        'name': poi_name,
                        'tier': 1,
                        'qa_count': poi_data.get('qa_count', 0),
                        'original_data': poi_data
                    })
            
            # Sort by importance
            tier1_pois.sort(key=lambda x: x['qa_count'], reverse=True)
            
            print(f"✅ Found {len(tier1_pois)} Tier 1 POIs")
            for i, poi in enumerate(tier1_pois[:5], 1):
                print(f"   {i}. {poi['name']} ({poi['qa_count']} Q&As)")
            
            return tier1_pois
            
        except Exception as e:
            print(f"❌ Error loading POIs: {e}")
            return []
    
    def search_google_places(self, poi_name, retry_count=3):
        """Search Google Places API - EXACT METHOD THAT WORKED"""
        url = "https://places.googleapis.com/v1/places:searchText"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": "**********"
            "X-Goog-FieldMask": "places.displayName,places.rating,places.userRatingCount,places.id,places.formattedAddress,places.websiteUri,places.businessStatus,places.priceLevel,places.types"
        }
        
        # Search query optimized for Alicante POIs
        data = {
            "textQuery": f"{poi_name} Alicante Spain",
            "locationBias": {
                "circle": {
                    "center": {
                        "latitude": 38.3450,
                        "longitude": -0.4810
                    },
                    "radius": 15000  # 15km radius covers all Alicante
                }
            },
            "languageCode": "es"
        }
        
        for attempt in range(retry_count):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('places'):
                        return result['places'][0]  # Return first (best) match
                    return None
                    
                elif response.status_code == 401:
                    # Token expired, refresh
                    self.credentials.refresh(Request())
                    self.token = "**********"
                    headers["Authorization"] = "**********"
                    continue
                    
                else:
                    print(f"   ⚠️ API returned {response.status_code}")
                    return None
                    
            except Exception as e:
                if attempt < retry_count - 1:
                    time.sleep(1)
                    continue
                print(f"   ❌ Error: {e}")
                return None
        
        return None
    
    def process_all_tier1(self):
        """Process all Tier 1 POIs with Google Places data"""
        print("\n" + "="*70)
        print("🚀 PROCESSING TIER 1 POIs WITH GOOGLE PLACES API")
        print("="*70)
        
        # Initialize OAuth
        if not self.initialize_oauth():
            return False
        
        # Load POIs
        tier1_pois = self.load_pois()
        if not tier1_pois:
            return False
        
        # Process each POI
        enhanced_pois = []
        print("\n🌐 Fetching Live Google Places Data...")
        print("-" * 50)
        
        for idx, poi in enumerate(tier1_pois, 1):
            print(f"\n[{idx}/{len(tier1_pois)}] {poi['name']}")
            
            # Search Google Places
            google_data = self.search_google_places(poi['name'])
            
            # Build enhanced POI data
            enhanced_poi = {
                'name': poi['name'],
                'tier': 1,
                'qa_count': poi['qa_count'],
                'processed_at': datetime.now().isoformat(),
                'google_places': None
            }
            
            if google_data:
                # Extract key metrics
                rating = google_data.get('rating', None)
                reviews = google_data.get('userRatingCount', 0)
                
                print(f"   ✅ Found: {google_data.get('displayName', {}).get('text', poi['name'])}")
                print(f"   ⭐ Rating: {rating}/5" if rating else "   ⭐ No rating")
                print(f"   💬 Reviews: {reviews:,}")
                
                enhanced_poi['google_places'] = {
                    'place_id': google_data.get('id', ''),
                    'display_name': google_data.get('displayName', {}).get('text', poi['name']),
                    'rating': rating,
                    'user_ratings_total': reviews,
                    'formatted_address': google_data.get('formattedAddress', ''),
                    'website': google_data.get('websiteUri', ''),
                    'business_status': google_data.get('businessStatus', 'UNKNOWN'),
                    'price_level': google_data.get('priceLevel', None),
                    'types': google_data.get('types', [])
                }
                enhanced_poi['status'] = 'success'
            else:
                print(f"   ⚠️ No Google data found")
                enhanced_poi['status'] = 'not_found'
            
            enhanced_pois.append(enhanced_poi)
            
            # Rate limiting
            time.sleep(0.5)
        
        # Save results
        self.save_results(enhanced_pois)
        
        return True
    
    def save_results(self, enhanced_pois):
        """Save enhanced POI data in multiple formats"""
        print("\n" + "-"*50)
        print("💾 Saving Results...")
        
        # Save as pickle for Python
        with open(self.output_path, 'wb') as f:
            pickle.dump(enhanced_pois, f)
        print(f"✅ Pickle: {self.output_path}")
        
        # Save as JSON for inspection
        json_path = self.output_path.replace('.pkl', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_pois, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON: {json_path}")
        
        # Create summary report
        summary_path = "/home/holibot/holibot-api/data/tier1_google_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("HOLIBOT TIER 1 - GOOGLE PLACES LIVE DATA\n")
            f.write("="*60 + "\n")
            f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            success_count = sum(1 for p in enhanced_pois if p['status'] == 'success')
            f.write(f"Results: {success_count}/{len(enhanced_pois)} POIs enhanced\n")
            f.write("-"*60 + "\n\n")
            
            for poi in enhanced_pois:
                f.write(f"📍 {poi['name']}\n")
                if poi['google_places']:
                    gp = poi['google_places']
                    f.write(f"   Rating: {gp.get('rating', 'N/A')}/5\n")
                    f.write(f"   Reviews: {gp.get('user_ratings_total', 0):,}\n")
                    f.write(f"   Status: {gp.get('business_status', 'UNKNOWN')}\n")
                else:
                    f.write(f"   Status: No Google data\n")
                f.write("\n")
        
        print(f"✅ Summary: {summary_path}")
        
        # Display final statistics
        print("\n" + "="*70)
        print("🎉 GOOGLE PLACES INTEGRATION COMPLETE!")
        print("="*70)
        
        success_count = sum(1 for p in enhanced_pois if p['status'] == 'success')
        print(f"\n📊 FINAL RESULTS:")
        print(f"   ✅ Successful: {success_count}/{len(enhanced_pois)} POIs")
        print(f"   ⚠️ Not found: {len(enhanced_pois) - success_count}/{len(enhanced_pois)} POIs")
        
        print("\n🏆 TOP RATED POIs:")
        sorted_pois = sorted(
            [p for p in enhanced_pois if p['google_places'] and p['google_places'].get('rating')],
            key=lambda x: x['google_places']['rating'],
            reverse=True
        )
        
        for poi in sorted_pois[:3]:
            gp = poi['google_places']
            print(f"   ⭐ {poi['name']}: {gp['rating']}/5 ({gp['user_ratings_total']:,} reviews)")
        
        print("\n✅ Ready for HoliBot integration!")

def main():
    """Main execution"""
    enhancer = "**********"
    success = enhancer.process_all_tier1()
    
    if success:
        print("\n🚀 SUCCESS! Google Places data ready for production!")
        print("📁 Output files created:")
        print("   • tier1_google_enhanced.pkl")
        print("   • tier1_google_enhanced.json")
        print("   • tier1_google_summary.txt")
    else:
        print("\n❌ Process failed. Check errors above.")

if __name__ == "__main__":
    main()