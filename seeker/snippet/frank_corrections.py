#date: 2025-08-29T16:47:07Z
#url: https://api.github.com/gists/6e09990b516e579feaf151fa19288297
#owner: https://api.github.com/users/FrankSpooren

#!/usr/bin/env python3
"""
FRANK'S SPECIFIC TIER CORRECTIONS
1. Fix MACA duplication (POI #7 and #9 are same museum)  
2. Move museums #11-20 from Tier 1 to Tier 2 (daily monitoring sufficient)
3. Optimize final Tier 1 to ~25 core attractions for EU tourism
"""

import pickle
import re

def load_strategic_manual_tiers():
    """Load current strategic manual override tier assignments"""
    try:
        with open('/home/holibot/holibot-api/data/STRATEGIC_manual_override_tiers.pkl', 'rb') as f:
            strategic_tiers = pickle.load(f)
        print(f"✅ Loaded strategic manual override tiers: {len(strategic_tiers)} POIs")
        return strategic_tiers
    except Exception as e:
        print(f"❌ Error loading strategic tiers: {e}")
        return None

def identify_maca_duplication(strategic_tiers):
    """Identify MACA museum duplication"""
    
    maca_variants = []
    for poi_name, data in strategic_tiers.items():
        poi_lower = poi_name.lower()
        if ('maca' in poi_lower or 
            ('contemporary' in poi_lower and 'art' in poi_lower) or
            ('arte contemporáneo' in poi_lower) or
            ('arte contemporaneo' in poi_lower)):
            maca_variants.append((poi_name, data))
    
    print(f"\n🔍 MACA DUPLICATION DETECTED:")
    for poi_name, data in maca_variants:
        tier = data['tier']
        score = data['total_score']
        print(f"   {poi_name} - Tier {tier}, Score {score}/100")
    
    return maca_variants

def identify_museums_for_tier2_demotion(strategic_tiers):
    """
    Identify museums that should be demoted from Tier 1 to Tier 2
    Based on Frank's input: museums #11-20 don't need hourly updates
    """
    
    # Museums that should be Tier 2 (daily monitoring sufficient)
    tier2_museum_keywords = [
        'museo de ciencias naturales',
        'museo de fogueres', 
        'museo de juguetes',
        'museo de la ciudad',
        'museo de la semana santa',
        'museo de música étnica',
        'museo del agua',
        'museo del mar',
        'museo interactivo de la ciencia',
        'natural sciences',
        'toys museum',
        'toy museum',
        'water museum',
        'sea museum',
        'interactive science',
        'ethnic music'
    ]
    
    museums_to_demote = []
    for poi_name, data in strategic_tiers.items():
        if data['tier'] == 1:  # Only check current Tier 1 POIs
            poi_lower = poi_name.lower()
            
            # Check if this is a museum that should be demoted
            for keyword in tier2_museum_keywords:
                if keyword in poi_lower:
                    museums_to_demote.append((poi_name, data))
                    break
    
    return museums_to_demote

def identify_core_tier1_attractions(strategic_tiers):
    """
    Identify core Tier 1 attractions that should definitely remain hourly
    These are the absolute must-haves for EU tourism
    """
    
    core_tier1_keywords = [
        'castillo', 'castle', 'santa barbara',
        'playa del postiguet', 'postiguet',
        'marq', 'archaeological',
        'explanada', 
        'central market', 'mercado central',
        'palacio maisonnave',
        'teatro principal', 'theatre',
        'ocean race museum',
        'tabarca island',
        'watersports rental',
        'contemporary art museum',  # Keep one MACA
        'mubag', 'gravina museum'
    ]
    
    core_attractions = []
    for poi_name, data in strategic_tiers.items():
        poi_lower = poi_name.lower()
        
        for keyword in core_tier1_keywords:
            if keyword in poi_lower:
                core_attractions.append((poi_name, data))
                break
    
    return core_attractions

def apply_frank_corrections(strategic_tiers):
    """Apply Frank's specific corrections"""
    
    corrected_tiers = strategic_tiers.copy()
    
    print(f"\n🔧 APPLYING FRANK'S SPECIFIC CORRECTIONS:")
    print("-" * 60)
    
    # 1. Fix MACA duplication - keep the higher scoring one
    maca_variants = identify_maca_duplication(strategic_tiers)
    if len(maca_variants) > 1:
        print(f"\n1️⃣ FIXING MACA DUPLICATION:")
        
        # Sort by score, keep the highest scoring one
        maca_sorted = sorted(maca_variants, key=lambda x: x[1]['total_score'], reverse=True)
        keep_maca = maca_sorted[0]
        remove_maca = maca_sorted[1:]
        
        print(f"   ✅ KEEPING: {keep_maca[0]} (Score: {keep_maca[1]['total_score']}/100)")
        
        for poi_name, data in remove_maca:
            # Move duplicate to Tier 4 (effectively removing from active monitoring)
            corrected_tiers[poi_name]['tier'] = 4
            corrected_tiers[poi_name]['tier_description'] = "Extended - Monthly"
            corrected_tiers[poi_name]['calls_per_day'] = 0.033
            corrected_tiers[poi_name]['monitor_freq'] = 'monthly'
            corrected_tiers[poi_name]['frank_correction'] = "Duplicate removal - MACA deduplication"
            print(f"   ❌ DUPLICATE: {poi_name} → Tier 4 (deactivated)")
    
    # 2. Demote museums #11-20 to Tier 2  
    museums_to_demote = identify_museums_for_tier2_demotion(strategic_tiers)
    if museums_to_demote:
        print(f"\n2️⃣ DEMOTING MUSEUMS TO TIER 2 (Daily monitoring):")
        
        for poi_name, data in museums_to_demote:
            if data['tier'] == 1:  # Only demote if currently Tier 1
                corrected_tiers[poi_name]['tier'] = 2
                corrected_tiers[poi_name]['tier_description'] = "Strategic - Daily"
                corrected_tiers[poi_name]['calls_per_day'] = 1
                corrected_tiers[poi_name]['monitor_freq'] = 'daily'
                corrected_tiers[poi_name]['frank_correction'] = "T1→T2: Museums don't need hourly updates"
                print(f"   ↘️ {poi_name[:45]:<45} T1→T2 (Museum - daily sufficient)")
    
    # 3. Ensure core attractions remain in Tier 1
    core_attractions = identify_core_tier1_attractions(strategic_tiers)
    print(f"\n3️⃣ VERIFIED CORE TIER 1 ATTRACTIONS:")
    
    for poi_name, data in core_attractions:
        if corrected_tiers[poi_name]['tier'] != 1:
            # Promote back to Tier 1 if accidentally demoted
            corrected_tiers[poi_name]['tier'] = 1
            corrected_tiers[poi_name]['tier_description'] = "Premium - Hourly"
            corrected_tiers[poi_name]['calls_per_day'] = 24
            corrected_tiers[poi_name]['monitor_freq'] = 'hourly'
            corrected_tiers[poi_name]['frank_correction'] = "Core attraction - T1 mandatory"
            print(f"   ↗️ {poi_name[:45]:<45} → T1 (Core attraction)")
        else:
            print(f"   ✅ {poi_name[:45]:<45} T1 confirmed")
    
    return corrected_tiers

def calculate_final_distribution(corrected_tiers):
    """Calculate final tier distribution after Frank's corrections"""
    
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    total_calls = 0
    tier_pois = {1: [], 2: [], 3: [], 4: []}
    
    for poi_name, data in corrected_tiers.items():
        tier = data['tier']
        tier_counts[tier] += 1
        total_calls += data['calls_per_day']
        tier_pois[tier].append((poi_name, data))
    
    # Sort each tier by score
    for tier in tier_pois:
        tier_pois[tier].sort(key=lambda x: x[1]['total_score'], reverse=True)
    
    return tier_counts, total_calls, tier_pois

def main():
    print("🔧 FRANK'S SPECIFIC TIER CORRECTIONS")
    print("=" * 70)
    print("Corrections:")
    print("1. Fix MACA duplication (keep highest scoring variant)")
    print("2. Move museums #11-20 from Tier 1 to Tier 2")  
    print("3. Ensure core attractions remain in Tier 1")
    print("=" * 70)
    
    # Load current strategic tiers
    strategic_tiers = load_strategic_manual_tiers()
    if not strategic_tiers:
        return
    
    # Show current Tier 1 status
    current_tier1 = [(poi, data) for poi, data in strategic_tiers.items() if data['tier'] == 1]
    print(f"\n📊 CURRENT TIER 1: {len(current_tier1)} POIs")
    
    # Apply Frank's specific corrections
    corrected_tiers = apply_frank_corrections(strategic_tiers)
    
    # Calculate final distribution
    final_counts, final_calls, tier_pois = calculate_final_distribution(corrected_tiers)
    
    print(f"\n🎯 FINAL CORRECTED DISTRIBUTION:")
    print("=" * 60)
    for tier in [1, 2, 3, 4]:
        count = final_counts[tier]
        calls = count * {1: 24, 2: 1, 3: 0.14, 4: 0.033}[tier]
        freq = {1: 'hourly', 2: 'daily', 3: 'weekly', 4: 'monthly'}[tier]
        print(f"   Tier {tier}: {count:3d} POIs = {calls:6.1f} calls/day ({freq})")
    print(f"   TOTAL: {final_calls:6.1f} calls/day ({final_calls/10:.1f}% quota)")
    
    # Show final optimized Tier 1
    print(f"\n🏆 FINAL OPTIMIZED TIER 1 POIs ({len(tier_pois[1])} POIs):")
    for i, (poi_name, data) in enumerate(tier_pois[1][:25], 1):
        correction = "🔧" if 'frank_correction' in data else "✅"
        print(f"{i:2d}. {correction} {poi_name[:40]:<40} {data['total_score']:2d}/100")
    
    # Show Tier 2 highlights (demoted museums)
    tier2_museums = [poi for poi, data in tier_pois[2] if 'museum' in poi[0].lower() or 'museo' in poi[0].lower()]
    if tier2_museums:
        print(f"\n🏛️ TIER 2 MUSEUMS (Daily monitoring - {len(tier2_museums)} museums):")
        for i, (poi_name, data) in enumerate(tier2_museums[:10], 1):
            print(f"{i:2d}. {poi_name[:50]:<50} Daily updates")
    
    # Save Frank's corrected tiers
    output_file = '/home/holibot/holibot-api/data/FRANK_corrected_tiers.pkl'
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(corrected_tiers, f)
        print(f"\n💾 Frank's corrected tiers saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving corrected tiers: {e}")
        return
    
    # Success summary
    print(f"\n🎉 FRANK'S CORRECTIONS APPLIED SUCCESSFULLY!")
    print("=" * 70)
    print(f"✅ MACA deduplication: Removed duplicate entries")
    print(f"✅ Museums optimized: {len([p for p, d in tier2_museums])} museums → Tier 2 (daily)")
    print(f"✅ Core attractions: Secured in Tier 1 (hourly)")
    print(f"✅ API efficiency: {final_calls/10:.1f}% quota utilization")
    print(f"🚀 READY FOR GOOGLE PLACES INTEGRATION!")

if __name__ == "__main__":
    main()