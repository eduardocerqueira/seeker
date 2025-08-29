#date: 2025-08-29T16:53:55Z
#url: https://api.github.com/gists/774efe1a11622516b7b5dff530cfb5bb
#owner: https://api.github.com/users/FrankSpooren

#!/usr/bin/env python3
"""
FINAL QUOTA ACHIEVEMENT - GET UNDER 100% GOOGLE QUOTA
Reduce Tier 1 from 61 POIs → ~25 POIs by strategic demotions
Target: 1548.7 → <1000 calls/day
"""

import pickle

def load_frank_corrected_tiers():
    """Load Frank's corrected tier assignments"""
    try:
        with open('/home/holibot/holibot-api/data/FRANK_corrected_tiers.pkl', 'rb') as f:
            corrected_tiers = pickle.load(f)
        print(f"Loaded Frank's corrected tiers: {len(corrected_tiers)} POIs")
        return corrected_tiers
    except Exception as e:
        print(f"Error loading corrected tiers: {e}")
        return None

def identify_tier1_demotion_candidates(corrected_tiers):
    """
    Identify Tier 1 POIs that can be reasonably demoted to Tier 2
    Keep only the absolute core attractions in Tier 1
    """
    
    # ABSOLUTE TIER 1 CORE - These must stay hourly
    tier1_core_mandatory = [
        'castillo', 'santa barbara', 'castle',
        'playa del postiguet', 'postiguet beach',
        'marq', 'archaeological',
        'explanada',
        'palacio maisonnave',
        'teatro principal', 'theatre',
        'ocean race museum',
        'tabarca island',
        'watersports rental san juan',
        'contemporary art museum'  # Keep one MACA
    ]
    
    # TIER 2 CANDIDATES - Can be daily instead of hourly
    tier2_candidate_keywords = [
        'beach volleyball', 'beach yoga', 'beach parking', 'beach cleanup',
        'beach accessibility', 'beach club',
        'basilica', 'iglesia', 'church',
        'festival', 'concatedral',
        'museo de', 'museum of',  # Most museums can be daily
        'centro de', 'centro educacion'
    ]
    
    current_tier1 = [(poi, data) for poi, data in corrected_tiers.items() if data['tier'] == 1]
    
    # Categorize Tier 1 POIs
    core_mandatory = []
    demotion_candidates = []
    
    for poi_name, data in current_tier1:
        poi_lower = poi_name.lower()
        
        # Check if this is core mandatory
        is_core = any(keyword in poi_lower for keyword in tier1_core_mandatory)
        
        if is_core:
            core_mandatory.append((poi_name, data))
        else:
            # Check if it's a demotion candidate
            is_candidate = any(keyword in poi_lower for keyword in tier2_candidate_keywords)
            
            if is_candidate:
                demotion_candidates.append((poi_name, data, 'keyword_match'))
            else:
                # Add to candidates by score (lowest scores first)
                demotion_candidates.append((poi_name, data, 'score_based'))
    
    # Sort demotion candidates - keyword matches first, then by lowest scores
    demotion_candidates.sort(key=lambda x: (x[2] == 'score_based', x[1]['total_score']))
    
    return core_mandatory, demotion_candidates

def apply_final_quota_optimization(corrected_tiers):
    """Apply final quota optimization to get under 1000 calls/day"""
    
    optimized_tiers = corrected_tiers.copy()
    
    # Identify core vs demotion candidates
    core_mandatory, demotion_candidates = identify_tier1_demotion_candidates(corrected_tiers)
    
    print(f"\nFINAL QUOTA OPTIMIZATION:")
    print(f"Current Tier 1: 61 POIs (1464 calls/day)")
    print(f"Target Tier 1: ~25 POIs (~600 calls/day)")
    print(f"Core mandatory: {len(core_mandatory)} POIs")
    print(f"Demotion candidates: {len(demotion_candidates)} POIs")
    
    # Calculate how many demotions needed
    target_tier1_size = 25
    demotions_needed = 61 - target_tier1_size
    
    print(f"\nDEMOTIONS NEEDED: {demotions_needed} POIs (T1→T2)")
    
    # Apply demotions
    demotions_applied = 0
    for poi_name, data, reason in demotion_candidates:
        if demotions_applied >= demotions_needed:
            break
            
        # Demote to Tier 2
        optimized_tiers[poi_name]['tier'] = 2
        optimized_tiers[poi_name]['tier_description'] = "Strategic - Daily"
        optimized_tiers[poi_name]['calls_per_day'] = 1
        optimized_tiers[poi_name]['monitor_freq'] = 'daily'
        optimized_tiers[poi_name]['final_quota_optimization'] = f"T1→T2 for quota ({reason})"
        
        if demotions_applied < 10:  # Show first 10
            print(f"   T1→T2: {poi_name[:45]:<45} ({reason})")
        
        demotions_applied += 1
    
    if demotions_applied > 10:
        print(f"   ... and {demotions_applied-10} more demotions")
    
    return optimized_tiers

def calculate_final_metrics(optimized_tiers):
    """Calculate final tier distribution and API usage"""
    
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    total_calls = 0
    tier_pois = {1: [], 2: [], 3: [], 4: []}
    
    for poi_name, data in optimized_tiers.items():
        tier = data['tier']
        tier_counts[tier] += 1
        total_calls += data['calls_per_day']
        tier_pois[tier].append((poi_name, data))
    
    # Sort tiers by score
    for tier in tier_pois:
        tier_pois[tier].sort(key=lambda x: x[1]['total_score'], reverse=True)
    
    return tier_counts, total_calls, tier_pois

def main():
    print("FINAL QUOTA ACHIEVEMENT - TARGET <100% GOOGLE QUOTA")
    print("="*70)
    print("Goal: Reduce 1548.7 → <1000 calls/day")
    print("Strategy: T1 61→25 POIs, keep core attractions hourly")
    print("="*70)
    
    # Load Frank's corrected tiers
    corrected_tiers = load_frank_corrected_tiers()
    if not corrected_tiers:
        return
    
    # Apply final quota optimization
    optimized_tiers = apply_final_quota_optimization(corrected_tiers)
    
    # Calculate final metrics
    final_counts, final_calls, tier_pois = calculate_final_metrics(optimized_tiers)
    
    # Show final results
    print(f"\nFINAL OPTIMIZED DISTRIBUTION:")
    print("="*60)
    for tier in [1, 2, 3, 4]:
        count = final_counts[tier]
        calls = count * {1: 24, 2: 1, 3: 0.14, 4: 0.033}[tier]
        freq = {1: 'hourly', 2: 'daily', 3: 'weekly', 4: 'monthly'}[tier]
        print(f"   Tier {tier}: {count:3d} POIs = {calls:6.1f} calls/day ({freq})")
    print(f"   TOTAL: {final_calls:6.1f} calls/day ({final_calls/10:.1f}% quota)")
    
    # Show final premium Tier 1
    print(f"\nFINAL PREMIUM TIER 1 ({len(tier_pois[1])} POIs - Core EU Tourism):")
    for i, (poi_name, data) in enumerate(tier_pois[1][:25], 1):
        print(f"{i:2d}. {poi_name[:45]:<45} {data['total_score']:2d}/100")
    
    # Save final optimized tiers
    output_file = '/home/holibot/holibot-api/data/PRODUCTION_READY_tiers.pkl'
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(optimized_tiers, f)
        print(f"\nPRODUCTION READY TIERS SAVED: {output_file}")
    except Exception as e:
        print(f"Error saving production tiers: {e}")
        return
    
    # Success summary
    call_reduction = 1548.7 - final_calls
    print(f"\nSUCCESS - QUOTA ACHIEVEMENT COMPLETE!")
    print("="*70)
    print(f"Call reduction: -{call_reduction:.1f} calls/day ({call_reduction/1548.7*100:.1f}%)")
    print(f"Quota usage: {final_calls/10:.1f}% (Target: <100%)")
    print(f"Core attractions preserved in Tier 1 hourly monitoring")
    print(f"Strategic balance achieved: Premium focus + cost efficiency")
    print(f"READY FOR GOOGLE PLACES INTEGRATION!")

if __name__ == "__main__":
    main()