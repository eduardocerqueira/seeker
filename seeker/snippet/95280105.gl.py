#date: 2024-04-18T17:07:19Z
#url: https://api.github.com/gists/9b5407ae34c622c30ee687a9332df04b
#owner: https://api.github.com/users/aEnigmatic

##
# Mission 'Centaur of Unknown Origin' (95280105)
# Enemy has first strike!
#
# Monsters
#  * Mechiron (900011400)
#
##

##
# Morale Rules - Before Battle
##
#
# Begin battle with 30% additional morale.
#
##

##
# Morale Rules - During Battle
##
#
# [Enemy] Player unit is KO'ed
# [Player] Attack enemy's weakness (Once per ability cast)
# [Both] Inflict enemy with status ailment (Once per ailment per turn)
# [Both] Boost ATK, DEF, MAG, SPR or reduce damage taken for ally (Once per status effect per unit per turn)
# [Both] Reduce ATK, DEF, MAG, or SPR for enemy (Once per status effect per unit per turn)
# [Both] Restore HP or MP for ally (Once per ability cast)
# [Both] Boost elemental resistance for ally (Once per status effect per unit per turn)
# [Both] Add water or light element to physical attack for ally (Once per status effect per unit per turn)
#
##

##
# Morale Thresholds
##
# 10% - Fatal Despair (915770) [Enemy]
#
# Instant KO (100%) to all enemies (ignores death resist)
#
##
# 40% - Enduring Will (Water/Light) (918526) [Player]
#
# Increase resistance to Water and Light by 150% for one turn to all allies
#
##
# 50% - Resurging Will (915320) [Player]
#
# Restore 5000 HP to all allies
# Restore 500 MP to all allies
#
##
# 60% - Enervating Despair (915968) [Enemy]
#
# Remove all buffs from all enemies
# Remove resistances to Poison, Blind, Sleep, Silence, Paralyze, Confusion, Disease and Petrify from all enemies for 2 turns
# Hybrid* damage (0.1x, ATK & MAG) as MP drain (10%) to all enemies
#
##
# 110% - Remedial Will (915541) [Player]
#
# Remove Zombie from all allies
# Remove all debuffs from all allies
#
##
# 115% - Scarring Despair (916116) [Enemy]
#
# Reduce healing received by 90% to all enemies for 4 turns
#
##
# 140% - Motionless Despair (915927) [Enemy]
#
# Inflict Stop (100%) for 2 turns on one enemy
#
##
# 150% - Resilient Will (915321) [Player]
#
# Increase DEF and SPR by 400% for one turn to all allies (can not be removed)
#
##
# 180% - Resounding Will (915322) [Player]
#
# Increase ATK and MAG by 400% for one turn to all allies (can not be removed)
#
##

##
# Monster Info
##
#
# Monster  Mechiron (900011400)
# Race     Beast, Human, Machina
# Level    99
# Actions  20-20
#
#
# Stats
#        HP        800000000
#        MP           100000
#        ATK            2000
#        DEF           40000
#        MAG            2000
#        SPR           40000
#
#
# Damage resist
#        physical          0%
#        magical           0%
#
#
# Element resist
#        Fire             85%
#        Ice              85%
#        Lightning         0%
#        Water           -40%
#        Wind             85%
#        Earth            85%
#        Light           -40%
#        Dark             85%
#        Non-Elemental    80%
#
#
# Status resist (+0% / application)
#        Poison          100%
#        Blind           100%
#        Sleep           100%
#        Silence         100%
#        Paralyze        100%
#        Confusion       100%
#        Disease         100%
#        Petrify         100%
#
#
# Debuff resist
#        ATK               0%
#        DEF               0%
#        MAG               0%
#        SPR               0%
#        Stop            100%
#        Charm           100%
#        Berserk         100%
#
#
# Immunity
#        Death             +
#        Gravity           +
#        Unknown (7)       +
#
###

###
# Passives
###
#
#  TFA Passive 70/100 Tier 5 (920036) [Passive]
#
#  Increase DEF and SPR by 100% and ATK and MAG by 70%
#
##

###
# Skills
###
#
#  Power Stomp (902175) [Physical]
#
#  Physical damage (50x, ATK) to one enemy (+100% accuracy)
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Laser Lance (902176) [Magic]
#
#  Magic damage (5x, MAG) to one enemy
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Laser Shower (902177) [Magic]
#
#  Magic damage (5x, MAG) to all enemies
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Divine Lance (902166) [Physical]
#
#  Physical light damage (50x, ATK) to one enemy (+100% accuracy)
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Aqua Lance (902167) [Physical]
#
#  Physical water damage (50x, ATK) to one enemy (+100% accuracy)
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Tactical Laser (902168) [Magic]
#
#  Magic damage (80x, MAG) to one enemy
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Divine Sweep (902169) [Physical]
#
#  Physical light damage (20x, ATK) to all enemies (+100% accuracy)
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Aqua Sweep (902170) [Physical]
#
#  Physical water damage (20x, ATK) to all enemies (+100% accuracy)
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Laser Blast (902171) [Magic]
#
#  Magic damage (20x, MAG) to all enemies
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Laser Arrows (902172) [Magic]
#
#  Magic damage (5x, MAG) to all enemies
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Flood of Centaurus (902173) [Physical]
#
#  Physical water and light damage (50x, ATK) to one enemy (+100% accuracy)
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Initiate Combat Protocol (902160) [None]
#
#  Increase ATK and MAG by 40% for 5 turns to caster
#  Reduce damage taken from physical attacks taken by 99% to caster for 2 turns
#  Reduce damage taken by magic attacks by 99% to caster for 2 turns
#  Increase morale by 5000 for this team
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Aegis Protocol (902161) [None]
#
#  Increase DEF and SPR by 80% for 2 turns to caster (can not be removed)
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Defense Matrix (902162) [None]
#
#  Increase DEF and SPR by 50% for 3 turns to caster
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Self-Repair Device (902163) [None]
#
#  Restore 5% HP to caster
#  Increase morale by 2000 for this team
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Cloaking Mechanism (902164) [None]
#
#  Dodge 3 physical attacks for one turn to caster
#  Increase morale by 3000 for this team
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Centaur's Rage (902165) [None]
#
#  Reduce resistance to Water and Light by 200% for 3 turns to all enemies
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Decimation Protocol (902174) [Physical]
#
#  Reduce resistance to Water and Light by 300% for one turn to all enemies
#  Physical water and light damage (300x, ATK) to all enemies (+100% accuracy)
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Show me the limits of your power! (902178) [None]
#
#  Physical* damage 0 to caster
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Centaur's Coercion (902179) [None]
#
#  Remove resistances to Paralyze from one enemy for 3 turns
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##
#
#  Curse of Centaurus (902180) [None]
#
#  Inflict Paralyze (100%) on all enemies
#
#  Sealable  -    Unknown1  +
#  Reflect   -    Unknown2  -
#
##

###
# AI
###
if once(4):
    if   True:
        useSkill(12, 'random')           # Initiate Combat Protocol (902160): Increase ATK and MAG by 40% for 5 turns to caster, Reduce damage taken from physical attacks taken by 99% to caster for 2 turns, Reduce damage taken by magic attacks by 99% to caster for 2 turns, Increase morale by 5000 for this team
        return
    
    else:
        useSkill(17, 'random')           # Centaur's Rage (902165): Reduce resistance to Water and Light by 200% for 3 turns to all enemies
        return
    
    else:
        endTurn()                        
        return
    
    if self.HP < 0.5:
        if   True:
            useSkill(13, 'random')       # Aegis Protocol (902161): Increase DEF and SPR by 80% for 2 turns to caster (can not be removed)
            return
        
        else:
            useSkill(17, 'random')       # Centaur's Rage (902165): Reduce resistance to Water and Light by 200% for 3 turns to all enemies
            return
        
        else:
            useSkill(16, 'random')       # Cloaking Mechanism (902164): Dodge 3 physical attacks for one turn to caster, Increase morale by 3000 for this team
            return
        
        else:
            ramen  = True                # persistent
            endTurn()                    
            return
        
    

if not var_61:
    var_61  = True                       # reset next turn

    if not self.sufferedDamageLastTurn('physical|magical', 'light|water'):
        wait()                           
        return
    
    else:
        useSkill(18, 'random')           # Decimation Protocol (902174): Reduce resistance to Water and Light by 300% for one turn to all enemies, Physical water and light damage (300x, ATK) to all enemies (+100% accuracy)
        return
    

if not var_62:
    if conditionNotImplemented('physics_fire_lb:0') \
      or conditionNotImplemented('physics_ice_lb:0') \
      or conditionNotImplemented('physics_thunder_lb:0') \
      or conditionNotImplemented('physics_water_lb:0') \
      or conditionNotImplemented('physics_aero_lb:0') \
      or conditionNotImplemented('physics_quake_lb:0') \
      or conditionNotImplemented('physics_light_lb:0') \
      or conditionNotImplemented('physics_dark_lb:0') \
      or conditionNotImplemented('magic_fire_lb:0') \
      or conditionNotImplemented('magic_ice_lb:0') \
      or conditionNotImplemented('magic_thunder_lb:0') \
      or conditionNotImplemented('magic_water_lb:0') \
      or conditionNotImplemented('magic_aero_lb:0') \
      or conditionNotImplemented('magic_quake_lb:0') \
      or conditionNotImplemented('magic_light_lb:0') \
      or conditionNotImplemented('magic_dark_lb:0') \
      or conditionNotImplemented('physics_none_lb:0') \
      or conditionNotImplemented('magic_none_lb:0'):
        var_62  = True                   # reset next turn
        wait()                           
        return
    
    if not var_65 and honey:
        if not var_63:
            var_63  = True               # reset next turn
            useSkill(20, 'random')       # Centaur's Coercion (902179): Remove resistances to Paralyze from one enemy for 3 turns
            return
        
        if not var_64:
            var_64  = True               # reset next turn
            var_62  = True               # reset next turn
            honey  = False               # persistent
            useSkill(20, 'random')       # Centaur's Coercion (902179): Remove resistances to Paralyze from one enemy for 3 turns
            return
        
    

if not honey and not var_65:
    var_65  = True                       # reset next turn

    if random() <= 0.30:
        honey  = True                    # persistent
        useSkill(19, 'random')           # Show me the limits of your power! (902178): Physical* damage 0 to caster
        return
    
    else:
        wait()                           
        return
    

if isTurnMod(3):
    if not ramen:
        if not var_66:
            var_66 = True                # unknown flag type  (66)
            useSkill(14, 'random')       # Defense Matrix (902162): Increase DEF and SPR by 50% for 3 turns to caster
            return
        
        if not var_67:
            var_67 = True                # unknown flag type  (67)
            useSkill(17, 'random')       # Centaur's Rage (902165): Reduce resistance to Water and Light by 200% for 3 turns to all enemies
            return
        
    
    if ramen:
        if not var_66:
            var_66 = True                # unknown flag type  (66)
            useSkill(14, 'random')       # Defense Matrix (902162): Increase DEF and SPR by 50% for 3 turns to caster
            return
        
        if not var_67:
            var_67 = True                # unknown flag type  (67)
            useSkill(15, 'random')       # Self-Repair Device (902163): Restore 5% HP to caster, Increase morale by 2000 for this team
            return
        
        if not var_68:
            var_68 = True                # unknown flag type  (68)
            useSkill(16, 'random')       # Cloaking Mechanism (902164): Dodge 3 physical attacks for one turn to caster, Increase morale by 3000 for this team
            return
        
        if not var_69:
            var_69 = True                # unknown flag type  (69)
            useSkill(17, 'random')       # Centaur's Rage (902165): Reduce resistance to Water and Light by 200% for 3 turns to all enemies
            return
        
    

if not var_70:
    var_70 = True                        # unknown flag type  (70)
    black  = 0                           
    white  = 0                           
    green += 1                           
    wait()                               
    return

if green == 1:
    if not ramen:
        if not apple:
            apple  = True                # reset next turn
            useSkill(10, 'random')       # Laser Arrows (902172): Magic damage (5x, MAG) to all enemies
            return
        
        if not berry:
            berry  = True                # reset next turn
            useSkill(4, 'random')        # Divine Lance (902166): Physical light damage (50x, ATK) to one enemy (+100% accuracy)
            return
        
        if not peach:
            peach  = True                # reset next turn
            useSkill(8, 'random')        # Aqua Sweep (902170): Physical water damage (20x, ATK) to all enemies (+100% accuracy)
            return
        
        if not olive:
            olive  = True                # reset next turn
            useSkill(6, 'random')        # Tactical Laser (902168): Magic damage (80x, MAG) to one enemy
            return
        
    
    if ramen:
        if not apple:
            apple  = True                # reset next turn
            useSkill(10, 'random')       # Laser Arrows (902172): Magic damage (5x, MAG) to all enemies
            return
        
        if not berry:
            berry  = True                # reset next turn
            useSkill(5, 'random')        # Aqua Lance (902167): Physical water damage (50x, ATK) to one enemy (+100% accuracy)
            return
        
        if not peach:
            peach  = True                # reset next turn
            useSkill(7, 'random')        # Divine Sweep (902169): Physical light damage (20x, ATK) to all enemies (+100% accuracy)
            return
        
        if not olive:
            olive  = True                # reset next turn
            useSkill(9, 'random')        # Laser Blast (902171): Magic damage (20x, MAG) to all enemies
            return
        
        if not mango:
            mango  = True                # reset next turn
            useSkill(6, 'random')        # Tactical Laser (902168): Magic damage (80x, MAG) to one enemy
            return
        
    

if green == 2:
    if not ramen:
        if not apple:
            apple  = True                # reset next turn
            useSkill(10, 'random')       # Laser Arrows (902172): Magic damage (5x, MAG) to all enemies
            return
        
        if not berry:
            berry  = True                # reset next turn
            useSkill(5, 'random')        # Aqua Lance (902167): Physical water damage (50x, ATK) to one enemy (+100% accuracy)
            return
        
        if not peach:
            peach  = True                # reset next turn
            useSkill(7, 'random')        # Divine Sweep (902169): Physical light damage (20x, ATK) to all enemies (+100% accuracy)
            return
        
        if not olive:
            olive  = True                # reset next turn
            useSkill(9, 'random')        # Laser Blast (902171): Magic damage (20x, MAG) to all enemies
            return
        
        if not mango:
            mango  = True                # reset next turn
            useSkill(21, 'random')       # Curse of Centaurus (902180): Inflict Paralyze (100%) on all enemies
            return
        
    
    if ramen:
        if not apple:
            apple  = True                # reset next turn
            useSkill(10, 'random')       # Laser Arrows (902172): Magic damage (5x, MAG) to all enemies
            return
        
        if not berry:
            berry  = True                # reset next turn
            useSkill(4, 'random')        # Divine Lance (902166): Physical light damage (50x, ATK) to one enemy (+100% accuracy)
            return
        
        if not peach:
            peach  = True                # reset next turn
            useSkill(8, 'random')        # Aqua Sweep (902170): Physical water damage (20x, ATK) to all enemies (+100% accuracy)
            return
        
        if not olive:
            olive  = True                # reset next turn
            useSkill(9, 'random')        # Laser Blast (902171): Magic damage (20x, MAG) to all enemies
            return
        
        if not mango:
            mango  = True                # reset next turn
            useSkill(6, 'random')        # Tactical Laser (902168): Magic damage (80x, MAG) to one enemy
            return
        
        if not lemon:
            lemon  = True                # reset next turn
            useSkill(21, 'random')       # Curse of Centaurus (902180): Inflict Paralyze (100%) on all enemies
            return
        
    

if green == 3:
    if not apple:
        apple  = True                    # reset next turn
        useSkill(10, 'random')           # Laser Arrows (902172): Magic damage (5x, MAG) to all enemies
        return
    
    if not berry:
        green  = 0                       
        berry  = True                    # reset next turn
        useSkill(11, 'random')           # Flood of Centaurus (902173): Physical water and light damage (50x, ATK) to one enemy (+100% accuracy)
        return
    

if random() <= 0.30 and white <= 2:
    white += 1                           
    useSkill(2, 'random')                # Laser Lance (902176): Magic damage (5x, MAG) to one enemy
    return

if random() <= 0.20 and black <= 8:
    black += 1                           
    useSkill(3, 'random')                # Laser Shower (902177): Magic damage (5x, MAG) to all enemies
    return

else:
    useSkill(1, 'random')                # Power Stomp (902175): Physical damage (50x, ATK) to one enemy (+100% accuracy)
    return

