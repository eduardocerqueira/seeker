#date: 2024-04-18T17:07:19Z
#url: https://api.github.com/gists/9b5407ae34c622c30ee687a9332df04b
#owner: https://api.github.com/users/aEnigmatic

##
# Mission 'Centaur of Unknown Origin' (95280101)
#
# Monsters
#  * Mechiron (900011396)
#
##

##
# Morale Rules - Before Battle
##
#
# Begin battle with 5% additional morale for every 3 EX levels included in party.Begin battle with 5% additional morale if all units in party are at EX +3.
# All units in party are EX+3
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
# Monster  Mechiron (900011396)
# Race     Beast, Human, Machina
# Level    99
# Actions  10-10
#
#
# Stats
#        HP        200000000
#        MP           100000
#        ATK             300
#        DEF            4000
#        MAG             300
#        SPR            4000
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
#        Non-Elemental    50%
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
#  TFA Passive 30/60 Tier 1 (920032) [Passive]
#
#  Increase DEF and SPR by 60% and ATK and MAG by 30%
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

###
# AI
###
if not var_70:
    var_70 = True                        # unknown flag type  (70)
    white  = 0                           
    wait()                               
    return

if random() <= 0.30 and white <= 2:
    white += 1                           
    useSkill(2, 'random')                # Laser Lance (902176): Magic damage (5x, MAG) to one enemy
    return

if random() <= 0.50:
    useSkill(1, 'random')                # Power Stomp (902175): Physical damage (50x, ATK) to one enemy (+100% accuracy)
    return

else:
    useSkill(3, 'random')                # Laser Shower (902177): Magic damage (5x, MAG) to all enemies
    return

