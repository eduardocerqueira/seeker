#date: 2024-03-22T17:00:11Z
#url: https://api.github.com/gists/cdbe3a0a5990dea6bd4df720940282d1
#owner: https://api.github.com/users/CodingDino


WeaponDamage = 2
AttackStat = 15
PotionFactor = 2
ResistanceRating = 1.5
ArmourRating = 2
HitPoints = 100

AttackDamage = WeaponDamage * AttackStat / 10
IncomingDamage = AttackDamage ** PotionFactor
DamageAfterResistance = IncomingDamage * ResistanceRating
EffectiveDamage = DamageAfterResistance - ArmourRating
RemainingHP = HitPoints - EffectiveDamage

RemainingHP2 = HitPoints - ((((WeaponDamage * AttackStat / 10) ** PotionFactor) * ResistanceRating) - ArmourRating)

print(RemainingHP)
print(RemainingHP2)

