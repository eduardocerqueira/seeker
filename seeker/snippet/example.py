#date: 2023-07-12T16:49:02Z
#url: https://api.github.com/gists/68c5f2e1a4ad718cc21105a571ed8dbb
#owner: https://api.github.com/users/bianchimro

from django.db.models import OuterRef

weapons = Weapon.objects.filter(unit__player_id=OuterRef('id'))
units = Unit.objects.filter(player_id=OuterRef('id'))

qs = Player.objects.annotate(weapon_count=SubqueryCount(weapons),
                             rarity_sum=SubquerySum(units, 'rarity'))