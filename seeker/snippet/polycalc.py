#date: 2024-11-11T17:06:52Z
#url: https://api.github.com/gists/5beaeadabee0d7d6a36fee82556ce569
#owner: https://api.github.com/users/yhs0602

import math
from enum import Enum
from typing import Optional, Tuple


def normal_round(n):
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)


class UnitType(Enum):
    WARRIOR = (10, 2, 2, 1)
    RIDER = (10, 2, 1, 1)
    DEFENDER = (15, 1, 3, 1)
    ARCHER = (10, 2, 1, 2)
    SHAMAN = (10, 1, 1, 1)
    SWORDSMAN = (15, 3, 3, 1)
    DAGGER = (10, 2, 2, 1)
    KNIGHT = (10, 3.5, 1, 1)
    TRIDENT = (15, 3, 1, 2)
    DOOMUX = (20, 4, 2, 1)
    CATAPULT = (10, 4, 0, 3)
    EXIDA = (10, 3, 1, 3)
    GIANT = (40, 5, 4, 1)


class UnitInfo:
    def __init__(
            self,
            unit_type: UnitType,
            promoted: bool = False,
            current_hp: Optional[int] = None,
    ):
        self.unit_type = unit_type
        self.max_hp = unit_type.value[0]
        self.promoted = promoted
        if self.promoted:
            self.max_hp += 5
        if current_hp is None:
            self.current_hp = self.max_hp
        else:
            self.current_hp = current_hp

    def __str__(self):
        return f"{self.unit_type.name}({self.promoted}, {self.current_hp})"

    def __repr__(self):
        return str(self)


def damage(
        attacker: UnitInfo, defender: UnitInfo, defence_bonus: float = 1
) -> Tuple[float, float]:
    attack_value = attacker.unit_type.value[1] * (attacker.current_hp / attacker.max_hp)
    defence_value = (defender.unit_type.value[2] * defence_bonus) * (
            defender.current_hp / defender.max_hp
    )
    total_value = attack_value + defence_value
    final_damage = (attack_value / total_value) * attacker.unit_type.value[1] * 4.5
    revenge_damage = (defence_value / total_value) * defender.unit_type.value[2] * 4.5
    return final_damage, revenge_damage


def simulate_damage(
        attacker: UnitInfo,
        defender: UnitInfo,
        defence_bonus: float = 1,
        revenge: bool = True,
) -> Tuple[UnitInfo, UnitInfo]:
    receive_damage, revenge_damage = damage(attacker, defender, defence_bonus)
    receive_damage = normal_round(receive_damage)
    revenge_damage = normal_round(revenge_damage)
    if not revenge:
        revenge_damage = 0
    new_attacker = UnitInfo(
        attacker.unit_type,
        attacker.promoted,
        attacker.current_hp - revenge_damage,
    )
    new_defender = UnitInfo(
        defender.unit_type,
        defender.promoted,
        defender.current_hp - receive_damage,
    )

    return new_attacker, new_defender


def main():
    giant = UnitInfo(UnitType.GIANT, current_hp=29)
    archer = UnitInfo(UnitType.ARCHER)
    print(giant, archer)
    archer, giant = simulate_damage(archer, giant, revenge=False)
    print(giant, archer)
    archer, giant = simulate_damage(archer, giant, revenge=True)
    print(giant, archer)
    archer, giant = simulate_damage(archer, giant, revenge=True)
    print(giant, archer)
    archer, giant = simulate_damage(archer, giant, revenge=False)
    print(giant, archer)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
