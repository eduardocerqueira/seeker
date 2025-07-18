#date: 2025-07-18T17:15:58Z
#url: https://api.github.com/gists/b30d4c63663d205fad16f5c2d5a2e7f8
#owner: https://api.github.com/users/SKCrawford

from enum import Enum
from typing import Literal, Optional, TypeAlias

# Types

AttrScaling: TypeAlias = Literal["S", "A", "B", "C", "D", "E", "?"] # "?" = scaling is unknown/irrelevant
AttrValue: TypeAlias = int


class AttrNames(Enum):
    """Valid attribute names."""

    VIGOR = "VIG"
    MIND = "MND"
    ENDURANCE = "END"
    STRENGTH = "STR"
    DEXTERITY = "DEX"
    INTELLIGENCE = "INT"
    FAITH = "FAI"
    ARCANE = "ARC"


class AttrIcons(Enum):
    """Basic label icons for attributes."""

    FAVORED = "\U00002B50" # star
    VIG = "\U00002764" # red heart
    MND = "\U0001f4A7" # blue tear drop
    END = "\U0001f3c3" # person running
    STR = "\U0001f4aa" # flexing bicep
    DEX = "\U00002694" # crossed swords
    INT = "\U0001f393" # mortarboard
    FAI = "\U0001f506" # sun
    ARC = "\U0001f4ab" # shooting star


class NightfarerNames(Enum):
    """Valid nightfarer names."""

    WYLDER = "WYLDER"
    GUARDIAN = "GUARDIAN"
    IRONEYE = "IRONEYE"
    DUCHESS = "DUCHESS"
    RAIDER = "RAIDER"
    REVENANT = "REVENANT"
    RECLUSE = "RECLUSE"
    EXECUTOR = "EXECUTOR"


# Data

# based on level 15; ripped from fextra
# these may change via the demon merchant
ATTR_DEFAULTS_BY_NF: dict[NightfarerNames, list[tuple[AttrScaling, AttrValue]]] = {
    #                                     STR        DEX        INT        FAI        ARC
    NightfarerNames.WYLDER.value:     [("A", 50), ("B", 40), ("C", 15), ("C", 15), ("C", 10)],
    NightfarerNames.GUARDIAN.value:   [("B", 39), ("C", 29), ("D", 10), ("C", 21), ("C", 10)],
    NightfarerNames.IRONEYE.value:    [("C", 19), ("A", 57), ("D",  7), ("D", 13), ("B", 13)],
    NightfarerNames.DUCHESS.value:    [("D", 11), ("B", 41), ("A", 42), ("B", 27), ("C", 11)],
    NightfarerNames.RAIDER.value:     [("S", 68), ("C", 19), ("D",  3), ("D", 12), ("C", 10)],
    NightfarerNames.REVENANT.value:   [("C", 21), ("C", 21), ("B", 30), ("S", 51), ("B", 12)],
    NightfarerNames.RECLUSE.value:    [("D", 12), ("C", 19), ("S", 51), ("S", 51), ("C", 10)],
    NightfarerNames.EXECUTOR.value:   [("C", 25), ("S", 63), ("D",  8), ("D",  6), ("S", 28)],
}


# Structures

class Attr:
    """The representation of an attribute, such as strength (STR) or intelligence (INT)."""

    def __init__(self, name: AttrNames, scaling: AttrScaling = "?", value: AttrValue = 0):
        self.name: AttrNames = name
        self.scaling: AttrScaling = scaling
        self.value: AttrValue = value


class Nightfarer:
    """The representation of a nightfarer."""

    def __init__(self, name: NightfarerNames, attr_defaults: list[Attr]):
        self.name: NightfarerNames = name

        self.base_str = Attr(AttrNames.STRENGTH.value,        scaling=attr_defaults[0][0], value=attr_defaults[0][1])
        self.base_dex = Attr(AttrNames.DEXTERITY.value,       scaling=attr_defaults[1][0], value=attr_defaults[1][1])
        self.base_int = Attr(AttrNames.INTELLIGENCE.value,    scaling=attr_defaults[2][0], value=attr_defaults[2][1])
        self.base_fai = Attr(AttrNames.FAITH.value,           scaling=attr_defaults[3][0], value=attr_defaults[3][1])
        self.base_arc = Attr(AttrNames.ARCANE.value,          scaling=attr_defaults[4][0], value=attr_defaults[4][1])
        
        self.base_attrs: list[Attr] = [self.base_str, self.base_dex, self.base_int, self.base_fai, self.base_arc]
        self.base_attrs.sort(reverse=True, key=lambda attr: attr.value)

    def best_attrs(self, how_many: int = 2) -> list[Attr]:
        """Get the nightfarer's n best attributes where n is `how_many`."""

        return self.base_attrs[0:how_many]


NIGHTFARERS: dict[NightfarerNames, Nightfarer] = {
    NightfarerNames.WYLDER.value:   Nightfarer(NightfarerNames.WYLDER.value,    ATTR_DEFAULTS_BY_NF[NightfarerNames.WYLDER.value]),
    NightfarerNames.GUARDIAN.value: Nightfarer(NightfarerNames.GUARDIAN.value,  ATTR_DEFAULTS_BY_NF[NightfarerNames.GUARDIAN.value]),
    NightfarerNames.IRONEYE.value:  Nightfarer(NightfarerNames.IRONEYE.value,   ATTR_DEFAULTS_BY_NF[NightfarerNames.IRONEYE.value]),
    NightfarerNames.DUCHESS.value:  Nightfarer(NightfarerNames.DUCHESS.value,   ATTR_DEFAULTS_BY_NF[NightfarerNames.DUCHESS.value]),
    NightfarerNames.RAIDER.value:   Nightfarer(NightfarerNames.RAIDER.value,    ATTR_DEFAULTS_BY_NF[NightfarerNames.RAIDER.value]),
    NightfarerNames.REVENANT.value: Nightfarer(NightfarerNames.REVENANT.value,  ATTR_DEFAULTS_BY_NF[NightfarerNames.REVENANT.value]),
    NightfarerNames.RECLUSE.value:  Nightfarer(NightfarerNames.RECLUSE.value,   ATTR_DEFAULTS_BY_NF[NightfarerNames.RECLUSE.value]),
    NightfarerNames.EXECUTOR.value: Nightfarer(NightfarerNames.EXECUTOR.value,  ATTR_DEFAULTS_BY_NF[NightfarerNames.EXECUTOR.value]),
}


def get_armament_best_attrs(grabbable_spec: dict, how_many: int = 2) -> list[Attr]:
    """Get an armament's n best attributes where n is `how_many`."""

    strength =      Attr(AttrNames.STRENGTH.value,      value=grabbable_spec[AttrNames.STRENGTH.value])
    dexterity =     Attr(AttrNames.DEXTERITY.value,     value=grabbable_spec[AttrNames.DEXTERITY.value])
    intelligence =  Attr(AttrNames.INTELLIGENCE.value,  value=grabbable_spec[AttrNames.INTELLIGENCE.value])
    faith =         Attr(AttrNames.FAITH.value,         value=grabbable_spec[AttrNames.FAITH.value])
    arcane =        Attr(AttrNames.ARCANE.value,        value=grabbable_spec[AttrNames.ARCANE.value])

    all_attrs: list[Attr] = [strength, dexterity, intelligence, faith, arcane]
    all_attrs.sort(reverse=True, key=lambda attr : attr.value)

    return all_attrs[0:how_many]


def get_armament_attrs(grabbable_spec: dict) -> list[Attr]:
    """Get an armament's attributes."""

    strength =      Attr(AttrNames.STRENGTH.value,      value=grabbable_spec[AttrNames.STRENGTH.value])
    dexterity =     Attr(AttrNames.DEXTERITY.value,     value=grabbable_spec[AttrNames.DEXTERITY.value])
    intelligence =  Attr(AttrNames.INTELLIGENCE.value,  value=grabbable_spec[AttrNames.INTELLIGENCE.value])
    faith =         Attr(AttrNames.FAITH.value,         value=grabbable_spec[AttrNames.FAITH.value])
    arcane =        Attr(AttrNames.ARCANE.value,        value=grabbable_spec[AttrNames.ARCANE.value])
    return [strength, dexterity, intelligence, faith, arcane]


def is_favored_attr(armament_attr: Attr, nightfarer: Nightfarer, how_many: int = 2) -> bool:
    """Return True if an attribute is one of a nightfarer's n best attributes where n is `how_many`. 
    Otherwise, return False.
    """

    return any(armament_attr.name == attr.name for attr in nightfarer.best_attrs(how_many=how_many))


# Label Generation

def attr_label(attr: Attr, is_favored: bool = False) -> str:
    """Create a label for an attribute. Add an icon if the attribute is favored by the nightfarer."""

    attr_value: str = str(attr.value) if attr.value > 0 else "--"
    if is_favored:
        return f"{AttrIcons.FAVORED}{attr.name} {attr_value}"
    return f"{attr.name} {attr_value}"


def nightfarer_label(nightfarer: Nightfarer) -> str:
    """Create a label for a nightfarer's two best attributes."""

    [nf_attr_a, nf_attr_b] = nightfarer.best_attrs(how_many=2)
    return f"{attr_label(nf_attr_a)}/{attr_label(nf_attr_b)}"


def armament_label(nightfarer: Nightfarer, grabbable_spec: dict) -> str:
    """Create a label for an armament and list its attributes."""

    arm_attr_labels: list[str] = []
    for arm_attr in get_armament_attrs(grabbable_spec):
        is_favored: bool = is_favored_attr(arm_attr, nightfarer)
        arm_attr_label: str = attr_label(arm_attr, is_favored=is_favored)
        arm_attr_labels.append(arm_attr_label)
    
    return f"{arm_attr_labels[0]}/{arm_attr_labels[1]}/{arm_attr_labels[2]}/{arm_attr_labels[3]}/{arm_attr_labels[4]}"


def get_basic_label_icons(character_spec: dict, grabbable_spec: dict) -> list[str]:
    icons: list[str] = []

    if grabbable_spec["type"] == "armament":
        for arm_attr in get_armament_best_attrs(grabbable_spec, how_many=4):
            if arm_attr.value:
                icons.append(AttrIcons[arm_attr.name].value)
    return icons


def get_advanced_label_text(character_spec: dict, grabbable_spec: dict) -> str:
    nightfarer: Nightfarer = NIGHTFARERS[character_spec["name"]]

    if grabbable_spec["type"] == "armament":
        return f"{armament_label(nightfarer, grabbable_spec)} ({nightfarer_label(nightfarer)})"
    return ""