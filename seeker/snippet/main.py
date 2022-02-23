#date: 2022-02-23T17:12:52Z
#url: https://api.github.com/gists/ceed406ef61d3d46b61d05085111a0f8
#owner: https://api.github.com/users/mypy-play

from enum import Enum

class VehicleClass(Enum):
    ELECTRIC_BICYCLE = "e-bike"
    BICYCLE = "bike"
    ELECTRIC_SCOOTER = "scooter"

def vehicle_distance(vehicle_class: VehicleClass):
    if vehicle_class == VehicleClass.ELECTRIC_BICYCLE.value:
        return 10
    elif vehicle_class == VehicleClass.BICYCLE.value:
        return 5
    elif vehicle_class == VehicleClass.ELECTRIC_SCOOTER.value:
        return 8

def vehicle_distance_2(vehicle_class: VehicleClass):
    match vehicle_class:
        case BICYCLE.value:
            return 5
        case ELECTRIC_BICYCLE.value:
            return 10
        case ELECTRIC_SCOOTER.value:
            return 8