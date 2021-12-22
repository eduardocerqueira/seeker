#date: 2021-12-22T17:18:48Z
#url: https://api.github.com/gists/3f1ef8298351b78d3254d5740f35538d
#owner: https://api.github.com/users/nda86

from dataclasses import dataclass, field


@dataclass
class VehicleSpecificationsForm:
    vehicle_body_length: int
    vehicle_body_height: int
    vehicle_body_width: int
    vehicle_volume_capacity: float = field(init=False)

    def __post_init__(self):
        self.vehicle_volume_capacity = (
            self.vehicle_body_length * self.vehicle_body_width * self.vehicle_body_height
        ) / 1000000  # см3 в м3


vsf = VehicleSpecificationsForm(vehicle_body_width=195, vehicle_body_height=200, vehicle_body_length=350)
print(vsf.vehicle_volume_capacity)
