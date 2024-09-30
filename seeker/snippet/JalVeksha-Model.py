#date: 2024-09-30T17:03:31Z
#url: https://api.github.com/gists/22b6ed7c4bd50be30b318b2b48cc3832
#owner: https://api.github.com/users/manyajsingh

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class WaterQuality(BaseModel):
    location: str = Field(..., description="Location along the Ganga River where the sample is taken")
    date: datetime = Field(..., description="Date of the water quality measurement")
    dissolved_oxygen: float = Field(..., gt=0, description="Dissolved Oxygen level in mg/L")
    biochemical_oxygen_demand: float = Field(..., gt=0, description="Biochemical Oxygen Demand in mg/L")
    nitrate_level: float = Field(..., gt=0, description="Nitrate level in mg/L")
    phosphorus_level: float = Field(..., gt=0, description="Phosphorus level in mg/L")
    fecal_coliform: int = Field(..., gt=0, description="Fecal coliform count per 100 mL")
    turbidity: float = Field(..., gt=0, description="Turbidity in NTU (Nephelometric Turbidity Units)")
    temperature: float = Field(..., description="Water temperature in degrees Celsius")
    status: str = Field(..., description="Overall status of water quality (e.g., Safe, Polluted)")

class BiodiversityIndex(BaseModel):
    location: str = Field(..., description="Location along the Ganga River where the biodiversity index is assessed")
    date: datetime = Field(..., description="Date of the biodiversity assessment")
    fish_population: int = Field(..., gt=0, description="Number of fish species identified")
    macroinvertebrate_population: int = Field(..., gt=0, description="Number of macroinvertebrate species identified")
    aquatic_plant_diversity: int = Field(..., gt=0, description="Number of aquatic plant species identified")
    habitat_quality: str = Field(..., description="Assessment of habitat quality (e.g., Good, Fair, Poor)")

class PollutionLevel(BaseModel):
    location: str = Field(..., description="Location along the Ganga River where pollution level is measured")
    date: datetime = Field(..., description="Date of the pollution level measurement")
    industrial_discharge: float = Field(..., gt=0, description="Industrial discharge in cubic meters")
    sewage_discharge: float = Field(..., gt=0, description="Sewage discharge in cubic meters")
    solid_waste: float = Field(..., gt=0, description="Solid waste level in kg")
    chemical_pollutants: str = Field(..., description="List of identified chemical pollutants")
    pollution_status: str = Field(..., description="Overall pollution status (e.g., High, Moderate, Low)")

# Example usage
if __name__ == "__main__":
    water_quality_example = WaterQuality(
        location="Varanasi",
        date=datetime.now(),
        dissolved_oxygen=5.5,
        biochemical_oxygen_demand=3.0,
        nitrate_level=10.0,
        phosphorus_level=0.5,
        fecal_coliform=200,
        turbidity=15.0,
        temperature=25.0,
        status="Polluted"
    )

    biodiversity_index_example = BiodiversityIndex(
        location="Varanasi",
        date=datetime.now(),
        fish_population=12,
        macroinvertebrate_population=25,
        aquatic_plant_diversity=8,
        habitat_quality="Fair"
    )

    pollution_level_example = PollutionLevel(
        location="Varanasi",
        date=datetime.now(),
        industrial_discharge=100.0,
        sewage_discharge=50.0,
        solid_waste=20.0,
        chemical_pollutants="Lead, Mercury, Arsenic",
        pollution_status="High"
    )

    print(water_quality_example)
    print(biodiversity_index_example)
    print(pollution_level_example)
