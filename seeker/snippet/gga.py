#date: 2025-01-31T16:41:37Z
#url: https://api.github.com/gists/55f4cc650775f7bf1645abe3bf811d3e
#owner: https://api.github.com/users/markusand

from enum import Enum
from dataclasses import dataclass
from .nmea import NMEA

class FixType(Enum):
    """RTK fix type"""
    NOT_VALID = "0"
    GPS_FIX = "1"
    DIFF_GPS_FIX = "2"
    NOT_APPLICABLE = "3"
    RTK_FIX = "4"
    RTK_FLOAT = "5"
    INS = "6"


def ddmm_to_decimal(ddmm: float, direction: str) -> float:
    """Convert coordinates from ddmm to decimal degrees"""
    degrees = int(ddmm // 100)
    minutes = ddmm % 100
    mult = -1 if direction in ("W", "S") else 1
    return mult * (degrees + (minutes / 60))

    
@dataclass(frozen=True)
class GGA(NMEA):
    """NMEA GGA message"""

    sentence: str
    utc: float  # UTC time seconds
    _lat: float  # Latitude in ddmm format
    lat_hemisphere: str  # N or S
    _lon: float  # Longitude in ddmm format
    lon_hemisphere: str  # E or W
    fix: FixType  # Fix type
    satellites_in_use: int  # Number of satellites in use
    hdop: float  # Horizontal dilution of precision
    alt: float  # Altitude relative to mean sea level
    alt_unit: str  # Altitude unit (meters)
    geoid_separation: float  # Geoid separation height
    geoid_separation_unit: str  # Geoid separation unit (meters)
    age_differential: int  # Approximate age of differential data (last GPS MSM message received)
    reference_station: str  # Reference station ID

    @property
    def lat(self) -> float:
        """Latitude in decimal degrees"""
        return ddmm_to_decimal(self._lat, self.lat_hemisphere)

    @property
    def lon(self) -> float:
        """Longitude in decimal degrees"""
        return ddmm_to_decimal(self._lon, self.lon_hemisphere)
