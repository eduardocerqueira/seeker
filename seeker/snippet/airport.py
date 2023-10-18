#date: 2023-10-18T17:05:19Z
#url: https://api.github.com/gists/a9dd8e9849437b076d113fafe8f58b60
#owner: https://api.github.com/users/alexandros173

from sqlalchemy import Column, Integer, String, Float

from database import Base


class Airport(Base):
    __tablename__ = 'airport'

    iata_code = Column(String, primary_key=True)
    name = Column(String(250))
    icao_code = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    city = Column(String)