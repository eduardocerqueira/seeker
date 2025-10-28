#date: 2025-10-28T17:10:24Z
#url: https://api.github.com/gists/a774170a69cb481f24168bb3439a3022
#owner: https://api.github.com/users/mix0073

from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False)
    device_id = Column(String(100), unique=True, nullable=False)
    auth_token = "**********"=True, nullable=False)

    # Связи
    cube_states = relationship("CubeState", back_populates="player")
    inventory = relationship("PlayerInventory", back_populates="player")


class CubeState(Base):
    __tablename__ = "cube_states"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    cube_position_id = Column(Integer, nullable=False)

    # Позиции и вращения
    rotation_x = Column(Float, default=0.0)
    rotation_y = Column(Float, default=0.0)
    rotation_z = Column(Float, default=0.0)
    initial_x = Column(Integer)
    initial_y = Column(Integer)
    initial_z = Column(Integer)

    # Грани куба (6 сторон)
    front_element_id = Column(Integer)
    front_is_unlocked = Column(Boolean, default=False)
    front_is_active = Column(Boolean, default=False)
    front_current_amount = Column(Float, default=0.0)

    back_element_id = Column(Integer)
    back_is_unlocked = Column(Boolean, default=False)
    back_is_active = Column(Boolean, default=False)
    back_current_amount = Column(Float, default=0.0)

    left_element_id = Column(Integer)
    left_is_unlocked = Column(Boolean, default=False)
    left_is_active = Column(Boolean, default=False)
    left_current_amount = Column(Float, default=0.0)

    right_element_id = Column(Integer)
    right_is_unlocked = Column(Boolean, default=False)
    right_is_active = Column(Boolean, default=False)
    right_current_amount = Column(Float, default=0.0)

    top_element_id = Column(Integer)
    top_is_unlocked = Column(Boolean, default=False)
    top_is_active = Column(Boolean, default=False)
    top_current_amount = Column(Float, default=0.0)

    bottom_element_id = Column(Integer)
    bottom_is_unlocked = Column(Boolean, default=False)
    bottom_is_active = Column(Boolean, default=False)
    bottom_current_amount = Column(Float, default=0.0)

    # Связи
    player = relationship("Player", back_populates="cube_states")


class Element(Base):
    __tablename__ = "elements"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(3), nullable=False)
    name = Column(String(50), nullable=False)
    base_production_rate = Column(Float, default=0.1)
    base_storage_capacity = Column(Float, default=100.0)
    is_initial = Column(Boolean, default=False)


class ElementUnlockRequirement(Base):
    __tablename__ = "element_unlock_requirements"

    id = Column(Integer, primary_key=True)
    element_id = Column(Integer, nullable=False)
    required_element_id = Column(Integer, nullable=False)
    required_amount = Column(Float, default=0.0)


class PlayerInventory(Base):
    __tablename__ = "player_inventory"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    element_id = Column(Integer, nullable=False)
    amount = Column(Float, default=0.0)

    # Связи
    player = relationship("Player", back_populates="inventory")s="inventory")