#date: 2022-08-26T16:53:36Z
#url: https://api.github.com/gists/8c1d1eadb00644812c73c42b798ff4fd
#owner: https://api.github.com/users/matteoferla

import operator
import pyrosetta
import pyrosetta_help as ph
import requests
from typing import Dict, Union, Tuple
from types import ModuleType
prc: ModuleType = pyrosetta.rosetta.core

logger = ph.configure_logger()
pyrosetta.init(extra_options=ph.make_option_string(no_optH=False))
prn: ModuleType = pyrosetta.rosetta.numeric


    
def shift_positive(pose) -> None:
    """Make all coordinated positive"""
    # these are not copies (no get)
    xyzs: List[prn.xyzVector_double_t] = [residue.xyz(ai) for residue in pose.residues for ai in range(1, residue.natoms()+1)]
    for axis in 'xyz':
        m_axis = min(map(operator.attrgetter(axis), xyzs))
        [setattr(xyz, axis, getattr(xyz, axis) - m_axis ) for xyz in xyzs]

def encode(seen_pose: pyrosetta.Pose, unseen_pose: pyrosetta.Pose) -> pyrosetta.Pose:
    encrypted = seen_pose.clone()
    for seen, unseen in zip(encrypted.residues, unseen_pose.residues):
        for ai in range(1,100):
            if seen.natoms() > ai and unseen.natoms() > ai:
                seen_xyz = seen.xyz(ai)
                unseen_xyz = unseen.xyz(ai)
                for axis in 'xyz':
                    setattr(seen_xyz, axis, round(getattr(seen_xyz, axis), 0) + round(getattr(unseen_xyz, axis), 0)/1e3)
            elif seen.natoms() > ai:
                seen_xyz = seem.xyz(ai)
                for axis in 'xyz':
                    setattr(seen_xyz, axis, round(getattr(seen_xyz, axis), 0))
            elif unseen.natoms() > ai:
                # atomic info lost
                pass
            else:
                break
    return encrypted

def decode(encrypted_pose: pyrosetta.Pose) -> Tuple[pyrosetta.Pose, pyrosetta.Pose]:
    seen_pose = encrypted.clone()
    unseen_pose = encrypted.clone()
    for seen, unseen in zip(seen_pose.residues, unseen_pose.residues):
        for ai in range(1, seen.natoms()):
            seen_xyz = seen.xyz(ai)
            unseen_xyz = unseen.xyz(ai)
            for axis in 'xyz':
                encrypted_value = getattr(seen_xyz, axis)
                setattr(seen_xyz, axis, round(encrypted_value, 0))
                # so rubbish way, much wow
                setattr(unseen_xyz, axis, (encrypted_value - round(encrypted_value, 0)) * 1e3)
    return [seen_pose, unseen_pose]

helix: pyrosetta.Pose = pyrosetta.pose_from_sequence('ELVISISALIVE')
ph.make_alpha_helical(helix)
shift_positive(helix)
sheet: pyrosetta.Pose = pyrosetta.pose_from_sequence('ELVISISALIVE')
ph.make_sheet(sheet)
shift_positive(sheet)

encrypted = encode(helix, sheet)
helix2, sheet2 = decode(encrypted)

import nglview as nv
from ipywidgets import TwoByTwoLayout

TwoByTwoLayout(top_left=nv.show_rosetta(helix),
               top_right=nv.show_rosetta(sheet),
               bottom_left=nv.show_rosetta(helix2),
               bottom_right=nv.show_rosetta(sheet2),
               )