#date: 2022-10-17T17:17:54Z
#url: https://api.github.com/gists/2860cf864ed1658ceec466bfb599e3fe
#owner: https://api.github.com/users/Yoshanuikabundi

from typing import Literal, Tuple, Union, Iterable, Dict, Optional, List
import uuid
from io import StringIO

from IPython.display import SVG
from openff.toolkit.topology import FrozenMolecule

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdDepictor import Compute2DCoords

import numpy as np
from numpy.typing import ArrayLike, NDArray

import nglview
from nglview.base_adaptor import Structure, Trajectory

from pdbfixer import PDBFixer

from openff.units import Quantity, unit
from openff.units.openmm import from_openmm

from openff.toolkit import Molecule, ForceField, Topology
from openff.toolkit.typing.engines.smirnoff.parameters import BondType, ParameterType
from openff.toolkit.utils.exceptions import ParameterLookupError

DEFAULT_WIDTH = 300
DEFAULT_HEIGHT = 200

Color = Iterable[float]
BondIndices = Tuple[int, int]

MOLECULE_DEFAULT_REPS = [
    dict(type="licorice", params=dict(radius=0.25, multipleBond=True))
]


CUBE = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
"""A regular square prism with image distance 1.0 and volume 1.0."""

RHOMBIC_DODECAHEDRON = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, np.sqrt(2.0) / 2.0],
    ]
)
"""A rhombic dodecahedron with image distance 1.0 and volume ~0.71."""


def draw_molecule(
    molecule: Union[Molecule, Chem.Mol],
    image_width: int = DEFAULT_WIDTH,
    image_height: int = DEFAULT_HEIGHT,
    highlight_atoms: Optional[Union[List[int], Dict[int, Color]]] = None,
    highlight_bonds: Optional[
        Union[List[BondIndices], Dict[BondIndices, Color]]
    ] = None,
    atom_notes: Optional[Dict[int, str]] = None,
    bond_notes: Optional[Dict[BondIndices, str]] = None,
    emphasize_atoms: Optional[List[int]] = None,
    explicit_hydrogens: Optional[bool] = None,
    color_by_element: Optional[bool] = None,
) -> SVG:
    """Draw a molecule

    Parameters
    ==========

    molecule
        The molecule to draw.
    image_width
        The width of the resulting image in pixels.
    image_height
        The height of the resulting image in pixels.
    highlight_atoms
        A list of atom indices to highlight, or a map from indices to colors.
        Colors should be given as triplets of floats between 0.0 and 1.0.
    highlight_bonds
        A list of pairs of atom indices indicating bonds to highlight, or a map
        from index pairs to colors. Colors should be given as triplets of floats
        between 0.0 and 1.0.
    atom_notes
        A map from atom indices to a string that should be printed near the
        atom.
    bond_notes
        A map from atom index pairs to a string that should be printed near the
        bond.
    emphasize_atoms
        A list of atom indices to emphasize by drawing other atoms (and their
        bonds) in light grey.
    explicit_hydrogens
        If ``False``, allow uncharged monovalent hydrogens to be hidden. If
        ``True``, make all hydrogens explicit. If ``None``, defer to the
        provided molecule.
    color_by_element
        If True, color heteroatoms according to their element; if False, color
        atoms and bonds monochromatically. By default, uses black and white when
        highlight_atoms or highlight_bonds is provided, and color otherwise.

    Raises
    ======

    KeyError
        When an atom or bond in highlight_atoms or highlight_bonds is missing
        from the image, including when it is present in the molecule but hidden.
    """
    # We're working in RDKit
    if isinstance(molecule, FrozenMolecule):
        rdmol = molecule.to_rdkit()
    else:
        rdmol = Chem.mol(molecule)

    # Process color_by_element argument
    if color_by_element is None:
        color_by_element = highlight_atoms is None and highlight_bonds is None

    if color_by_element:
        set_atom_palette = lambda draw_options: draw_options.useDefaultAtomPalette()
    else:
        set_atom_palette = lambda draw_options: draw_options.useBWAtomPalette()

    # Process explicit_hydrogens argument
    # If we need to remove atoms, create a map from the original indices to the
    # new ones.
    if explicit_hydrogens is None:
        idx_map = {i: i for i in range(rdmol.GetNumAtoms())}
    elif explicit_hydrogens:
        idx_map = {i: i for i in range(rdmol.GetNumAtoms())}
        rdmol = Chem.AddHs(rdmol, explicitOnly=True)
    else:
        idx_map = {
            old: new
            for new, old in enumerate(
                a.GetIdx()
                for a in rdmol.GetAtoms()
                if a.GetAtomicNum() != 1 and a.GetMass() != 1
            )
        }
        rdmol = Chem.RemoveHs(rdmol, updateExplicitCount=True)

    # Process highlight_atoms argument for highlightAtoms and highlightAtomColors
    # highlightAtoms takes a list of atom indices
    # highlightAtomColors takes a mapping from atom indices to colors
    if highlight_atoms is None:
        highlight_atoms = []
        highlight_atom_colors = None
    elif isinstance(highlight_atoms, dict):
        highlight_atom_colors = {
            idx_map[i]: tuple(c) for i, c in highlight_atoms.items() if i in idx_map
        }
        highlight_atoms = list(highlight_atoms.keys())
    else:
        highlight_atoms = [idx_map[i] for i in highlight_atoms if i in idx_map]
        highlight_atom_colors = None

    # Process highlight_bonds argument for highlightBonds and highlightBondColors
    # highlightBonds takes a list of bond indices
    # highlightBondColors takes a mapping from bond indices to colors
    if highlight_bonds is None:
        highlight_bonds = []
        highlight_bond_colors = None
    elif isinstance(highlight_bonds, dict):
        highlight_bond_colors = {
            rdmol.GetBondBetweenAtoms(idx_map[i_a], idx_map[i_b]).GetIdx(): tuple(v)
            for (i_a, i_b), v in highlight_bonds.items()
            if i_a in idx_map and i_b in idx_map
        }

        highlight_bonds = list(highlight_bond_colors.keys())
    else:
        highlight_bonds = [
            rdmol.GetBondBetweenAtoms(idx_map[i_a], idx_map[i_b]).GetIdx()
            for i_a, i_b in highlight_bonds
            if i_a in idx_map and i_b in idx_map
        ]
        highlight_bond_colors = None

    # Process bond_notes argument and place notes in the molecule
    if bond_notes is not None:
        for (i_a, i_b), note in bond_notes.items():
            if i_a not in idx_map or i_b not in idx_map:
                continue
            rdbond = rdmol.GetBondBetweenAtoms(idx_map[i_a], idx_map[i_b])
            rdbond.SetProp("bondNote", note)

    # Process atom_notes argument and place notes in the molecule
    if atom_notes is not None:
        for i, note in atom_notes.items():
            if i not in idx_map:
                continue
            rdatom = rdmol.GetAtomWithIdx(idx_map[i])
            rdatom.SetProp("atomNote", note)

    # Fix kekulization so it is the same for all drawn molecules
    Chem.rdmolops.Kekulize(rdmol, clearAromaticFlags=True)

    # Compute 2D coordinates
    Compute2DCoords(rdmol)

    # Construct the drawing object and get a handle to its options
    drawer = Draw.MolDraw2DSVG(image_width, image_height)
    draw_options = drawer.drawOptions()

    # Specify the scale to fit all atoms
    # This is important for emphasize_atoms
    coords_2d = next(rdmol.GetConformers()).GetPositions()[..., (0, 1)]
    drawer.SetScale(
        image_width,
        image_height,
        Point2D(*(coords_2d.min(axis=0) - 1.0)),
        Point2D(*(coords_2d.max(axis=0) + 1.0)),
    )

    # Set the colors used for each element according to the emphasize_atoms and
    # color_by_element arguments
    if emphasize_atoms:
        draw_options.setAtomPalette(
            {i: (0.8, 0.8, 0.8) for i in range(rdmol.GetNumAtoms())}
        )
    else:
        set_atom_palette(draw_options)

    # Draw the molecule
    # Note that if emphasize_atoms is used, this will be the un-emphasized parts
    # of the molecule
    drawer.DrawMolecule(
        rdmol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=highlight_bond_colors,
    )

    # Draw an overlapping molecule for the emphasized atoms
    if emphasize_atoms:
        # Set the atom palette according to the color_by_element argument
        set_atom_palette(draw_options)

        # Create a copy of the molecule that removes atoms that aren't emphasized
        emphasized = Chem.rdchem.RWMol(rdmol)
        emphasized.BeginBatchEdit()
        for i in set(idx_map) - set(emphasize_atoms):
            emphasized.RemoveAtom(idx_map[i])
        emphasized.CommitBatchEdit()

        # Draw the molecule. The scale has been fixed and we're re-using the
        # same coordinates, so this will overlap the background molecule
        drawer.DrawMolecule(emphasized)

    # Finalize the SVG
    drawer.FinishDrawing()

    # Return an SVG object that we can view in notebook
    svg_contents = drawer.GetDrawingText()
    return SVG(svg_contents)


class OpenFFMoleculeTrajectory(Structure, Trajectory):
    """OpenFF Molecule adaptor.

    Example
    -------
    >>> import nglview as nv
    >>> import mdtraj as md
    >>> mol = Molecule.from_polymer_pdb(pdb_filename)
    >>> t = OpenFFMoleculeTrajectory(mol)
    >>> w = nv.NGLWidget(t)
    >>> w
    """

    def __init__(self, molecule: Molecule, ext: str = "MOL2"):
        if not molecule.conformers:
            raise ValueError(
                "Cannot visualize a molecule without conformers with NGLView"
            )
        self.molecule = molecule
        self.ext = ext.lower()
        self.params = {}
        self.id = str(uuid.uuid4())

    def get_coordinates(self, index: int):
        return self.molecule.conformers[index].m_as(unit.angstrom)

    @property
    def n_frames(self):
        return len(self.molecule.conformers)

    def get_structure_string(self):
        with StringIO() as f:
            self.molecule.to_file(f, file_format=self.ext)
            sdf_str = f.getvalue()
        return sdf_str


class OpenFFTopologyStructure(Structure):
    """OpenFF Molecule adaptor.

    Example
    -------
    >>> import nglview as nv
    >>> import mdtraj as md
    >>> mol = Molecule.from_polymer_pdb(pdb_filename)
    >>> t = OpenFFMoleculeTrajectory(mol)
    >>> w = nv.NGLWidget(t)
    >>> w
    """

    def __init__(
        self,
        topology: Topology,
        ext: str,
    ):
        self.topology = topology
        self.ext = ext.lower()
        self.params = {}
        self.id = str(uuid.uuid4())

    def get_structure_string(self):
        with StringIO() as f:
            self.topology.to_file(f, file_format=self.ext)
            sdf_str = f.getvalue()
        return sdf_str


def visualize(obj: Union[Molecule, Topology], ext=None, representations=None):
    """Visualize a topology or molecule with nglview"""
    if isinstance(obj, Molecule):
        if representations is None:
            representations = MOLECULE_DEFAULT_REPS
        if ext is None:
            ext = "MOL2"
        return nglview.NGLWidget(
            OpenFFMoleculeTrajectory(obj, ext=ext),
            representations=representations,
        )
    elif isinstance(obj, Topology):
        if ext is None:
            ext = "PDB"
        return nglview.NGLWidget(
            OpenFFTopologyStructure(obj, ext=ext),
            representations=representations,
        )
    else:
        if ext is not None:
            raise ValueError(
                "ext parameter only supported for topologies and molecules"
            )
        return nglview.NGLWidget(
            structure=obj,
            representations=representations,
        )


def solvate(
    topology: Topology,
    positive_ion: Literal["Cs+", "K+", "Li+", "Na+", "Rb+"] = "Na+",
    negative_ion: Literal["Cl-", "Br-", "F-", "I-"] = "Cl-",
    ionic_strength: Quantity = 0.0 * unit.molar,
):
    """
    Solvate a topology with box vectors in water.

    Parameters
    ==========
    topology
        The topology to solvate. A copy of the topology is returned; it is not
        mutated. The topology must have box vectors.
    positive_ion
        The type of positive ion to add.
    negative_ion
        The type of negative ion to add.
    ionic_strength : OpenFF ``Quantity`` with dimensions of concentration
        The total concentration of ions (both positive and negative) to add.
        This does not include ions that are added to neutralize the system.

    Raises
    ======
    ValueError
        If the given topology has no box vectors
    """
    if topology.box_vectors is None:
        raise ValueError("Topology has no box vectors")

    with StringIO() as f:
        topology.to_file(f)
        f.seek(0)
        fixer = PDBFixer(pdbfile=f)

    fixer.addSolvent(
        boxVectors=topology.box_vectors.to_openmm(),
        positiveIon=positive_ion,
        negativeIon=negative_ion,
        ionicStrength=ionic_strength.to_openmm(),
    )

    solvated_topology = Topology.from_openmm(
        fixer.topology,
        unique_molecules={
            *topology.molecules,
            Molecule.from_smiles("O"),
            Molecule.from_smiles(f"[{positive_ion}]"),
            Molecule.from_smiles(f"[{negative_ion}]"),
        },
    )
    solvated_topology.set_positions(from_openmm(fixer.positions))
    return solvated_topology


def max_dist_between_points(points: Quantity[ArrayLike]) -> Quantity[float]:
    """
    Compute the greatest distance between two points in the array.
    """
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist

    # Support Pint/OpenFF units
    try:
        units = points.units
        points: NDArray = np.asarray(points.magnitude)
    except AttributeError:
        units = 1.0
        points = np.asarray(points)

    if points.shape[1] != 3 or points.ndim != 2:
        raise ValueError("Points should be an n*3 array")

    if points.shape[0] >= 4:
        # Maximum distance is guaranteed to be on the convex hull, which can be
        # computed in O(n log n)
        # See also https://stackoverflow.com/a/60955825
        hull = ConvexHull(points)
        hullpoints = points[hull.vertices, :]
    else:
        hullpoints = points

    # Now compute all the distances and get the greatest distance in O(h^2)
    max_dist = pdist(hullpoints, metric="euclidean").max()
    return max_dist * units
