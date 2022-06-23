#date: 2022-06-23T17:01:03Z
#url: https://api.github.com/gists/371d91bac62d4466b9c595b1f7bcf1ff
#owner: https://api.github.com/users/wiederm

# imports
import openmm as mm
from openmm.app import Simulation
from openff.evaluator.protocols import coordinates
from openff.evaluator.utils import packmol
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit
from openff.toolkit.topology import Molecule, Topology
from openmmml import MLPotential

forcefield = ForceField("openff_unconstrained-2.0.0.offxml")
# define ligand
ligand = Molecule.from_smiles("CC(C)=O")
ligand.name = "acetone"
# define solvent
water = Molecule.from_smiles("O")
water.name = ""
# build waterbox
molecules = [ligand, water]
n_molecules = [1, 1000]
trajectory, _ = packmol.pack_box(
    molecules=molecules,
    number_of_copies=n_molecules,
    mass_density=0.95 * unit.grams / unit.milliliters,
)
from simtk import unit
### define units
distance_unit = unit.angstrom
time_unit = unit.femtoseconds
speed_unit = distance_unit / time_unit

# constants
stepsize = 1 * time_unit
collision_rate = 1 / unit.picosecond
temperature = 300 * unit.kelvin
topology = Topology.from_molecules([ligand, *1000 * [water]])
# bring units in
trajectory.unitcell_vectors * unit.nanometer

vectors = [
    trajectory.unitcell_vectors[0][0],
    trajectory.unitcell_vectors[0][1],
    trajectory.unitcell_vectors[0][2],
] * unit.nanometer

topology.box_vectors = vectors
# create system
system = forcefield.create_openmm_system(topology)
platform = 'CUDA'
potential = MLPotential("ani2x")
platform = mm.Platform.getPlatformByName(platform)
system = potential.createSystem(topology.to_openmm())
integrator = mm.LangevinIntegrator(temperature, collision_rate, stepsize)

sim = Simulation(topology, system, integrator, platform=platform)
print("Initializing QML system")
