#date: 2021-11-05T16:55:02Z
#url: https://api.github.com/gists/2845d1830981b795ec4fe858618496d0
#owner: https://api.github.com/users/JBorrow

"""
Calculates star formation rates based on physical apertures.
Only calculates for central galaxies.
"""

from typing import List, Dict
from swiftsimio import load as load_snapshot, SWIFTDataset
from velociraptor import load as load_catalogue

from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar
from pathlib import Path

import unyt
import numpy as np

from tqdm import tqdm


APERTURE_SIZES = [
    unyt.unyt_quantity(x, "kpc")
    for x in [
        5.0,
        10.0,
        15.0,
        30.0,
        50.0,
        100.0,
    ]
]

TIMESCALES = [
    unyt.unyt_quantity(x, "Myr")
    for x in [
        100.0,
        1000.0,
    ]
]


def get_scale_factor_for_timescale(
    snapshot: SWIFTDataset,
    timescale: unyt.unyt_quantity,
) -> float:
    """
    Gets the scale factor associated with a
    lookback time from the current time to
    timescale.
    """

    current_redshift = snapshot.metadata.z
    cosmology = snapshot.metadata.cosmology

    current_lookback_time = cosmology.lookback_time(current_redshift)
    timescale_astropy = timescale.to_astropy()
    required_lookback_time = current_lookback_time + timescale_astropy

    def time_difference(redshift):
        dt = cosmology.lookback_time(redshift) - required_lookback_time
        return abs(dt.value)

    redshift = minimize_scalar(time_difference).x

    return 1.0 / (1.0 + redshift)


def build_tree(snapshot: SWIFTDataset) -> cKDTree:
    """
    Builds a tree (in physical co-ordinates) of all star particles.
    """

    star_coords = snapshot.stars.coordinates
    star_coords.convert_to_units("kpc")

    # We only want to build the tree with particles we're going to use
    # anyway.

    max_timescale = max(TIMESCALES)

    min_scale_factor = get_scale_factor_for_timescale(snapshot, max_timescale)

    print(f"Limiting particles to {max_timescale} (a > {min_scale_factor})")

    mask = snapshot.stars.birth_scale_factors > min_scale_factor

    return cKDTree(star_coords[mask] * snapshot.metadata.a)


def search_particles(
    catalogue,
    snapshot: SWIFTDataset,
    tree: cKDTree,
    apertures: List[unyt.unyt_quantity],
    timescales: List[unyt.unyt_quantity],
):

    mask = catalogue.structure_type.structuretype == 10

    centers = np.array(
        [getattr(catalogue.positions, f"{x}cmbp")[mask].to("kpc").v for x in "xyz"]
    ).T


    # Need to re-mask particle quantities again to cut out vast majority of 'old'
    # stars.
    max_timescale = max(TIMESCALES)
    min_scale_factor = get_scale_factor_for_timescale(snapshot, max_timescale)

    particle_mask = snapshot.stars.birth_scale_factors > min_scale_factor

    masses = snapshot.stars.initial_masses.v[particle_mask]
    birth_scale_factors = snapshot.stars.birth_scale_factors.v[particle_mask]
    coords = snapshot.stars.coordinates.v[particle_mask] * snapshot.metadata.a

    # We want to start with the largest aperture, and the highest timescale.
    aperture_values = np.sort(np.array(apertures))[::-1]
    timescale_values = np.sort(np.array(timescales))[::-1]
    scale_factor_values = np.array(
        [
            get_scale_factor_for_timescale(snapshot, timescale)
            for timescale in timescale_values * timescales[0].units
        ]
    )

    def get_name(aperture_size, timescale):
        return f"Aperture_SFR{int(timescale)}Myr_gas_{int(aperture_size)}_kpc"

    star_formation_names = []

    for aperture in aperture_values:
        for timescale in timescale_values:
            star_formation_names.append(get_name(aperture, timescale))

    star_formation_rates = {
        name: np.empty(len(centers), dtype=float) for name in star_formation_names
    }

    # import pdb; pdb.set_trace()

    for halo, center in enumerate(tqdm(centers)):
        particle_indices = np.array(tree.query_ball_point(center, r=aperture_values[0]))

        for timescale, scale_factor in zip(timescale_values, scale_factor_values):
            # Because of sort we can be destructive. There may be no particles, though
            # and you can't slice an empty array.
            if len(particle_indices) > 0:
                mask = particle_indices[birth_scale_factors[particle_indices] > scale_factor]
            else:
                mask = np.s_[0:0]

            r = np.linalg.norm(center - coords[mask], axis=1)
            m = masses[mask]

            for aperture in aperture_values:
                formed_mass_in_aperture = m[r < aperture].sum()
                sfr_in_aperture = formed_mass_in_aperture / timescale

                star_formation_rates[get_name(aperture, timescale)][
                    halo
                ] = sfr_in_aperture

    return star_formation_rates



def write_to_disk(
        output_file: Path,
        star_formation_rates: Dict[str, np.array],
        snapshot: SWIFTDataset,
        catalogue,
    ):
    """
    Writes the star formation rates to disk.
    """

    for name, sfr in star_formation_rates.items():
        # Need to un-central-only.
        output = np.zeros_like(catalogue.apertures.mass_star_50_kpc)

        output[catalogue.structure_type.structuretype == 10] = sfr

        unyt.unyt_array(
            output,
            snapshot.stars.masses.units / TIMESCALES[0].units,
            name=name,
        ).write_hdf5(
            output_file,
            dataset_name=name,
        )

    return


if __name__ == "__main__":
    import argparse as ap

    parser = ap.ArgumentParser(
        prog="aperture_sfrs",
        description="""
        Calculates aperture-based star formation rates at 100 and 1000 Myrs.
        """,
        epilog=(
            "Example usage: python3 aperture_sfrs -c halo_0003.properties"
            " -s eagle_0003.hdf5"
        ),
    )

    parser.add_argument(
        "-c", "--catalogue", required=True, type=Path, help="Catalogue filename."
    )

    parser.add_argument(
        "-s", "--snapshot", required=True, type=Path, help="Snapshot filename"
    )

    parser.add_argument(
        "-o", "--output", required=True, type=Path, help="Output filename"
        )

    args = parser.parse_args()

    catalogue = load_catalogue(args.catalogue, disregard_units=True)
    snapshot = load_snapshot(args.snapshot)

    tree = build_tree(snapshot=snapshot)

    star_formation_rates = search_particles(
        catalogue=catalogue,
        snapshot=snapshot,
        tree=tree,
        apertures=APERTURE_SIZES,
        timescales=TIMESCALES,
    )

    write_to_disk(
        output_file=args.output,
        star_formation_rates=star_formation_rates,
        catalogue=catalogue,
        snapshot=snapshot,
    )