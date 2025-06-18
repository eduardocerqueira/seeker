#date: 2025-06-18T17:01:31Z
#url: https://api.github.com/gists/3c12c17a0099e6c401de8c5e842fcc12
#owner: https://api.github.com/users/h-mayorquin

"""
CaImAn HDF5 File Stubbing Utility

This module provides functionality to create smaller "stubbed" versions of CaImAn HDF5 output files
by reducing the number of neurons/components and timestamps while preserving the complete data structure.

CaImAn (Calcium Imaging Analysis) produces HDF5 files containing:
- Spatial footprints of neurons (where they are in the image)
- Temporal traces (calcium activity over time)
- Quality metrics and parameters
- Background components and noise estimates

Stubbing is useful for creating test datasets, prototyping, and reducing memory usage.
"""

import h5py
import numpy as np
from typing import Optional
from scipy.sparse import csc_matrix
from pathlib import Path


def stub_caiman_hdf5(input_file: str, output_file: Optional[str] = None, n_timestamps: int = 10, n_units: Optional[int] = None, stub_one_photon_quantities: bool = True):
    """
    Create a stubbed version of a CaImAn HDF5 file with reduced units and timestamps.

    This is useful for:
    - Creating small test datasets for development/debugging
    - Rapid prototyping of analysis pipelines
    - Testing visualization code with minimal data
    - Reducing memory requirements for demos

    The stubbing preserves the complete CaImAn data structure while reducing:
    - Number of neurons/components (spatial and temporal data)
    - Number of time points (frames) in the recording

    Parameters
    ----------
    input_file : str
        Path to the input CaImAn HDF5 file
    output_file : str, optional
        Path to the output stubbed file. If None, creates a file with '_stubbed' suffix
    n_timestamps : int, default=10
        Number of timestamps to keep in the stubbed file
    n_units : int or None, default=None
        Number of units/components to keep in the stubbed file.
        If None, keeps all units and only stubs time dimension.
    stub_one_photon_quantities : bool, default=True
        If True, removes one-photon specific data (W matrix and b0) from the stub.
        This reduces file size when testing two-photon workflows.

    Returns
    -------
    str
        Path to the created stubbed file
    """

    if output_file is None:
        input_path = Path(input_file)
        suffix_parts = []
        
        if n_units is not None:
            suffix_parts.append(f"{n_units}units")
        suffix_parts.append(f"{n_timestamps}frames")
        
        if not stub_one_photon_quantities:
            suffix_parts.append("with1p")
            
        suffix = "_".join(suffix_parts)
        output_file = str(input_path.with_stem(f"{input_path.stem}_stubbed_{suffix}"))

    print(f"Creating stubbed file: {output_file}")
    if n_units is None:
        print(f"Keeping all units, reducing to {n_timestamps} timestamps")
    else:
        print(f"Reducing to {n_units} units and {n_timestamps} timestamps")
    
    if stub_one_photon_quantities:
        print("Removing one-photon specific data (W matrix and b0)")

    with h5py.File(input_file, "r") as src, h5py.File(output_file, "w") as dst:
        # Get original dimensions from temporal components
        # C is the denoised calcium traces matrix (components × time)
        orig_n_components = src["/estimates/C"].shape[0]
        orig_n_timestamps = src["/estimates/C"].shape[1]

        # Handle n_units
        if n_units is None:
            n_units_to_use = orig_n_components  # Keep all units
            stub_units = False
        else:
            n_units_to_use = min(n_units, orig_n_components)
            stub_units = True

        # Ensure we don't request more timestamps than available
        n_timestamps = min(n_timestamps, orig_n_timestamps)

        print(f"Original: {orig_n_components} components × {orig_n_timestamps} timestamps")
        if stub_units:
            print(f"Stubbed: {n_units_to_use} components × {n_timestamps} timestamps")
        else:
            print(f"Stubbed: all {orig_n_components} components × {n_timestamps} timestamps")

        # Copy root-level attributes directly
        for attr_name, attr_value in src.attrs.items():
            dst.attrs[attr_name] = attr_value

        # Handle each major component of CaImAn output
        # 1. Estimates: neuron traces, spatial footprints, quality metrics
        handle_estimates(src, dst, n_units_to_use, n_timestamps, stub_units, stub_one_photon_quantities)
        # 2. Parameters: all algorithm settings (only K needs updating if stubbing units)
        handle_parameters(src, dst, n_units_to_use, stub_units)
        # 3. Other root items: dims, flags, etc.
        handle_other_root_items(src, dst)

    print(f"Successfully created stubbed file: {output_file}")
    return output_file


def handle_estimates(src, dst, n_units, n_timestamps, stub_units=True, stub_one_photon_quantities=True):
    """Handle all datasets in the estimates group."""
    if "/estimates" not in src:
        return

    # Create estimates group
    estimates_group = dst.create_group("/estimates")

    # Copy group attributes
    for attr_name, attr_value in src["/estimates"].attrs.items():
        estimates_group.attrs[attr_name] = attr_value

    # Handle temporal components (2D arrays that need both dimensions reduced)
    # These are the core results of CaImAn analysis:
    # - C: Denoised temporal traces for each neuron
    # - S: Deconvolved neural activity (spikes) 
    # - F_dff: DF/F normalized fluorescence traces
    # - YrA: Residual signals after denoising
    # All have shape (n_components × n_timesteps), so we stub both dimensions
    temporal_components = ['C', 'S', 'F_dff', 'YrA']
    for comp in temporal_components:
        if f'/estimates/{comp}' in src:
            dataset = src[f'/estimates/{comp}']
            if dataset.shape == ():  # Handle edge case of scalar
                data = dataset[()]
            elif stub_units:
                data = dataset[:n_units, :n_timestamps]
            else:
                data = dataset[:, :n_timestamps]
            copy_dataset_with_attrs(dataset, estimates_group.create_dataset(comp, data=data))

    # Handle per-component arrays (1D arrays that need unit dimension reduced)
    # These store quality metrics and parameters for each neuron:
    # - SNR_comp: Signal-to-noise ratio for each component
    # - r_values: Spatial footprint consistency (correlation with raw data)
    # - bl: Baseline value for each trace
    # - c1: Initial value for each trace
    # - lam: Regularization parameter for each component
    # - neurons_sn: Noise standard deviation for each trace
    # All have length n_components, so we keep only first n_units
    component_arrays = ['SNR_comp', 'r_values', 'bl', 'c1', 'lam', 'neurons_sn']
    for arr in component_arrays:
        if f'/estimates/{arr}' in src:
            dataset = src[f'/estimates/{arr}']
            if dataset.shape == ():  # Handle edge case of scalar
                data = dataset[()]
            elif stub_units:
                data = dataset[:n_units]
            else:
                data = dataset[:]
            copy_dataset_with_attrs(dataset, estimates_group.create_dataset(arr, data=data))

    # Handle g (can be 1D or 2D)
    # g contains the time constants for each trace from the autoregressive model.
    # Format varies: older versions store as 1D array, newer versions as 2D (n_components × p)
    # where p is the order of the AR model (usually 1 or 2)
    if '/estimates/g' in src:
        g_dataset = src['/estimates/g']
        g_data = g_dataset[()] if g_dataset.shape == () else g_dataset[:]
        
        if stub_units and g_data.size > 0:  # Only slice if not empty/scalar
            if g_data.ndim == 2:
                g_stubbed = g_data[:n_units, :]
            elif g_data.ndim == 1:
                g_stubbed = g_data[:n_units]
            else:
                g_stubbed = g_data
        else:
            g_stubbed = g_data
        copy_dataset_with_attrs(g_dataset, estimates_group.create_dataset('g', data=g_stubbed))

    # Handle component indices (filter to valid range)
    # These lists track which components passed/failed quality evaluation
    # We must filter out indices >= n_units since those components no longer exist
    if '/estimates/idx_components' in src:
        dataset = src['/estimates/idx_components']
        idx_data = dataset[()] if dataset.shape == () else dataset[:]
        
        if stub_units and idx_data.size > 0:
            idx_filtered = idx_data[idx_data < n_units]
        else:
            idx_filtered = idx_data
        copy_dataset_with_attrs(dataset, estimates_group.create_dataset('idx_components', data=idx_filtered))
    
    if '/estimates/idx_components_bad' in src:
        dataset = src['/estimates/idx_components_bad']
        idx_bad_data = dataset[()] if dataset.shape == () else dataset[:]
        
        if stub_units and idx_bad_data.size > 0:
            idx_bad_filtered = idx_bad_data[idx_bad_data < n_units]
        else:
            idx_bad_filtered = idx_bad_data
        copy_dataset_with_attrs(dataset, estimates_group.create_dataset('idx_components_bad', data=idx_bad_filtered))

    # Handle spatial components (sparse matrix A)
    # A contains spatial footprints of neurons (pixels × components)
    # Stored as sparse matrix since each neuron only occupies small region
    # We stub by keeping only first n_units columns (neurons)
    handle_spatial_components_A(src, estimates_group, n_units, stub_units)
    
    # Handle ring model matrix W (for 1p data)
    # W is used for background computation in 1-photon (CNMF-E) analysis
    # It's pixel×pixel, not component-based, so we keep it unchanged
    # Unless stub_one_photon_quantities is True, then we skip it
    if not stub_one_photon_quantities:
        handle_ring_model_W(src, estimates_group)
    
    # Handle pixel-based arrays (unchanged)
    # These relate to the imaging field, not individual neurons:
    # - b0: Baseline fluorescence value for each pixel (1p only) - skipped if stub_one_photon_quantities
    # - sn: Noise standard deviation for each pixel
    # - Cn: Correlation image showing pixel correlations (for visualization)
    # Keep unchanged since stubbing doesn't change image dimensions
    pixel_arrays = ['sn', 'Cn']  # Always include these
    if not stub_one_photon_quantities:
        pixel_arrays.append('b0')  # Only include b0 if not stubbing 1p data
        
    for arr in pixel_arrays:
        if f'/estimates/{arr}' in src:
            dataset = src[f'/estimates/{arr}']
            # Handle potential scalar datasets
            if dataset.shape == ():
                data = dataset[()]
            else:
                data = dataset[:]
            copy_dataset_with_attrs(dataset, estimates_group.create_dataset(arr, data=data))
    
    # Handle scalars and other unchanged datasets
    # These are mostly auxiliary data or online processing variables:
    # - b, f: Background spatial/temporal components (2p analysis)
    # - center: Centroid coordinates for spatial footprints
    # - Online processing: AtA, CY, CC (correlation matrices), mn/vr (mean/variance)
    # - dims: Image dimensions [height, width]
    # Most are kept unchanged as they're either metadata or not component-specific
    unchanged_scalars = [
        "Ab", "Ab_dense", "AtA", "AtY_buf", "CC", "CY", "C_on", 
        "Cf", "Yr_buf", "b", "f", "center", "discarded_components", 
        "ecc", "ind_new", "mn", "noisyC", "rho_buf", "sv", "vr",
        "dims", "A_thr", "R", "OASISinstances", "shifts",
    ]
    
    for scalar in unchanged_scalars:
        if f"/estimates/{scalar}" in src:
            dataset = src[f"/estimates/{scalar}"]
            # Handle scalar datasets differently
            if dataset.shape == ():
                data = dataset[()]
            else:
                data = dataset[:]
            copy_dataset_with_attrs(dataset, estimates_group.create_dataset(scalar, data=data))
    
    # Handle special cases
    if "/estimates/nr" in src:
        # nr stores the number of components - update to reflect stubbing
        dataset = src["/estimates/nr"]
        if stub_units:
            copy_dataset_with_attrs(dataset, estimates_group.create_dataset("nr", data=n_units))
        else:
            # Handle scalar
            if dataset.shape == ():
                data = dataset[()]
            else:
                data = dataset[:]
            copy_dataset_with_attrs(dataset, estimates_group.create_dataset("nr", data=data))
    
    # Handle cnn_preds (might be empty)
    # CNN predictions for component shape evaluation (0-1 score)
    # If present and non-empty, keep only predictions for first n_units
    if '/estimates/cnn_preds' in src:
        dataset = src['/estimates/cnn_preds']
        cnn_data = dataset[()] if dataset.shape == () else dataset[:]
        
        if cnn_data.size > 0 and stub_units and cnn_data.ndim > 0:
            cnn_stubbed = cnn_data[:n_units]
        else:
            cnn_stubbed = cnn_data
        copy_dataset_with_attrs(dataset, estimates_group.create_dataset('cnn_preds', data=cnn_stubbed))


def handle_spatial_components_A(src, estimates_group, n_units, stub_units=True):
    """Handle the sparse spatial components matrix A.
    
    A contains spatial footprints showing where each neuron is located.
    Shape: (n_pixels × n_components), stored as sparse CSC matrix because
    each neuron only occupies ~0.3-1% of the image pixels.
    We stub by keeping only the first n_units columns (neurons).
    """
    if '/estimates/A' not in src:
        return
    
    # Read sparse matrix components
    # CSC format stores: data (non-zero values), indices (row positions),
    # indptr (column boundaries), and shape
    data_dataset = src['/estimates/A/data']
    indices_dataset = src['/estimates/A/indices']
    indptr_dataset = src['/estimates/A/indptr']
    shape_dataset = src['/estimates/A/shape']
    
    # Read the data, handling potential scalars
    data = data_dataset[()] if data_dataset.shape == () else data_dataset[:]
    indices = indices_dataset[()] if indices_dataset.shape == () else indices_dataset[:]
    indptr = indptr_dataset[()] if indptr_dataset.shape == () else indptr_dataset[:]
    shape = shape_dataset[()] if shape_dataset.shape == () else shape_dataset[:]
    
    if stub_units:
        # Reconstruct the sparse matrix
        A_sparse = csc_matrix((data, indices, indptr), shape=tuple(shape))
        
        # Stub the matrix (keep only first n_units columns/components)
        # This efficiently extracts spatial footprints for first n_units neurons
        A_stubbed = A_sparse[:, :n_units]
        
        # Create A group
        A_group = estimates_group.create_group('A')
        
        # Save stubbed sparse matrix components
        A_group.create_dataset('data', data=A_stubbed.data)
        A_group.create_dataset('indices', data=A_stubbed.indices)
        A_group.create_dataset('indptr', data=A_stubbed.indptr)
        A_group.create_dataset('shape', data=A_stubbed.shape)
    else:
        # Keep A unchanged
        A_group = estimates_group.create_group('A')
        A_group.create_dataset('data', data=data)
        A_group.create_dataset('indices', data=indices)
        A_group.create_dataset('indptr', data=indptr)
        A_group.create_dataset('shape', data=shape)
    
    # Copy group attributes
    for attr_name, attr_value in src['/estimates/A'].attrs.items():
        A_group.attrs[attr_name] = attr_value


def handle_ring_model_W(src, estimates_group):
    """Handle the ring model sparse matrix W (used in 1p processing).
    
    W is used in CNMF-E (1-photon) to compute background using ring model.
    Shape: (n_pixels × n_pixels) - relates pixels to their surrounding rings.
    We keep it unchanged because it's based on pixel relationships, not components.
    """
    if '/estimates/W' not in src:
        return
    
    # W is pixel×pixel, so we keep it unchanged
    W_group = estimates_group.create_group('W')
    
    for dataset_name in ['data', 'indices', 'indptr', 'shape']:
        if f'/estimates/W/{dataset_name}' in src:
            dataset = src[f'/estimates/W/{dataset_name}']
            # Handle potential scalar datasets
            if dataset.shape == ():
                data = dataset[()]
            else:
                data = dataset[:]
            W_group.create_dataset(dataset_name, data=data)
    
    # Copy group attributes
    for attr_name, attr_value in src['/estimates/W'].attrs.items():
        W_group.attrs[attr_name] = attr_value


def handle_parameters(src, dst, n_units, stub_units=True):
    """Handle all parameter groups, updating K to reflect stubbed components.
    
    Parameters control CaImAn's behavior and are organized by analysis stage:
    - data: Dataset info (dimensions, framerate, decay time)
    - init: Initialization params (K, gSig, patch size)
    - motion: Motion correction settings
    - quality: Component evaluation thresholds
    - spatial/temporal: Processing parameters
    Most stay unchanged except K (expected components per patch).
    """
    if '/params' not in src:
        return
    
    # Create params group
    params_group = dst.create_group('/params')
    
    # Copy group attributes
    for attr_name, attr_value in src['/params'].attrs.items():
        params_group.attrs[attr_name] = attr_value
    
    # Handle each parameter subgroup
    param_groups = ['data', 'init', 'motion', 'online', 'patch', 
                   'preprocess', 'quality', 'spatial', 'temporal', 
                   'merging', 'ring_CNN']
    
    for group_name in param_groups:
        if f'/params/{group_name}' in src:
            handle_parameter_group(src, params_group, group_name, n_units, stub_units)


def handle_parameter_group(src, params_group, group_name, n_units, stub_units=True):
    """Handle a specific parameter group.
    
    Only the 'K' parameter in 'init' group needs updating - it specifies
    the expected number of components per patch during initialization.
    All other parameters remain unchanged.
    """
    src_group = src[f'/params/{group_name}']
    dst_group = params_group.create_group(group_name)
    
    # Copy group attributes
    for attr_name, attr_value in src_group.attrs.items():
        dst_group.attrs[attr_name] = attr_value
    
    # Copy all datasets in the group
    for item_name in src_group:
        if isinstance(src_group[item_name], h5py.Dataset):
            dataset = src_group[item_name]
            # Handle scalar datasets
            if dataset.shape == ():
                data = dataset[()]
            else:
                data = dataset[:]

            # Special case: update K parameter in init group
            # K is the expected number of components per patch
            # Must be updated to match our stubbed component count
            if stub_units and group_name == "init" and item_name == "K":
                data = n_units

            copy_dataset_with_attrs(dataset, dst_group.create_dataset(item_name, data=data))


def handle_other_root_items(src, dst):
    """Handle other root-level items (dims, dview, etc.).

    These are miscellaneous items at the root level:
    - dims: Image dimensions [height, width] - kept unchanged
    - dview: Parallel processing view object reference - kept unchanged
    - remove_very_bad_comps: Flag for component filtering - kept unchanged
    - skip_refinement: Flag for skipping spatial refinement - kept unchanged
    """
    root_items = ["dims", "dview", "remove_very_bad_comps", "skip_refinement"]

    for item in root_items:
        if item in src and isinstance(src[item], h5py.Dataset):
            dataset = src[item]
            # Handle scalar datasets
            if dataset.shape == ():
                data = dataset[()]
            else:
                data = dataset[:]
            copy_dataset_with_attrs(dataset, dst.create_dataset(item, data=data))


def copy_dataset_with_attrs(src_dataset, dst_dataset):
    """Copy all attributes from source dataset to destination dataset."""
    for attr_name, attr_value in src_dataset.attrs.items():
        dst_dataset.attrs[attr_name] = attr_value


def verify_stubbed_file(stubbed_file):
    """
    Verify the contents of a stubbed file and print key dimensions.

    This is a utility function to quickly check that stubbing worked correctly
    by printing the dimensions of key arrays and counts of components.

    Parameters
    ----------
    stubbed_file : str
        Path to the stubbed HDF5 file
    """
    print(f"\nVerifying stubbed file: {stubbed_file}")
    print("-" * 50)

    with h5py.File(stubbed_file, "r") as f:
        # Check temporal components
        if "/estimates/C" in f:
            C_shape = f["/estimates/C"].shape
            print(f"Temporal components (C): {C_shape}")

        if "/estimates/S" in f:
            S_shape = f["/estimates/S"].shape
            print(f"Deconvolved activity (S): {S_shape}")

        # Check spatial components
        if "/estimates/A/shape" in f:
            A_shape = f["/estimates/A/shape"][:]
            print(f"Spatial components (A): {tuple(A_shape)}")

        # Check component evaluation
        if "/estimates/idx_components" in f:
            n_accepted = len(f["/estimates/idx_components"][:])
            print(f"Accepted components: {n_accepted}")

        if "/estimates/idx_components_bad" in f:
            n_rejected = len(f["/estimates/idx_components_bad"][:])
            print(f"Rejected components: {n_rejected}")

        # Check parameters
        if "/params/init/K" in f:
            K = f["/params/init/K"][()]
            print(f"K parameter: {K}")

        if "/estimates/nr" in f:
            nr = f["/estimates/nr"][()]
            print(f"nr (number of components): {nr}")
        
        # Check for one-photon specific data
        if "/estimates/W" in f:
            print("W matrix (1p ring model): Present")
        else:
            print("W matrix (1p ring model): Not present (stubbed)")
            
        if "/estimates/b0" in f:
            print("b0 (1p baseline): Present")
        else:
            print("b0 (1p baseline): Not present (stubbed)")


# Example usage
if __name__ == "__main__":
    # Path to the CaImAn HDF5 file
    folder_path = Path("/home/heberto/data/segmentation_example/mini_sample")
    file_path = folder_path / "mini_750_caiman.hdf5"

    # Verify paths exist
    assert folder_path.exists(), f"Folder not found: {folder_path}"
    assert file_path.exists(), f"File not found: {file_path}"

    # Quick check of file contents
    with h5py.File(file_path, "r") as f:
        print("File keys:", list(f.keys()))

    # Create a stubbed version keeping all units but reducing to 50 timestamps
    # This is useful for testing temporal processing without changing spatial structure
    # By default, one-photon specific data (W matrix and b0) are removed
    stubbed_file = stub_caiman_hdf5(
        str(file_path), 
        n_timestamps=5,
        n_units=10,  # Keep all units
        stub_one_photon_quantities=True  # Remove 1p-specific data
    )
    verify_stubbed_file(stubbed_file)