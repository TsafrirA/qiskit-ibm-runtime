# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Post-processing functions for converting QuantumProgramResult to primitive-specific formats."""

from __future__ import annotations
from typing import cast

import numpy as np
from qiskit.primitives import PrimitiveResult, DataBin
from qiskit.primitives.containers.estimator_pub import ObservablesArray
from qiskit.quantum_info import PauliList, Pauli

from ......quantum_program.quantum_program_result import QuantumProgramResult
from ......utils.estimator_pub_result import EstimatorPubResult
from ...helpers import get_pauli_basis, identify_measure_basis, compute_exp_val
from ..utils import register_post_processor


@register_post_processor("v0.1")
def estimator_v2_post_processor_v0_1(result: QuantumProgramResult) -> PrimitiveResult:
    """Convert a quantum program result to a primitives result, for a V2 estimator.

    Convert :class:`~.QuantumProgramResult` to a :class:`~qiskit.primitives.PrimitiveResult`,
    for :class:`~qiskit_ibm_runtime.executor.routines.estimator_v2.EstimatorV2`.

    This function transforms the raw quantum program execution results into the
    format expected by :class:`~qiskit_ibm_runtime.executor.routines.estimator_v2.EstimatorV2`,
    computing expectation values from measurement data and creating
    :class:`~qiskit_ibm_runtime.utils.estimator_pub_result.EstimatorPubResult` containers
    for each pub.

    Args:
        result: The raw quantum program result containing measurement data.

    Returns:
        Primitive result for
        :class:`~qiskit_ibm_runtime.executor.routines.estimator_v2.EstimatorV2`.
    """
    if len(result) == 0:
        return PrimitiveResult([])

    if not isinstance(result.passthrough_data, dict):
        raise ValueError(
            "Wrong type for passthrough data: Expected a 'dict', found "
            f"'{type(result.passthrough_data)}'."
        )

    passthrough = cast(dict, result.passthrough_data or {})
    if (post_processor_data := passthrough.get("post_processor", None)) is None:
        raise ValueError("Missing 'post_processor' in passthrough data.")

    # Extract data from passthrough
    observables_list = passthrough.get("observables", None)
    if observables_list is None:
        raise ValueError("Missing 'observables' in passthrough data.")

    measure_bases_list = passthrough.get("measure_bases", None)
    if measure_bases_list is None:
        raise ValueError("Missing 'measure_bases' in passthrough data.")

    # Extract circuit metadata if present
    circuits_metadata = post_processor_data.get("circuits_metadata", None)

    # Validate circuits_metadata length if provided
    circuits_metadata = circuits_metadata or [None] * len(result)
    if circuits_metadata is not None and len(circuits_metadata) != len(result):
        raise ValueError(
            f"Number of circuit metadata items ({len(circuits_metadata)}) does not match "
            f"number of pubs ({len(result)})."
        )

    # Build EstimatorPubResult for each pub
    pub_results = []
    for idx, (item_data, observables_data, measure_bases_labels) in enumerate(
        zip(result, observables_list, measure_bases_list)
    ):
        # Validate that measurement data exists
        if not item_data:
            raise ValueError(f"Pub {idx} has no measurement data")

        # Reconstruct observables and measure_bases
        observables = ObservablesArray(observables_data)
        measure_bases = PauliList(measure_bases_labels)

        # Get measurement data
        # Shape: param_shape + (num_bases,) + (shots, num_bits)
        meas_data = item_data["meas"]

        # Extract param_shape from measurement data
        # meas_data has shape: param_shape + (num_bases,) + (shots, num_bits)
        param_shape = meas_data.shape[:-3] if meas_data.ndim > 3 else ()

        # Compute output shape by broadcasting param_shape with obs_shape
        obs_shape = observables.shape
        output_shape = np.broadcast_shapes(param_shape, obs_shape)

        # Compute expectation values for all observables first
        # Each exp_val has shape param_shape
        exp_vals_dict = {}
        stds_dict = {}

        for obs_idx, observable in np.ndenumerate(observables):
            # Initialize with numpy arrays (works for both scalar and array shapes)
            # np.zeros(()) creates a 0-d array for scalar case
            exp_val = np.zeros(param_shape, dtype=float)
            variance = np.zeros(param_shape, dtype=float)

            for observable_term, coeff in observable.items():
                # Find which basis measured this term
                pauli_basis = Pauli(get_pauli_basis(observable_term))
                basis_idx = identify_measure_basis(pauli_basis, measure_bases)

                # Get measurement data for this basis
                # Shape: param_shape + (shots, num_qubits)
                datum = meas_data[..., basis_idx, :, :]
                term_exp_val, term_std = compute_exp_val(observable_term, datum)

                # Accumulate with coefficient
                exp_val = exp_val + coeff * term_exp_val
                variance = variance + (coeff**2) * (term_std**2)

            exp_vals_dict[obs_idx] = exp_val
            stds_dict[obs_idx] = np.sqrt(variance)

        # Now construct output arrays using broadcasting-aware indexing
        # This unified approach works for all shape combinations:
        # - Both scalar: param_shape=(), obs_shape=()
        # - Scalar params: param_shape=(), obs_shape=(n,)
        # - Scalar obs: param_shape=(m,), obs_shape=()
        # - Both arrays: param_shape=(m,n), obs_shape=(k,)

        # Create index grids (0-d arrays for scalar shapes, regular arrays otherwise)
        if param_shape == ():
            param_grid = np.array(0)  # 0-d array with value 0
        else:
            param_grid = np.arange(np.prod(param_shape)).reshape(param_shape)

        if obs_shape == ():
            obs_grid = np.array(0)  # 0-d array with value 0
        else:
            obs_grid = np.arange(np.prod(obs_shape)).reshape(obs_shape)

        # Broadcast grids to output shape
        param_grid_bc, obs_grid_bc = np.broadcast_arrays(param_grid, obs_grid)

        # Fill output arrays using explicit loop (simpler and more efficient than np.vectorize)
        evs_array = np.empty(output_shape)
        stds_array = np.empty(output_shape)

        for out_idx in np.ndindex(output_shape):
            # Get flat indices for this position
            p_flat = param_grid_bc[out_idx]
            o_flat = obs_grid_bc[out_idx]

            # Convert to multi-dimensional indices
            p_idx = np.unravel_index(int(p_flat), param_shape) if param_shape else ()
            o_idx = np.unravel_index(int(o_flat), obs_shape) if obs_shape else ()

            # Look up values from dictionaries
            # exp_vals_dict[o_idx] has shape param_shape
            ev_array = exp_vals_dict[o_idx]
            std_array = stds_dict[o_idx]

            # Extract the value at the parameter index
            # For scalar param_shape, p_idx is () and we need to extract the scalar
            if param_shape:
                evs_array[out_idx] = ev_array[p_idx]
                stds_array[out_idx] = std_array[p_idx]
            else:
                # param_shape is (), so ev_array is a 0-d array
                # Use .item() to extract the scalar value
                evs_array[out_idx] = ev_array.item()
                stds_array[out_idx] = std_array.item()

        data_bin = DataBin(
            evs=evs_array,
            stds=stds_array,
        )

        # Get circuit metadata for this pub if available
        pub_metadata = {}
        if circuits_metadata is not None:
            circuit_meta = circuits_metadata[idx]
            if circuit_meta is not None:
                pub_metadata["circuit_metadata"] = circuit_meta

        pub_result = EstimatorPubResult(data=data_bin, metadata=pub_metadata)
        pub_results.append(pub_result)

    return PrimitiveResult(pub_results, metadata=result.metadata or {})
