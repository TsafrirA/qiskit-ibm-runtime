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


def _broadcast_expectation_values(
    exp_vals_dict: dict,
    stds_dict: dict,
    param_shape: tuple,
    obs_shape: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast expectation values and standard deviations to output shape.

    This function handles broadcasting of parameter and observable shapes to create
    the final output arrays. It supports all combinations:
    - Scalar parameters + scalar observables
    - Scalar parameters + array observables
    - Array parameters + scalar observables
    - Array parameters + array observables

    Args:
        exp_vals_dict: Dictionary mapping observable indices to expectation value arrays
                      with shape param_shape.
        stds_dict: Dictionary mapping observable indices to standard deviation arrays
                  with shape param_shape.
        param_shape: Shape of parameter sweep (empty tuple for scalar).
        obs_shape: Shape of observables array (empty tuple for scalar).

    Returns:
        Tuple of (expectation_values, standard_deviations), each with shape
        np.broadcast_shapes(param_shape, obs_shape).
    """
    output_shape = np.broadcast_shapes(param_shape, obs_shape)

    # Handle the simple scalar case efficiently
    if param_shape == () and obs_shape == ():
        # Both scalar: just extract the single values
        ev = exp_vals_dict[()].item()
        std = stds_dict[()].item()
        return np.array(ev), np.array(std)

    # For non-scalar cases, use numpy's advanced indexing
    evs_array = np.empty(output_shape)
    stds_array = np.empty(output_shape)

    # Create meshgrid for parameter and observable indices
    if param_shape == ():
        # Scalar parameters: all output positions map to the same param index
        param_indices = np.zeros(output_shape, dtype=int)
    else:
        param_indices = np.arange(np.prod(param_shape)).reshape(param_shape)
        # Broadcast to output shape
        param_indices = np.broadcast_to(param_indices, output_shape)

    if obs_shape == ():
        # Scalar observable: all output positions map to the same obs index
        obs_indices = np.zeros(output_shape, dtype=int)
    else:
        obs_indices = np.arange(np.prod(obs_shape)).reshape(obs_shape)
        # Broadcast to output shape
        obs_indices = np.broadcast_to(obs_indices, output_shape)

    # Fill output arrays by looking up values from dictionaries
    for out_idx in np.ndindex(output_shape):
        p_flat = param_indices[out_idx]
        o_flat = obs_indices[out_idx]

        # Convert flat indices back to multi-dimensional indices
        p_idx = np.unravel_index(p_flat, param_shape) if param_shape else ()
        o_idx = np.unravel_index(o_flat, obs_shape) if obs_shape else ()

        # Look up values from dictionaries
        ev_array = exp_vals_dict[o_idx]
        std_array = stds_dict[o_idx]

        # Extract the value at the parameter index
        if param_shape:
            evs_array[out_idx] = ev_array[p_idx]
            stds_array[out_idx] = std_array[p_idx]
        else:
            # param_shape is (), so ev_array is a 0-d array
            evs_array[out_idx] = ev_array.item()
            stds_array[out_idx] = std_array.item()

    return evs_array, stds_array


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
    if len(circuits_metadata) != len(result):
        raise ValueError(
            f"Number of circuit metadata items ({len(circuits_metadata)}) does not match "
            f"number of pubs ({len(result)})."
        )

    shots = result[0]["meas"].shape[0] * result[0]["meas"].shape[-2]

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
        # Shape: (num_randomizations,) + param_shape + (num_bases,) + (shots, num_bits)
        meas_data = item_data["meas"]
        # Apply measurement flips if present
        if "measurement_flips.meas" in item_data:
            meas_data ^= item_data["measurement_flips.meas"]

        # Extract param_shape from measurement data
        param_shape = meas_data.shape[1:-3] if meas_data.ndim > 4 else ()
        obs_shape = observables.shape

        # Compute expectation values for all observables first
        # Each exp_val has shape param_shape
        exp_vals_dict = {}
        stds_dict = {}

        for obs_idx, observable in np.ndenumerate(observables):
            exp_val = np.zeros(param_shape, dtype=float)
            variance = np.zeros(param_shape, dtype=float)

            for observable_term, coeff in observable.items():
                # Find which basis measured this term
                pauli_basis = Pauli(get_pauli_basis(observable_term))
                basis_idx = identify_measure_basis(pauli_basis, measure_bases)

                # Get measurement data for this basis
                # Shape: (num_randomizations) + param_shape + (shots, num_qubits)
                datum = meas_data[..., basis_idx, :, :]
                term_exp_val, term_variance = compute_exp_val(observable_term, datum)

                # Accumulate with coefficient
                exp_val = exp_val + coeff * term_exp_val
                variance = variance + (coeff**2) * term_variance

            exp_vals_dict[obs_idx] = exp_val
            stds_dict[obs_idx] = np.sqrt(variance / shots)  # Standard error

        # Broadcast expectation values and standard deviations to output shape
        evs_array, stds_array = _broadcast_expectation_values(
            exp_vals_dict, stds_dict, param_shape, obs_shape
        )

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
