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

from ...quantum_program.quantum_program_result import QuantumProgramResult
from ...utils.estimator_pub_result import EstimatorPubResult
from ..utils import get_pauli_basis, identify_measure_basis, compute_exp_val
from .registry import register_post_processor


def _broadcast_expectation_values(
    exp_vals_array: np.ndarray,
    stds_array: np.ndarray,
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
        exp_vals_array: Array of expectation values with shape obs_shape + param_shape.
        stds_array: Array of standard deviations with shape obs_shape + param_shape.
        param_shape: Shape of parameter sweep (empty tuple for scalar).
        obs_shape: Shape of observables array (empty tuple for scalar).

    Returns:
        Tuple of (expectation_values, standard_deviations), each with shape
        np.broadcast_shapes(param_shape, obs_shape).
    """
    output_shape = np.broadcast_shapes(param_shape, obs_shape)

    # Calculate dimensions
    num_obs = int(np.prod(obs_shape)) if obs_shape else 1
    num_params = int(np.prod(param_shape)) if param_shape else 1

    # Reshape input arrays to (num_obs,) + param_shape for easier indexing
    evs_lookup = exp_vals_array.reshape((num_obs,) + param_shape)
    stds_lookup = stds_array.reshape((num_obs,) + param_shape)

    # Create index arrays for broadcasting
    # Shape: param_shape or (1,) for scalar
    param_indices = np.arange(num_params).reshape(param_shape or (1,))
    # Shape: obs_shape or (1,) for scalar
    obs_indices = np.arange(num_obs).reshape(obs_shape or (1,))

    # Broadcast indices to output shape
    param_bc = np.broadcast_to(param_indices, output_shape)
    obs_bc = np.broadcast_to(obs_indices, output_shape)

    # Vectorized lookup using advanced indexing
    if param_shape:
        # Unravel param indices for multi-dimensional indexing
        param_unraveled = np.unravel_index(param_bc.ravel(), param_shape)
        index_tuple = (obs_bc.ravel(),) + param_unraveled
        evs_result = evs_lookup[index_tuple].reshape(output_shape)
        stds_result = stds_lookup[index_tuple].reshape(output_shape)
    else:
        # Scalar params: just index by obs
        evs_result = evs_lookup[obs_bc]
        stds_result = stds_lookup[obs_bc]

    # Handle scalar output
    if output_shape == ():
        return evs_result.item(), stds_result.item()

    return evs_result, stds_result


@register_post_processor("v0.1")
def estimator_v2_post_processor_v0_1(result: QuantumProgramResult) -> PrimitiveResult:
    """Convert a quantum program result to a primitives result, for a V2 estimator.

    Convert :class:`~.QuantumProgramResult` to a :class:`~qiskit.primitives.PrimitiveResult`,
    for :class:`~qiskit_ibm_runtime.exeutor_estimator.estimator.EstimatorV2`.

    This function transforms the raw quantum program execution results into the
    format expected by :class:`~qiskit_ibm_runtime.executor_estimator.estimator.EstimatorV2`,
    computing expectation values from measurement data and creating
    :class:`~qiskit_ibm_runtime.utils.estimator_pub_result.EstimatorPubResult` containers
    for each pub.

    Args:
        result: The raw quantum program result containing measurement data.

    Returns:
        Primitive result for
        :class:`~qiskit_ibm_runtime.executor_estimator.estimator.EstimatorV2`.
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

    shots = (
        result[0]["wrapper_estimator_data"].shape[0] * result[0]["wrapper_estimator_data"].shape[-2]
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
        # Shape: (num_randomizations,) + param_shape + (num_bases,) + (shots, num_bits)
        meas_data = item_data["wrapper_estimator_data"]
        # Apply measurement flips if present
        if "measurement_flips.wrapper_estimator_data" in item_data:
            meas_data ^= item_data["measurement_flips.wrapper_estimator_data"]

        # Extract param_shape from measurement data
        param_shape = meas_data.shape[1:-3] if meas_data.ndim > 4 else ()
        obs_shape = observables.shape

        # Compute expectation values for all observables
        # Pre-allocate arrays with shape obs_shape + param_shape
        exp_vals_array = np.zeros(obs_shape + param_shape, dtype=float)
        stds_array = np.zeros(obs_shape + param_shape, dtype=float)

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

            # Store in pre-allocated arrays
            exp_vals_array[obs_idx] = exp_val
            stds_array[obs_idx] = np.sqrt(variance / shots)  # Standard error

        # Broadcast expectation values and standard deviations to output shape
        exp_vals_array, stds_array = _broadcast_expectation_values(
            exp_vals_array, stds_array, param_shape, obs_shape
        )

        data_bin = DataBin(
            evs=exp_vals_array,
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
