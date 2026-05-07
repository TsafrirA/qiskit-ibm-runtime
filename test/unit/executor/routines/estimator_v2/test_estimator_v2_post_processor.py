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

"""Unit tests for EstimatorV2 post-processor."""

import unittest
import numpy as np

from qiskit.primitives import PrimitiveResult

from qiskit_ibm_runtime.quantum_program.quantum_program_result import QuantumProgramResult
from qiskit_ibm_runtime.executor.routines.estimator_v2.post_processors.v0_1.post_processor import (
    estimator_v2_post_processor_v0_1,
)
from qiskit_ibm_runtime.utils.estimator_pub_result import EstimatorPubResult


class TestEstimatorV2PostProcessor(unittest.TestCase):
    """Tests for estimator_v2_post_processor_v0_1."""

    def test_post_processor_single_pub_single_observable(self):
        """Test post-processor with single pub and single observable."""
        # Create mock measurement data
        # Shape: (num_randomizations, num_bases, shots, num_qubits)
        # For ZZ observable, we need 1 basis (Z basis)
        meas_data = np.array(
            [
                [
                    # Basis 0 (Z basis): 10 shots, 2 qubits
                    [
                        [False, False],  # 00 -> +1
                        [False, False],  # 00 -> +1
                        [False, True],  # 01 -> -1
                        [True, False],  # 10 -> -1
                        [True, True],  # 11 -> +1
                        [False, False],  # 00 -> +1
                        [False, False],  # 00 -> +1
                        [False, False],  # 00 -> +1
                        [False, False],  # 00 -> +1
                        [False, False],  # 00 -> +1
                    ]
                ]
            ]
        )

        # Create QuantumProgramResult
        result_data = [{"wrapper_estimator_data": meas_data}]

        passthrough_data = {
            "post_processor": {
                "version": "v0.1",
                "circuits_metadata": [None],
            },
            "observables": [[{"ZZ": 1.0}]],
            "measure_bases": [["ZZ"]],
        }

        quantum_result = QuantumProgramResult(
            data=result_data, metadata=None, passthrough_data=passthrough_data
        )
        quantum_result._semantic_role = "estimator_v2"

        # Run post-processor
        primitive_result = estimator_v2_post_processor_v0_1(quantum_result)

        # Verify result structure
        self.assertIsInstance(primitive_result, PrimitiveResult)
        self.assertEqual(len(primitive_result), 1)

        pub_result = primitive_result[0]
        self.assertIsInstance(pub_result, EstimatorPubResult)

        # Verify expectation value
        # 8 * (+1) + 2 * (-1) = 6, average = 6/10 = 0.6
        self.assertAlmostEqual(pub_result.data.evs[0], 0.6)

    def test_post_processor_multiple_observables(self):
        """Test post-processor with multiple observables."""
        # Create mock measurement data for two bases
        # Shape: (num_randomizations, num_bases, shots, num_qubits)
        # Basis 0: Z basis for ZZ observable
        # Basis 1: X basis for XX observable
        meas_data = np.array(
            [
                [
                    # Basis 0 (Z basis): all 00
                    [[False, False]] * 10,
                    # Basis 1 (X basis): all 00
                    [[False, False]] * 10,
                ]
            ]
        )

        result_data = [{"wrapper_estimator_data": meas_data}]

        passthrough_data = {
            "post_processor": {
                "version": "v0.1",
                "circuits_metadata": [None],
            },
            "observables": [[{"ZZ": 1.0}, {"XX": 1.0}]],
            "measure_bases": [["ZZ", "XX"]],
        }

        quantum_result = QuantumProgramResult(
            data=result_data, metadata=None, passthrough_data=passthrough_data
        )
        quantum_result._semantic_role = "estimator_v2"

        primitive_result = estimator_v2_post_processor_v0_1(quantum_result)

        # Verify we have expectation values for both observables
        pub_result = primitive_result[0]
        self.assertEqual(len(pub_result.data.evs), 2)

    def test_post_processor_with_coefficients(self):
        """Test post-processor with observable coefficients."""
        # ZZ observable with coefficient 2.0
        # Shape: (num_randomizations, num_bases, shots, num_qubits)
        meas_data = np.array([[[[False, False]] * 10]])  # All 00 -> +1

        result_data = [{"wrapper_estimator_data": meas_data}]

        passthrough_data = {
            "post_processor": {
                "version": "v0.1",
                "circuits_metadata": [None],
            },
            "observables": [[{"ZZ": 2.0}]],  # Coefficient 2.0
            "measure_bases": [["ZZ"]],
        }

        quantum_result = QuantumProgramResult(
            data=result_data, metadata=None, passthrough_data=passthrough_data
        )
        quantum_result._semantic_role = "estimator_v2"

        primitive_result = estimator_v2_post_processor_v0_1(quantum_result)

        # Expectation value should be 2.0 * 1.0 = 2.0
        self.assertAlmostEqual(primitive_result[0].data.evs[0], 2.0)

    def test_post_processor_multiple_pubs(self):
        """Test post-processor with multiple pubs."""
        # Create data for two pubs
        # Shape: (num_randomizations, num_bases, shots, num_qubits)
        meas_data_1 = np.array([[[[False, False]] * 10]])
        meas_data_2 = np.array([[[[True, True]] * 10]])

        result_data = [
            {"wrapper_estimator_data": meas_data_1},
            {"wrapper_estimator_data": meas_data_2},
        ]

        passthrough_data = {
            "post_processor": {
                "version": "v0.1",
                "circuits_metadata": [None, None],
            },
            "observables": [[{"ZZ": 1.0}], [{"ZZ": 1.0}]],
            "measure_bases": [["ZZ"], ["ZZ"]],
        }

        quantum_result = QuantumProgramResult(
            data=result_data, metadata=None, passthrough_data=passthrough_data
        )
        quantum_result._semantic_role = "estimator_v2"

        primitive_result = estimator_v2_post_processor_v0_1(quantum_result)

        # Verify two pub results
        self.assertEqual(len(primitive_result), 2)

        # First pub: all 00 -> +1
        self.assertAlmostEqual(primitive_result[0].data.evs[0], 1.0)

        # Second pub: all 11 -> +1
        self.assertAlmostEqual(primitive_result[1].data.evs[0], 1.0)

    def test_post_processor_with_parameter_sweep(self):
        """Test post-processor with parameter sweep."""
        # Create data with parameter sweep dimension
        # Shape: (num_randomizations, num_param_values, num_bases, shots, num_qubits)
        meas_data = np.array(
            [
                [
                    [
                        # Parameter value 0: all 00
                        [[False, False]] * 5,
                    ],
                    [
                        # Parameter value 1: all 11
                        [[True, True]] * 5,
                    ],
                ]
            ]
        )

        result_data = [{"wrapper_estimator_data": meas_data}]

        passthrough_data = {
            "post_processor": {
                "version": "v0.1",
                "circuits_metadata": [None],
            },
            "observables": [[{"ZZ": 1.0}]],
            "measure_bases": [["ZZ"]],
        }

        quantum_result = QuantumProgramResult(
            data=result_data, metadata=None, passthrough_data=passthrough_data
        )
        quantum_result._semantic_role = "estimator_v2"

        primitive_result = estimator_v2_post_processor_v0_1(quantum_result)

        # Verify shape: param_shape=(2,), obs_shape=(1,) -> broadcast to (2,)
        evs = primitive_result[0].data.evs
        self.assertEqual(evs.shape, (2,))

        # First parameter value: all 00 -> +1
        self.assertAlmostEqual(evs[0], 1.0)

        # Second parameter value: all 11 -> +1
        self.assertAlmostEqual(evs[1], 1.0)

    def test_post_processor_missing_passthrough_data(self):
        """Test post-processor raises error with missing passthrough data."""
        result_data = [{"wrapper_estimator_data": np.array([[[False]]])}]

        # Missing post_processor in passthrough_data
        quantum_result = QuantumProgramResult(data=result_data, metadata=None, passthrough_data={})

        with self.assertRaises(ValueError) as context:
            estimator_v2_post_processor_v0_1(quantum_result)

        self.assertIn("post_processor", str(context.exception))

    def test_post_processor_missing_observables(self):
        """Test post-processor raises error with missing observables."""
        result_data = [{"wrapper_estimator_data": np.array([[[False]]])}]

        passthrough_data = {
            "post_processor": {
                "version": "v0.1",
            },
            # Missing observables
        }

        quantum_result = QuantumProgramResult(
            data=result_data, metadata=None, passthrough_data=passthrough_data
        )

        with self.assertRaises(ValueError) as context:
            estimator_v2_post_processor_v0_1(quantum_result)

        self.assertIn("observables", str(context.exception))

    def test_post_processor_empty_result(self):
        """Test post-processor with empty result."""
        result_data = []
        quantum_result = QuantumProgramResult(data=result_data, metadata=None, passthrough_data={})

        primitive_result = estimator_v2_post_processor_v0_1(quantum_result)

        # Should return empty PrimitiveResult
        self.assertIsInstance(primitive_result, PrimitiveResult)
        self.assertEqual(len(primitive_result), 0)

    def test_post_processor_with_circuit_metadata(self):
        """Test post-processor preserves circuit metadata."""
        # Shape: (num_randomizations, num_bases, shots, num_qubits)
        meas_data = np.array([[[[False, False]] * 10]])

        result_data = [{"wrapper_estimator_data": meas_data}]

        circuit_metadata = {"experiment_id": "test_123", "custom_field": "value"}

        passthrough_data = {
            "post_processor": {
                "version": "v0.1",
                "circuits_metadata": [circuit_metadata],
            },
            "observables": [[{"ZZ": 1.0}]],
            "measure_bases": [["ZZ"]],
        }

        quantum_result = QuantumProgramResult(
            data=result_data, metadata=None, passthrough_data=passthrough_data
        )
        quantum_result._semantic_role = "estimator_v2"

        primitive_result = estimator_v2_post_processor_v0_1(quantum_result)

        # Verify circuit metadata is preserved
        pub_result = primitive_result[0]
        self.assertIn("circuit_metadata", pub_result.metadata)
        self.assertEqual(pub_result.metadata["circuit_metadata"], circuit_metadata)

    def test_post_processor_complex_broadcasting_with_checkerboard(self):
        """Test post-processor with complex broadcasting and checkerboard observable pattern."""
        # param_shape = (3, 4, 1, 1), obs_shape = (4, 3)
        # broadcast((3,4,1,1), (4,3)) = (3,4,4,3)

        # Create measurement data:
        # (num_randomizations, 3, 4, 1, 1, num_bases, shots, qubits)
        # Use 1 basis for simplicity (all observables are ZZ)
        meas_data = np.zeros((1, 3, 4, 1, 1, 1, 10, 2), dtype=bool)

        # Fill with a pattern: param (i,j,0,0) gives measurement based on (i+j) % 4
        for i in range(3):
            for j in range(4):
                pattern = (i + j) % 4
                if pattern == 0:  # 00 -> +1
                    meas_data[0, i, j, 0, 0, 0, :, :] = False
                elif pattern == 1:  # 01 -> -1
                    meas_data[0, i, j, 0, 0, 0, :, 0] = False
                    meas_data[0, i, j, 0, 0, 0, :, 1] = True
                elif pattern == 2:  # 10 -> -1
                    meas_data[0, i, j, 0, 0, 0, :, 0] = True
                    meas_data[0, i, j, 0, 0, 0, :, 1] = False
                else:  # 11 -> +1
                    meas_data[0, i, j, 0, 0, 0, :, :] = True

        result_data = [{"wrapper_estimator_data": meas_data}]

        # Create 4x3 observables array with checkerboard coefficients
        # Coefficient is +1 if (i+j) is even, -1 if odd
        # Need to structure as nested list to get shape (4, 3)
        observables = [
            [
                [{"ZZ": 1.0}, {"ZZ": -1.0}, {"ZZ": 1.0}],  # row 0: +, -, +
                [{"ZZ": -1.0}, {"ZZ": 1.0}, {"ZZ": -1.0}],  # row 1: -, +, -
                [{"ZZ": 1.0}, {"ZZ": -1.0}, {"ZZ": 1.0}],  # row 2: +, -, +
                [{"ZZ": -1.0}, {"ZZ": 1.0}, {"ZZ": -1.0}],  # row 3: -, +, -
            ]
        ]

        passthrough_data = {
            "post_processor": {
                "version": "v0.1",
                "circuits_metadata": [None],
            },
            "observables": observables,
            "measure_bases": [["ZZ"]],
        }

        quantum_result = QuantumProgramResult(
            data=result_data, metadata=None, passthrough_data=passthrough_data
        )
        quantum_result._semantic_role = "estimator_v2"

        primitive_result = estimator_v2_post_processor_v0_1(quantum_result)

        # Verify shape: broadcast((3,4,1,1), (4,3)) = (3,4,4,3)
        evs = primitive_result[0].data.evs
        self.assertEqual(evs.shape, (3, 4, 4, 3))

        # Verify specific values considering both measurement pattern and coefficient pattern
        # For obs at flat index k (which maps to (obs_i, obs_j) in 4x3 grid),
        # param at position (param_i, param_j, 0, 0):
        # - measurement gives: +1 if (param_i+param_j)%4 in {0,3}, -1 if in {1,2}
        # - coefficient is: +1 if (obs_i+obs_j) even, -1 if odd
        # - final value is: measurement * coefficient

        # Example: param (0,0,0,0), obs (0,0)
        # measurement: (0+0)%4=0 -> 00 -> +1
        # coefficient: (0+0) even -> +1
        # result: +1 * +1 = +1
        self.assertAlmostEqual(evs[0, 0, 0, 0], 1.0)

        # Example: param (0,1,0,0), obs (0,1)
        # measurement: (0+1)%4=1 -> 01 -> -1
        # coefficient: (0+1) odd -> -1
        # result: -1 * -1 = +1
        self.assertAlmostEqual(evs[0, 1, 0, 1], 1.0)

        # Example: param (1,0,0,0), obs (1,0)
        # measurement: (1+0)%4=1 -> 01 -> -1
        # coefficient: (1+0) odd -> -1
        # result: -1 * -1 = +1
        self.assertAlmostEqual(evs[1, 0, 1, 0], 1.0)

        # Example: param (2,2,0,0), obs (2,2)
        # measurement: (2+2)%4=0 -> 00 -> +1
        # coefficient: (2+2) even -> +1
        # result: +1 * +1 = +1
        self.assertAlmostEqual(evs[2, 2, 2, 2], 1.0)

        # Example: param (1,2,0,0), obs (3,1)
        # measurement: (1+2)%4=3 -> 11 -> +1
        # coefficient: (3+1) even -> +1
        # result: +1 * +1 = +1
        self.assertAlmostEqual(evs[1, 2, 3, 1], 1.0)

        # Example: param (0,2,0,0), obs (1,2)
        # measurement: (0+2)%4=2 -> 10 -> -1
        # coefficient: (1+2) odd -> -1
        # result: -1 * -1 = +1
        self.assertAlmostEqual(evs[0, 2, 1, 2], 1.0)

        # Example with different result: param (0,1,0,0), obs (0,0)
        # measurement: (0+1)%4=1 -> 01 -> -1
        # coefficient: (0+0) even -> +1
        # result: -1 * +1 = -1
        self.assertAlmostEqual(evs[0, 1, 0, 0], -1.0)
