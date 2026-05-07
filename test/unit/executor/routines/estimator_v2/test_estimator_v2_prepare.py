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

"""Unit tests for EstimatorV2 prepare function."""

import unittest
from unittest.mock import MagicMock
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives.containers.estimator_pub import EstimatorPub, ObservablesArray
from qiskit.quantum_info import SparsePauliOp

from qiskit_ibm_runtime.executor.routines.estimator_v2.estimator_v2 import prepare
from qiskit_ibm_runtime.executor.routines.options.estimator_options import EstimatorOptions
from qiskit_ibm_runtime.quantum_program import QuantumProgram
from qiskit_ibm_runtime.quantum_program.quantum_program import SamplexItem
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_ibm_runtime.exceptions import IBMInputValueError


def create_mock_backend():
    """Create a mock backend for testing."""
    backend = MagicMock()
    backend.name = "fake_backend"
    backend.target = FakeManilaV2().target
    return backend


class TestPrepareFunction(unittest.TestCase):
    """Tests for the prepare function."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = create_mock_backend()
        self.options = EstimatorOptions()

    def test_prepare_single_pub_no_parameters(self):
        """Test prepare with single pub without parameters."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        observable = SparsePauliOp.from_list([("ZZ", 1)])
        pub = EstimatorPub.coerce((circuit, observable))

        shots = 1024
        quantum_program = prepare([pub], self.options.twirling, shots)

        # Verify QuantumProgram structure
        self.assertIsInstance(quantum_program, QuantumProgram)
        self.assertEqual(quantum_program.shots, 1024)
        self.assertEqual(len(quantum_program.items), 1)
        self.assertIsInstance(quantum_program.items[0], SamplexItem)

        self.assertEqual(quantum_program.items[0].shape, (1, 1))

        # Verify passthrough data
        self.assertIn("post_processor", quantum_program.passthrough_data)
        self.assertIn("observables", quantum_program.passthrough_data)
        self.assertIn("measure_bases", quantum_program.passthrough_data)

    def test_prepare_with_parameters(self):
        """Test prepare with parametric circuit."""
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        circuit.rx(theta, 0)
        circuit.cx(0, 1)

        observable = SparsePauliOp.from_list([("ZZ", 1)])
        parameter_values = np.array([[0], [np.pi / 2], [np.pi]])
        pub = EstimatorPub.coerce((circuit, observable, parameter_values))

        shots = 1024
        quantum_program = prepare([pub], self.options.twirling, shots)

        # Verify item shape includes parameter sweep
        item = quantum_program.items[0]
        self.assertIsInstance(item, SamplexItem)
        # Shape should be (num_randomizations,) + param_shape + (num_bases,)
        # num_randomizations is 1 (no twirling)
        # param_shape is (3,) for 3 parameter sets
        # num_bases is 1 for ZZ observable
        self.assertEqual(item.shape, (1, 3, 1))

    def test_prepare_multiple_observables(self):
        """Test prepare with multiple observables."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        observables = ObservablesArray.coerce([{"ZZ": 1}, {"XX": 1}, {"YY": 1}])
        pub = EstimatorPub.coerce((circuit, observables))

        shots = 1024
        quantum_program = prepare([pub], self.options.twirling, shots)

        self.assertEqual(quantum_program.items[0].shape, (1, 3))

        # Verify observables are stored
        self.assertEqual(len(quantum_program.passthrough_data["observables"]), 1)
        self.assertEqual(len(quantum_program.passthrough_data["observables"][0]), 3)

    def test_prepare_multiple_pubs(self):
        """Test prepare with multiple pubs."""
        circuit1 = QuantumCircuit(2)
        circuit1.h(0)

        circuit2 = QuantumCircuit(3)
        circuit2.h([0, 1, 2])

        observable1 = SparsePauliOp.from_list([("ZZ", 1)])
        observable2 = SparsePauliOp.from_list([("ZZZ", 1)])

        pub1 = EstimatorPub.coerce((circuit1, observable1))
        pub2 = EstimatorPub.coerce((circuit2, observable2))

        shots = 1024
        quantum_program = prepare([pub1, pub2], self.options.twirling, shots)

        # Verify multiple items
        self.assertEqual(len(quantum_program.items), 2)

        self.assertEqual(quantum_program.items[0].shape, (1, 1))
        self.assertEqual(quantum_program.items[1].shape, (1, 1))

        self.assertEqual(len(quantum_program.passthrough_data["observables"]), 2)
        self.assertEqual(len(quantum_program.passthrough_data["measure_bases"]), 2)

    def test_prepare_uses_shots(self):
        """Test that prepare uses shots correctly."""
        circuit = QuantumCircuit(2)
        circuit.h(0)

        observable = SparsePauliOp.from_list([("ZZ", 1)])
        pub = EstimatorPub.coerce((circuit, observable))

        shots = 10000
        quantum_program = prepare([pub], self.options.twirling, shots)

        self.assertEqual(quantum_program.shots, 10000)
        self.assertEqual(quantum_program.items[0].shape, (1, 1))

    def test_prepare_with_twirling_options(self):
        """Test that prepare correctly uses twirling options."""
        circuit = QuantumCircuit(2)
        circuit.h(0)

        observable = SparsePauliOp.from_list([("ZZ", 1)])
        pub = EstimatorPub.coerce((circuit, observable))

        options = EstimatorOptions()
        shots = 1024
        quantum_program = prepare([pub], options.twirling, shots)

        self.assertEqual(quantum_program.items[0].shape, (1, 1))

    def test_prepare_passthrough_data_structure(self):
        """Test the structure of passthrough_data."""
        circuit = QuantumCircuit(2)
        circuit.h(0)

        observable = SparsePauliOp.from_list([("ZZ", 1)])
        pub = EstimatorPub.coerce((circuit, observable))

        shots = 1024
        quantum_program = prepare([pub], self.options.twirling, shots)

        self.assertEqual(quantum_program.items[0].shape, (1, 1))

        passthrough = quantum_program.passthrough_data

        # Verify post_processor metadata
        self.assertEqual(passthrough["post_processor"]["version"], "v0.1")

        # Verify semantic role is set
        self.assertEqual(quantum_program._semantic_role, "estimator_v2")

        # Verify data lists
        self.assertIsInstance(passthrough["observables"], list)
        self.assertIsInstance(passthrough["measure_bases"], list)

    def test_prepare_meas_level(self):
        """Test that prepare sets meas_level to classified."""
        circuit = QuantumCircuit(2)
        circuit.h(0)

        observable = SparsePauliOp.from_list([("ZZ", 1)])
        pub = EstimatorPub.coerce((circuit, observable))

        shots = 1024
        quantum_program = prepare([pub], self.options.twirling, shots)

        self.assertEqual(quantum_program.meas_level, "classified")
        self.assertEqual(quantum_program.items[0].shape, (1, 1))

    def test_prepare_with_mid_circuit_measurements(self):
        """Test that prepare raises error for circuits with mid-circuit measurements."""
        # Create a circuit with mid-circuit measurements
        circuit = QuantumCircuit(3, 3)
        circuit.h(0)
        circuit.cx(0, 1)
        # Add mid-circuit measurement
        circuit.measure(0, 0)
        # Continue with more gates after measurement
        circuit.h(0)
        circuit.cx(0, 2)

        observable = SparsePauliOp.from_list([("ZZZ", 1)])
        pub = EstimatorPub.coerce((circuit, observable))

        shots = 1024

        # Should raise an error - mid-circuit measurements are not supported
        with self.assertRaises(IBMInputValueError) as context:
            prepare([pub], self.options.twirling, shots)

        self.assertIn("mid-circuit measurements", str(context.exception))
