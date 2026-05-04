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

        quantum_program, executor_options = prepare([pub], self.options, 0.03125)

        # Verify QuantumProgram structure
        self.assertIsInstance(quantum_program, QuantumProgram)
        # precision=0.03125 -> shots = ceil(1/0.03125^2) = 1024
        self.assertEqual(quantum_program.shots, 1024)
        self.assertEqual(len(quantum_program.items), 1)
        self.assertIsInstance(quantum_program.items[0], SamplexItem)

        # Verify shape: param_shape=() + num_bases=1 -> (1,)
        self.assertEqual(quantum_program.items[0].shape, (1,))

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

        quantum_program, _ = prepare([pub], self.options, 0.03125)

        # Verify item shape includes parameter sweep
        item = quantum_program.items[0]
        self.assertIsInstance(item, SamplexItem)
        # Shape should be param_shape + (num_bases,)
        # param_shape is (3,) for 3 parameter sets
        # num_bases is 1 for ZZ observable
        self.assertEqual(item.shape, (3, 1))

    def test_prepare_multiple_observables(self):
        """Test prepare with multiple observables."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        observables = ObservablesArray.coerce([{"ZZ": 1}, {"XX": 1}, {"YY": 1}])
        pub = EstimatorPub.coerce((circuit, observables))

        quantum_program, _ = prepare([pub], self.options, 0.03125)

        # Verify shape: param_shape=() + num_bases=3 (ZZ, XX, YY don't commute) -> (3,)
        self.assertEqual(quantum_program.items[0].shape, (3,))

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

        quantum_program, _ = prepare([pub1, pub2], self.options, 0.03125)

        # Verify multiple items
        self.assertEqual(len(quantum_program.items), 2)

        # Verify shapes: both have param_shape=() + num_bases=1 -> (1,)
        self.assertEqual(quantum_program.items[0].shape, (1,))
        self.assertEqual(quantum_program.items[1].shape, (1,))

        self.assertEqual(len(quantum_program.passthrough_data["observables"]), 2)
        self.assertEqual(len(quantum_program.passthrough_data["measure_bases"]), 2)

    def test_prepare_uses_precision(self):
        """Test that prepare uses precision when pub doesn't specify."""
        circuit = QuantumCircuit(2)
        circuit.h(0)

        observable = SparsePauliOp.from_list([("ZZ", 1)])
        pub = EstimatorPub.coerce((circuit, observable))  # No precision

        precision = 0.01
        quantum_program, _ = prepare([pub], self.options, precision=precision)

        self.assertEqual(quantum_program.shots, 10000)
        # Verify shape
        self.assertEqual(quantum_program.items[0].shape, (1,))

    def test_prepare_options_mapping(self):
        """Test that prepare correctly maps EstimatorOptions to ExecutorOptions."""
        circuit = QuantumCircuit(2)
        circuit.h(0)

        observable = SparsePauliOp.from_list([("ZZ", 1)])
        pub = EstimatorPub.coerce((circuit, observable))

        options = EstimatorOptions()
        options.execution.init_qubits = True
        options.execution.rep_delay = 0.001
        options.max_execution_time = 300

        quantum_program, executor_options = prepare([pub], options, precision=0.03125)

        # Verify shape
        self.assertEqual(quantum_program.items[0].shape, (1,))

        # Verify options were mapped
        self.assertTrue(executor_options.execution.init_qubits)
        self.assertEqual(executor_options.execution.rep_delay, 0.001)
        self.assertEqual(executor_options.environment.max_execution_time, 300)

    def test_prepare_passthrough_data_structure(self):
        """Test the structure of passthrough_data."""
        circuit = QuantumCircuit(2)
        circuit.h(0)

        observable = SparsePauliOp.from_list([("ZZ", 1)])
        pub = EstimatorPub.coerce((circuit, observable))

        quantum_program, _ = prepare([pub], self.options, precision=0.03125)

        # Verify shape
        self.assertEqual(quantum_program.items[0].shape, (1,))

        passthrough = quantum_program.passthrough_data

        # Verify post_processor metadata
        self.assertEqual(passthrough["post_processor"]["context"], "estimator_v2")
        self.assertEqual(passthrough["post_processor"]["version"], "v0.1")

        # Verify data lists
        self.assertIsInstance(passthrough["observables"], list)
        self.assertIsInstance(passthrough["measure_bases"], list)

    def test_prepare_meas_level(self):
        """Test that prepare sets meas_level to classified."""
        circuit = QuantumCircuit(2)
        circuit.h(0)

        observable = SparsePauliOp.from_list([("ZZ", 1)])
        pub = EstimatorPub.coerce((circuit, observable))

        quantum_program, _ = prepare([pub], self.options, precision=0.03125)

        self.assertEqual(quantum_program.meas_level, "classified")
        # Verify shape
        self.assertEqual(quantum_program.items[0].shape, (1,))

    def test_prepare_rejects_mid_circuit_measurements(self):
        """Test that prepare raises error for circuits with mid-circuit measurements."""
        circuit = QuantumCircuit(3, 3)
        circuit.h(0)
        circuit.cx(0, 1)
        # Add mid-circuit measurement
        circuit.measure(0, 0)
        circuit.h(0)
        circuit.measure_all()

        observable = SparsePauliOp.from_list([("ZZZ", 1)])
        pub = EstimatorPub.coerce((circuit, observable))

        # Should raise error due to mid-circuit measurement
        with self.assertRaises(IBMInputValueError) as context:
            prepare([pub], self.options, precision=0.03125)

        self.assertIn("mid-circuit measurements", str(context.exception))
        self.assertIn("not supported", str(context.exception))

    def test_prepare_mismatched_precision_raises_error(self):
        """Test that pubs with different precision values raise an error."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        observable = SparsePauliOp.from_list([("ZZ", 1)])

        # Create pubs with different precision values
        pub1 = EstimatorPub.coerce((circuit, observable), precision=0.01)
        pub2 = EstimatorPub.coerce((circuit, observable), precision=0.02)

        with self.assertRaises(IBMInputValueError) as context:
            prepare([pub1, pub2], self.options, precision=None)

        self.assertIn("same precision", str(context.exception))
