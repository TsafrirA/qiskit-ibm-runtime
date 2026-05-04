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

"""Unit tests for EstimatorV2 options."""

import unittest

from qiskit_ibm_runtime.executor.routines.options.estimator_options import (
    EstimatorOptions,
    EstimatorExecutionOptions,
)
from qiskit_ibm_runtime.executor.routines.options.environment_options import EnvironmentOptions
from qiskit_ibm_runtime.options.executor_options import ExecutorOptions, ExecutionOptions


class TestEstimatorExecutionOptions(unittest.TestCase):
    """Tests for EstimatorExecutionOptions."""

    def test_default_values(self):
        """Test default values."""
        options = EstimatorExecutionOptions()
        self.assertTrue(options.init_qubits)
        self.assertIsNone(options.rep_delay)

    def test_set_values(self):
        """Test setting values."""
        options = EstimatorExecutionOptions(init_qubits=True, rep_delay=0.001)
        self.assertTrue(options.init_qubits)
        self.assertEqual(options.rep_delay, 0.001)

    def test_to_executor_options(self):
        """Test conversion to ExecutorOptions."""
        options = EstimatorExecutionOptions(init_qubits=True, rep_delay=0.001)
        executor_options = options.to_executor_options()
        self.assertIsInstance(executor_options, ExecutionOptions)
        self.assertTrue(executor_options.init_qubits)
        self.assertEqual(executor_options.rep_delay, 0.001)


class TestEstimatorOptions(unittest.TestCase):
    """Tests for EstimatorOptions."""

    def test_default_values(self):
        """Test default values."""
        options = EstimatorOptions()
        self.assertEqual(options.default_precision, 0.015625)
        self.assertIsInstance(options.execution, EstimatorExecutionOptions)
        self.assertIsNone(options.experimental)
        self.assertIsNone(options.max_execution_time)
        self.assertIsInstance(options.environment, EnvironmentOptions)

    def test_set_default_precision(self):
        """Test setting default_precision."""
        options = EstimatorOptions(default_precision=0.01)
        self.assertEqual(options.default_precision, 0.01)

    def test_set_execution_options(self):
        """Test setting execution options."""
        exec_options = EstimatorExecutionOptions(init_qubits=True)
        options = EstimatorOptions(execution=exec_options)
        self.assertTrue(options.execution.init_qubits)

    def test_set_max_execution_time(self):
        """Test setting max_execution_time."""
        options = EstimatorOptions(max_execution_time=300)
        self.assertEqual(options.max_execution_time, 300)

    def test_set_experimental(self):
        """Test setting experimental options."""
        options = EstimatorOptions(experimental={"test_option": "value"})
        self.assertEqual(options.experimental, {"test_option": "value"})

    def test_to_executor_options(self):
        """Test conversion to ExecutorOptions."""
        options = EstimatorOptions(
            default_precision=0.022097,
            max_execution_time=300,
        )
        options.execution.init_qubits = True
        options.execution.rep_delay = 0.001

        executor_options = options.to_executor_options()

        self.assertIsInstance(executor_options, ExecutorOptions)
        self.assertTrue(executor_options.execution.init_qubits)
        self.assertEqual(executor_options.execution.rep_delay, 0.001)
        self.assertEqual(executor_options.environment.max_execution_time, 300)

    def test_to_executor_options_with_experimental(self):
        """Test conversion with experimental options."""
        options = EstimatorOptions(experimental={"image": "custom_image", "other": "value"})

        executor_options = options.to_executor_options()

        self.assertEqual(executor_options.environment.image, "custom_image")
        self.assertEqual(executor_options.experimental.get("other"), "value")

    def test_no_twirling_options(self):
        """Test that EstimatorOptions doesn't have twirling options."""
        options = EstimatorOptions()
        self.assertFalse(hasattr(options, "twirling"))

    def test_no_dynamical_decoupling_options(self):
        """Test that EstimatorOptions doesn't have dynamical decoupling options."""
        options = EstimatorOptions()
        self.assertFalse(hasattr(options, "dynamical_decoupling"))

    def test_no_meas_type_in_execution(self):
        """Test that EstimatorExecutionOptions doesn't have meas_type."""
        options = EstimatorExecutionOptions()
        self.assertFalse(hasattr(options, "meas_type"))
