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

"""Options for the executor-based EstimatorV2."""

from __future__ import annotations

from pydantic import Field
from pydantic.dataclasses import dataclass

from .environment_options import EnvironmentOptions
from ....options.executor_options import ExecutorOptions, ExecutionOptions


@dataclass
class EstimatorExecutionOptions(ExecutionOptions):
    """Execution options for the estimator primitive.

    Args:
        init_qubits: Whether to reset the qubits to the ground state for each shot.
            Inherited from :class:`~qiskit_ibm_runtime.options.executor_options.ExecutionOptions`.
        rep_delay: The repetition delay. Inherited from
            :class:`~qiskit_ibm_runtime.options.executor_options.ExecutionOptions`.
    """

    # Inherits init_qubits and rep_delay from ExecutionOptions
    # No meas_type needed (always classified for estimator)

    def to_executor_options(self) -> ExecutionOptions:
        """Convert to execution options.

        Returns:
            ExecutionOptions with the same field values.
        """
        return ExecutionOptions(**vars(self))


@dataclass
class EstimatorOptions:
    """Options for the executor-based EstimatorV2.

    This is a minimal implementation without twirling, dynamical decoupling,
    or error mitigation features.
    """

    default_precision: float = 0.015625
    """The default precision for expectation value estimates if not specified in the PUBs
    or in the run method."""

    execution: EstimatorExecutionOptions = Field(default_factory=EstimatorExecutionOptions)
    """Execution options.

    See :class:`.EstimatorExecutionOptions` for all available options."""

    experimental: dict | None = None
    """Experimental options."""

    max_execution_time: int | None = None
    """Maximum execution time in seconds, based on system execution time (not wall clock time)."""

    environment: EnvironmentOptions = Field(default_factory=EnvironmentOptions)
    """Options related to the execution environment."""

    def to_executor_options(self) -> ExecutorOptions:
        """Map EstimatorOptions to ExecutorOptions, ignoring all irrelevant fields.

        Returns:
            Mapped executor options.
        """
        executor_options = ExecutorOptions()

        executor_options.environment = self.environment.to_executor_options()
        executor_options.execution = self.execution.to_executor_options()
        executor_options.environment.max_execution_time = self.max_execution_time

        if self.experimental:
            executor_options.environment.image = self.experimental.pop("image", None)
            executor_options.experimental.update(self.experimental)

        return executor_options
