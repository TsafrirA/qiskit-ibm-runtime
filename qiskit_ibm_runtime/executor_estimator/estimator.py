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

"""Executor-based EstimatorV2 primitive."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import logging

import numpy as np
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from qiskit.providers import BackendV2
from samplomatic import build
from samplomatic.transpiler import generate_boxing_pass_manager
from qiskit.circuit import ClassicalRegister
from qiskit.circuit.exceptions import CircuitError

from ..runtime_job_v2 import RuntimeJobV2
from ..executor import Executor
from ..session import Session
from ..batch import Batch
from ..quantum_program import QuantumProgram
from ..quantum_program.quantum_program import SamplexItem
from ..quantum_program.datatree import is_datatree_compatible
from qiskit_ibm_runtime.options_models.executor_options import ExecutorOptions
from ..exceptions import IBMInputValueError
from ..options_models.estimator_options import EstimatorOptions
from ..options_models.twirling_options import TwirlingOptions
from .helpers import get_bases, pauli_to_ints
from ..executor_sampler.utils import resolve_precision, calculate_twirling_shots

logger = logging.getLogger(__name__)


def prepare(
    pubs: list[EstimatorPub],
    twirling_options: TwirlingOptions,
    shots: int,
) -> QuantumProgram:
    """Convert a list of ``EstimatorPub`` objects to a ``QuantumProgram``.

    Args:
        pubs: List of estimator pubs to convert.
        twirling_options: ``TwirlingOptions`` object.
        shots: The number of shots to use. Will be overridden by
            `num_randomizations * shots_per_randomization` when both are specified explicitly
            and twirling is on.

    Returns:
        :class:`~.QuantumProgram` with :class:`~.SamplexItem` objects for each pub,
        with passthrough_data configured for
        :class:`~qiskit_ibm_runtime.executor_estimator.estimator.EstimatorV2` post-processing.

    Raises:
        IBMInputValueError: If pubs have mismatched precision,
            if a circuit contains mid-circuit measurements, or if a circuit already uses the
            reserved classical register name ``wrapper_estimator_data``.
    """
    if twirling_options.enable_gates or twirling_options.enable_measure:
        num_randomizations, shots_per_randomization = calculate_twirling_shots(
            shots,
            twirling_options.num_randomizations,
            twirling_options.shots_per_randomization,
        )
    else:
        num_randomizations = 1
        shots_per_randomization = int(shots)

    # Create items
    items: list[SamplexItem] = []
    observables_list = []
    measure_bases_list = []

    for i, pub in enumerate(pubs):
        logger.info("Processing pub %d/%d", i + 1, len(pubs))

        # Determine measurement bases
        measure_bases = get_bases(pub.observables)

        # Remove any existing final measurements
        prepared_circuit = pub.circuit.remove_final_measurements(inplace=False)

        # TODO: Adjust so change basis is applied only to the last box.
        if prepared_circuit.count_ops().get("measure", 0) > 0:
            raise IBMInputValueError(
                f"Pub {i} contains mid-circuit measurements, which are temporarily not supported"
                " by EstimatorV2. Only final measurements are allowed."
            )

        # TODO: Optimization - We can measure only the needed qubits.
        # TODO: Optimization - We can remove the old classical registers which are not needed,
        # to minimize the returned data.
        creg = ClassicalRegister(prepared_circuit.num_qubits, "wrapper_estimator_data")
        try:
            prepared_circuit.add_register(creg)
        except CircuitError:
            raise IBMInputValueError(
                "Name `wrapper_estimator_data` is reserved for a dedicated classical register."
            )

        prepared_circuit.measure(prepared_circuit.qubits, creg)

        boxing_pm = generate_boxing_pass_manager(
            enable_gates=twirling_options.enable_gates,
            enable_measures=True,
            twirling_strategy=twirling_options.strategy.replace("-", "_"),
            measure_annotations="all" if twirling_options.enable_measure else "change_basis",
        )
        prepared_circuit = boxing_pm.run(prepared_circuit)

        template, samplex = build(prepared_circuit)

        # Prepare samplex_arguments
        if pub.parameter_values.num_parameters > 0:
            param_array = pub.parameter_values.as_array()
            param_shape = pub.parameter_values.shape
            samplex_args = {
                "parameter_values": param_array.reshape(
                    param_shape + (1, pub.parameter_values.num_parameters)
                )
            }
        else:
            samplex_args = {}
            param_shape = ()

        # Item shape: (num_randomizations,) + param_shape + (num_bases,)
        item_shape = (num_randomizations,) + param_shape + (len(measure_bases),)

        # Add basis changes to samplex_arguments
        basis_changes_specs = samplex.inputs().get_specs("basis_changes")
        basis_changes_name = basis_changes_specs[0].name
        # Create basis array with shape (num_bases, num_qubits)
        measure_bases_int = np.array([pauli_to_ints(basis) for basis in measure_bases])

        samplex_arguments = samplex.inputs().make_broadcastable()
        samplex_arguments.bind(**{**samplex_args, basis_changes_name: measure_bases_int})

        # Create SamplexItem
        items.append(
            SamplexItem(
                circuit=template,
                samplex=samplex,
                samplex_arguments=samplex_arguments,
                shape=item_shape,
            )
        )

        # Store data for passthrough
        observables_list.append(pub.observables.tolist())
        measure_bases_list.append(measure_bases.to_labels())

    # Collect circuit metadata from each pub
    circuits_metadata = [pub.circuit.metadata for pub in pubs]

    # Validate that circuit metadata is compatible with DataTree format
    for idx, metadata in enumerate(circuits_metadata):
        if metadata is not None and not is_datatree_compatible(metadata):
            raise IBMInputValueError(
                f"Circuit metadata at index {idx} is not compatible with DataTree format. "
                f"Metadata must be a nested structure of lists, dicts (with string keys), "
                f"numpy arrays, or primitive types (str, int, float, bool, None)."
            )

    passthrough_data = {
        "post_processor": {
            "version": "v0.1",
            "circuits_metadata": circuits_metadata,
        },
        "observables": observables_list,
        "measure_bases": measure_bases_list,
    }

    # Create QuantumProgram
    quantum_program = QuantumProgram(
        shots=shots_per_randomization,
        items=items,
        passthrough_data=passthrough_data,
    )

    # Set semantic role for post-processing dispatch
    quantum_program._semantic_role = "estimator_v2"

    return quantum_program


class EstimatorV2(BaseEstimatorV2):
    """Executor-based EstimatorV2 primitive for Qiskit Runtime.

    This is an implementation of EstimatorV2 built on top of the Executor primitive,
    enabling transparent client-side processing with faster feedback loops and greater
    user control.

    **Limitations:**

    - Circuits must not contain BoxOp instructions
    - No twirling support in Phase 1
    - No dynamical decoupling in Phase 1
    - No error mitigation in Phase 1

    **Custom Prepare Function:**

    You can inject a custom prepare function to replace the default conversion logic
    from EstimatorPub objects to QuantumProgram. The custom function must have the
    following signature:

    ```python

        def my_prepare(
            pubs: list[EstimatorPub],
            options: EstimatorOptions,
            shots: int | None = None,
        ) -> tuple[QuantumProgram, ExecutorOptions]:
            ...
    ```

    The custom function can be provided either at initialization via the ``custom_prepare``
    parameter or later via the ``custom_prepare`` property. Set to ``None`` to restore
    the default prepare function.

    Example:
        .. code-block:: python

            from qiskit import QuantumCircuit
            from qiskit.quantum_info import SparsePauliOp
            from qiskit_ibm_runtime import QiskitRuntimeService
            from qiskit_ibm_runtime.executor_estimator import EstimatorV2

            service = QiskitRuntimeService()
            backend = service.least_busy(operational=True, simulator=False)

            # Create a simple circuit
            circuit = QuantumCircuit(2)
            circuit.h(0)
            circuit.cx(0, 1)

            # Define observable
            observable = SparsePauliOp.from_list([("ZZ", 1), ("XX", 1)])

            # Run the estimator with options
            estimator = EstimatorV2(mode=backend)
            estimator.options.default_precision = 0.01
            estimator.options.execution.init_qubits = True
            job = estimator.run([(circuit, observable)])
            result = job.result()

            # Example with custom prepare function
            def my_prepare(pubs, options, precision=None):
                # Custom logic here
                ...
                return quantum_program, executor_options

            estimator = EstimatorV2(mode=backend, custom_prepare=my_prepare)
            # Or set it later:
            # estimator.custom_prepare = my_prepare

    Args:
        mode: The execution mode used to make the primitive query. It can be:

            * A :class:`~qiskit.providers.BackendV2` if you are using job mode.
            * A :class:`~qiskit_ibm_runtime.Session` if you are using session execution mode.
            * A :class:`~qiskit_ibm_runtime.Batch` if you are using batch execution mode.

            Refer to the `Qiskit Runtime documentation
            <https://quantum.cloud.ibm.com/docs/guides/execution-modes>`_
            for more information about execution modes.

        options: Estimator options.
            See
            :class:`~qiskit_ibm_runtime.options_models.estimator_options.EstimatorOptions`
            for all available options.
        custom_prepare: Optional custom prepare function to replace the default conversion
            logic. If ``None``, the default function is used.
    """

    def __init__(
        self,
        mode: BackendV2 | Session | Batch | None = None,
        options: EstimatorOptions | dict | None = None,
        custom_prepare: (
            Callable[
                [list[EstimatorPub], EstimatorOptions, float | None],
                tuple[QuantumProgram, ExecutorOptions],
            ]
            | None
        ) = None,
    ):
        super().__init__()

        self._executor = Executor(mode=mode)

        # Initialize options
        if options is None:
            self._options = EstimatorOptions()
        elif isinstance(options, dict):
            self._options = EstimatorOptions(**options)
        else:
            self._options = options

        self._custom_prepare = custom_prepare

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> RuntimeJobV2:
        """Submit a request to the estimator primitive.

        For moderate and complex workloads, the client-side processing done to map estimator inputs
        to executor inputs can be resource intensive and cause a delay between invoking the function
        and the ``job`` being submitted. In order to check the progress of the call, it is
        recommended to setup logging (with an ``INFO`` level) - see
        `Qiskit Runtime documentation
        <https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/runtime-service#logging>`_
        for more information.

        Args:
            pubs: An iterable of pub-like objects. For example, a list of circuits
                  and observables or tuples ``(circuit, observables, parameter_values)``.
            precision: The target precision for expectation value estimates of each
                       estimator pub that does not specify its own precision. If ``None``,
                       the value from ``options.default_precision`` will be used.

        Returns:
            The submitted job.

        Raises:
            ValueError: If backend is not provided.
            IBMInputValueError: If precision is not properly specified or if unsupported
                options are detected.
        """
        # Coerce pubs to EstimatorPub objects
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]

        # Convert pubs to QuantumProgram and map options using the selected prepare function
        logger.info("Starting pre-processing")

        # Use the correct prepare function
        if self._custom_prepare is not None:
            # Use custom prepare function without validation
            quantum_program, executor_options = self._custom_prepare(
                coerced_pubs, self._options, precision
            )
        else:
            resolved_precision = resolve_precision(coerced_pubs, precision)
            if resolved_precision is not None:
                shots = int(np.ceil(1.0 / (resolved_precision**2)))
            elif self.options.default_shots is not None:
                shots = int(self.options.default_shots)
            else:
                shots = int(np.ceil(1.0 / (self.options.default_precision**2)))

            quantum_program = prepare(coerced_pubs, self.options.twirling, shots)
            executor_options = self.options.to_executor_options()

        # Set executor options
        self._executor.options = executor_options

        # Submit to executor
        logger.info(
            "Submitting %d pub%s to executor with %d shots",
            len(coerced_pubs),
            "s" if len(coerced_pubs) > 1 else "",
            quantum_program.shots,
        )

        return self._executor.run(quantum_program)

    @property
    def options(self) -> EstimatorOptions:
        """Return the options.

        Returns:
            The estimator options.
        """
        return self._options

    @property
    def custom_prepare(
        self,
    ) -> (
        Callable[
            [list[EstimatorPub], EstimatorOptions, float | None],
            tuple[QuantumProgram, ExecutorOptions],
        ]
        | None
    ):
        """Return the custom prepare function.

        Returns:
            The custom prepare function, or None if using default dispatching.
        """
        return self._custom_prepare

    @custom_prepare.setter
    def custom_prepare(
        self,
        fn: (
            Callable[
                [list[EstimatorPub], EstimatorOptions, float | None],
                tuple[QuantumProgram, ExecutorOptions],
            ]
            | None
        ),
    ) -> None:
        """Set the custom prepare function.

        Args:
            fn: The prepare function to use. Pass None to restore default dispatching.

        Raises:
            TypeError: If fn is not None and not callable.
        """
        if fn is not None and not callable(fn):
            raise TypeError(f"custom_prepare must be callable or None, got {type(fn).__name__}")
        self._custom_prepare = fn
