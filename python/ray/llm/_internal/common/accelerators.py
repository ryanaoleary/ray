"""Shared accelerator configurations and backend abstractions for LLM serving and batch inference."""

import copy
import logging
import math
import os
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

import ray
import ray._raylet as _raylet
import ray.util.accelerators.accelerators as accelerators
from ray._private.accelerators.tpu import (
    get_chips_per_host,
    get_num_chips_from_topology,
    infer_tpu_pod_type_from_topology,
)
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.tpu import (
    RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR,
    get_tpu_nodes_for_slice,
    get_tpu_version_from_type,
    slice_placement_group,
)

logger = logging.getLogger(__name__)

# Constants for TPU batch scheduling
PARENT_ACTOR_CPU_RESERVE = 1
DEFAULT_USER_CPU_PER_HOST = 1
CPU_ACCELERATOR_TYPE_LITERAL = "CPU"

# Waiting for a TPU slice can outlast the default when the cluster has to autoscale
# one. Exposed as an env var (not public API) so operators can wait longer without
# widening the typed config surface for this alpha MVP.
DEFAULT_PG_READY_TIMEOUT_S = 180.0
SLICE_READY_TIMEOUT_ENV_VAR = "RAY_LLM_BATCH_TPU_SLICE_READY_TIMEOUT_S"

# Read by tpu_inference's TPU platform to select its Ray executor. See
# tpu_inference/platforms/tpu_jax.py.
TPU_MULTIHOST_BACKEND_ENV_VAR = "TPU_MULTIHOST_BACKEND"
TPU_MULTIHOST_BACKEND_RAY = "ray"

# Environment that the TPU engine actor and its child workers must observe.
TPU_ENGINE_ENV_VARS = {
    TPU_MULTIHOST_BACKEND_ENV_VAR: TPU_MULTIHOST_BACKEND_RAY,
    RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR: "1",
}


def _require_positive_int(value: Any, name: str) -> int:
    """Validate that a value is a positive integer (excluding bools)."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return value


def _slice_ready_timeout_s() -> float:
    """Resolve how long to wait for a TPU slice placement group to become ready."""
    raw = os.environ.get(SLICE_READY_TIMEOUT_ENV_VAR)
    if raw is None:
        return DEFAULT_PG_READY_TIMEOUT_S
    try:
        timeout_s = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{SLICE_READY_TIMEOUT_ENV_VAR} must be a number of seconds; got {raw!r}."
        ) from exc
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(
            f"{SLICE_READY_TIMEOUT_ENV_VAR} must be a finite positive number of "
            f"seconds; got {timeout_s}."
        )
    return timeout_s


def _wait_for_placement_group(pg: PlacementGroup, timeout_s: float) -> None:
    """Block until the placement group is scheduled, or raise ``GetTimeoutError``."""
    ray.get(pg.ready(), timeout=timeout_s)


class BatchResourceHandle(Protocol):
    """Protocol for driver-local batch resource handles."""

    def shutdown(self) -> None:
        """Release underlying placement groups or cluster resources."""
        ...


AcceleratorType = Enum("AcceleratorType", vars(accelerators))

# Set of TPU string values from Ray's known accelerators.
TPU_ACCELERATOR_VALUES = {
    member.value
    for name, member in AcceleratorType.__members__.items()
    if name.startswith("GOOGLE_TPU")
}


def normalize_tpu_accelerator_type(accelerator_type_str: str) -> str:
    """Normalize a TPU accelerator type string to uppercase standard form."""
    return accelerator_type_str.strip().upper().replace("_", "-")


def format_ray_accelerator_resource(accelerator_type_str: str) -> str:
    """Formats the accelerator type into a Ray custom resource string."""
    return f"accelerator_type:{accelerator_type_str}"


def infer_hardware_kind_from_bundles(
    placement_group_config: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Inspects placement group bundles and returns the inferred hardware kind."""
    if not placement_group_config:
        return None

    bundle_per_worker = placement_group_config.get("bundle_per_worker") or {}
    bundles = placement_group_config.get("bundles") or []
    all_bundles = [bundle_per_worker] + bundles

    if any(b.get("TPU", 0) > 0 for b in all_bundles):
        return "tpu"
    if any(b.get("GPU", 0) > 0 for b in all_bundles):
        return "gpu"

    # If a config was provided but lacks GPUs or TPUs, it is a CPU deployment
    return "cpu"


@dataclass(frozen=True)
class TPUReplicaLayout:
    """Resolved TPU topology layout for one model replica.

    Attributes:
        topology: The requested topology string, such as ``"4x4"`` or ``"2x4"``.
        accelerator_type: The canonical Ray accelerator type, such as ``"TPU-V6E"``.
        accelerator_version: The generation alone, such as ``"v6e"``.
        total_chips: Total chips across the topology. ``tensor_parallel_size`` must
            equal this.
        chips_per_vm: Chips on each resolved TPU VM (Ray's default ``chips_per_host`` /
            ``chips_per_vm``). With ``RAY_TPU_RESOURCE_PER_CHIP == 1``, this is also
            the Ray ``TPU`` resource each host bundle advertises.
        num_vms: Number of TPU VMs, which is also the placement-group bundle count.
    """

    topology: str
    accelerator_type: str
    accelerator_version: str
    total_chips: int
    chips_per_vm: int
    num_vms: int

    @property
    def is_single_vm(self) -> bool:
        return self.num_vms == 1


@dataclass(frozen=True)
class BatchSchedulingRequest:
    """Input request for batch scheduling strategy construction.

    Parallel-size fields default to vLLM's single-replica values (1). Callers that
    omit them intentionally get that default; TPU admission still validates TP
    against the topology chip count and rejects coerced bool/float spellings.
    """

    accelerator_type: Optional[str] = None
    accelerator_config: Optional["AcceleratorConfig"] = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    executor_backend: Optional[str] = None
    placement_group_config: Optional[Dict[str, Any]] = None
    runtime_env: Optional[Dict[str, Any]] = None
    concurrency: Any = 1

    @property
    def model_world_size(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size


@dataclass(frozen=True)
class BatchSchedulingPlan:
    """Pure configuration plan safe to embed in the lazy Ray Data dataset DAG.

    Attributes:
        map_batches_kwargs: Ray Data ``map_batches`` arguments for the engine actor.
        required_engine_env_vars: Environment variables that must already hold these
            exact values inside the engine actor before the engine initializes.
    """

    map_batches_kwargs: Dict[str, Any]
    required_engine_env_vars: Optional[Dict[str, str]] = None


@dataclass
class AcquiredBatchResources:
    """Driver-local batch resources. Never serialized or embedded in the dataset DAG."""

    plan: BatchSchedulingPlan
    close_handle: Optional[BatchResourceHandle] = None


class AcceleratorConfig(BaseModel):
    kind: str


class CPUConfig(AcceleratorConfig):
    kind: Literal["cpu"] = "cpu"


class GPUConfig(AcceleratorConfig):
    kind: Literal["gpu"] = "gpu"


class TPUConfig(AcceleratorConfig):
    kind: Literal["tpu"] = "tpu"
    topology: Optional[str] = Field(
        default=None,
        description=(
            "Physical TPU topology string (e.g. '4x4', '2x4', '2x2x2'). Required for "
            "topology-backed TPU batch inference. Ambiguous shapes such as v6e '2x4' "
            "use Ray's default chips-per-VM resolution."
        ),
    )


AnyAcceleratorConfig = Annotated[
    Union[CPUConfig, GPUConfig, TPUConfig],
    Field(discriminator="kind"),
]


class AcceleratorBackend(ABC):
    @abstractmethod
    def default_bundles(
        self,
        *,
        num_devices: int,
        accelerator_type_str: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        pass

    @abstractmethod
    def create_placement_group(
        self,
        *,
        bundles: List[Dict[str, float]],
        strategy: str,
        name: str,
        accelerator_type_str: Optional[str] = None,
    ) -> PlacementGroup:
        pass

    @property
    def requires_deferred_placement_group(self) -> bool:
        """
        If True, Ray Serve will not provision a placement group for the deployment.
        Instead, creation is deferred to the replica at runtime.
        Defaults to False.
        """
        return False

    @property
    @abstractmethod
    def requires_remote_initialization(self) -> bool:
        """Boolean indicating whether this backend needs a remote Ray task to query hardware during init."""
        pass

    @abstractmethod
    def get_remote_options(self, accelerator_type_str: str = None) -> Dict[str, Any]:
        """Returns the hardware-specific kwargs for ray.remote().options()."""
        pass

    def shutdown(self) -> None:
        """Release any resources owned by this backend. Idempotent."""
        return

    @abstractmethod
    def build_batch_scheduling_plan(
        self, request: BatchSchedulingRequest
    ) -> AcquiredBatchResources:
        """Construct the batch scheduling plan and acquire necessary driver-side resources."""
        pass


class CPUAccelerator(AcceleratorBackend):
    # stateless — no __init__
    def default_bundles(
        self, *, num_devices: int, accelerator_type_str: Optional[str] = None
    ):
        return [{"CPU": 1} for _ in range(num_devices)]

    def create_placement_group(
        self,
        *,
        bundles: List[Dict[str, float]],
        strategy: str,
        name: str,
        accelerator_type_str: Optional[str] = None,
    ):
        return placement_group(bundles=bundles, strategy=strategy, name=name)

    @property
    def requires_remote_initialization(self) -> bool:
        return False

    def get_remote_options(self, accelerator_type_str: str = None):
        return {}

    def build_batch_scheduling_plan(
        self, request: BatchSchedulingRequest
    ) -> AcquiredBatchResources:
        map_batches_kwargs: Dict[str, Any] = {
            "num_cpus": 1,
            "num_gpus": 0,
            "resources": {},
            "runtime_env": copy.deepcopy(request.runtime_env),
        }
        return AcquiredBatchResources(
            plan=BatchSchedulingPlan(map_batches_kwargs=map_batches_kwargs),
            close_handle=None,
        )


class GPUAccelerator(AcceleratorBackend):
    # stateless — no __init__
    def default_bundles(
        self, *, num_devices: int, accelerator_type_str: Optional[str] = None
    ):
        bundle = {"GPU": 1}
        if accelerator_type_str:
            bundle[format_ray_accelerator_resource(accelerator_type_str)] = 0.001
        return [bundle.copy() for _ in range(num_devices)]

    def create_placement_group(
        self,
        *,
        bundles: List[Dict[str, float]],
        strategy: str,
        name: str,
        accelerator_type_str: Optional[str] = None,
    ):
        return placement_group(bundles=bundles, strategy=strategy, name=name)

    @property
    def requires_remote_initialization(self) -> bool:
        return True

    def get_remote_options(self, accelerator_type_str: str = None):
        options = {"num_gpus": 0.001}
        if accelerator_type_str:
            options["accelerator_type"] = accelerator_type_str
        return options

    def build_batch_scheduling_plan(
        self, request: BatchSchedulingRequest
    ) -> AcquiredBatchResources:
        ray_remote_args: Dict[str, Any] = {}
        if request.accelerator_type:
            ray_remote_args["accelerator_type"] = request.accelerator_type

        num_bundles_per_replica = request.model_world_size
        pg_config = (
            copy.deepcopy(request.placement_group_config)
            if request.placement_group_config
            else None
        )
        if pg_config is not None:
            bundle_per_worker = pg_config.pop("bundle_per_worker", None)
            if bundle_per_worker is not None:
                pg_config["bundles"] = [
                    bundle_per_worker.copy() for _ in range(num_bundles_per_replica)
                ]

        executor_backend = request.executor_backend or (
            "uni" if num_bundles_per_replica == 1 else "ray"
        )

        map_batches_kwargs: Dict[str, Any] = {
            "runtime_env": copy.deepcopy(request.runtime_env),
        }

        if executor_backend == "ray":
            map_batches_kwargs["ray_remote_args_fn"] = partial(
                _gpu_ray_scheduling_strategy_fn,
                num_bundles_per_replica,
                request.accelerator_type,
                pg_config,
                self,
            )
            ray_remote_args["num_gpus"] = 0
        else:
            if not pg_config:
                ray_remote_args["num_gpus"] = num_bundles_per_replica
            else:
                bundles = pg_config["bundles"]
                resource_counter = Counter()
                for bundle in bundles:
                    resource_counter.update(bundle)
                total_cpus = resource_counter.pop("CPU", 0)
                total_gpus = resource_counter.pop("GPU", 0)
                if total_cpus:
                    ray_remote_args["num_cpus"] = total_cpus
                if total_gpus:
                    ray_remote_args["num_gpus"] = total_gpus
                if resource_counter:
                    ray_remote_args["resources"] = dict(resource_counter)

        map_batches_kwargs.update(ray_remote_args)
        return AcquiredBatchResources(
            plan=BatchSchedulingPlan(map_batches_kwargs=map_batches_kwargs),
            close_handle=None,
        )


def _gpu_ray_scheduling_strategy_fn(
    num_bundles_per_replica: int,
    accelerator_type: Optional[str] = None,
    placement_group_config: Optional[Dict[str, Any]] = None,
    backend: Optional[GPUAccelerator] = None,
) -> Dict[str, Any]:
    """Helper function for legacy GPU dynamic placement group creation."""

    def _get_bundle() -> Dict[str, float]:
        bundle = {"GPU": 1, "CPU": 1}
        if accelerator_type:
            bundle[f"accelerator_type:{accelerator_type}"] = 0.001
        return bundle

    if placement_group_config:
        placement_group_config = copy.deepcopy(placement_group_config)
        bundles = placement_group_config.get("bundles") or []
        if accelerator_type:
            for bundle in bundles:
                bundle[f"accelerator_type:{accelerator_type}"] = 0.001
        if backend is not None:
            pg = backend.create_placement_group(
                bundles=bundles,
                strategy=placement_group_config.get("strategy") or "PACK",
                name="",
            )
        else:
            placement_group_config["bundles"] = bundles
            pg = ray.util.placement_group(**placement_group_config)
    else:
        bundles = [_get_bundle()] * num_bundles_per_replica
        if backend is not None:
            pg = backend.create_placement_group(
                bundles=bundles,
                strategy="PACK",
                name="",
            )
        else:
            pg = ray.util.placement_group(
                bundles,
                strategy="PACK",
            )
    return dict(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            pg, placement_group_capture_child_tasks=True
        )
    )


class TPUAccelerator(AcceleratorBackend):
    """TPU backend for Ray Serve deployments and Ray Data batch inference.

    For batch inference, the backend reserves one ``SlicePlacementGroup`` on the driver
    while the processor is built, waits for it under a bounded timeout, releases the
    head reservation markers, and returns a static scheduling strategy that pins one
    Ray Data engine actor to bundle 0.

    The slice handle is driver-local and never enters the lazy Dataset graph, so the
    processor owns the slice for its explicit lifetime. There is no finalizer: the
    Dataset graph can outlive the ``Processor`` object, and driver exit is the fallback
    boundary. Callers must finish every derived Dataset before closing the processor.
    """

    def __init__(self, config: Optional[TPUConfig] = None):
        self._config = config or TPUConfig()
        self._slice_pg_wrapper = None

    def default_bundles(
        self, *, num_devices: int, accelerator_type_str: Optional[str] = None
    ):
        if not self._config.topology:
            # Fallback to per-chip bundles if no topology is specified
            bundle = {"TPU": 1}
            if accelerator_type_str:
                bundle[format_ray_accelerator_resource(accelerator_type_str)] = 0.001
            return [bundle.copy() for _ in range(num_devices)]

        # Topology is specified, compute per-host bundles
        if not accelerator_type_str:
            raise ValueError(
                "`accelerator_type` must be specified when `topology` is present "
                "in order to compute TPU resource requirements."
            )
        version = get_tpu_version_from_type(accelerator_type_str)
        chips_per_host = get_chips_per_host(self._config.topology, version)

        if num_devices > chips_per_host and num_devices % chips_per_host != 0:
            raise ValueError(
                f"num_devices ({num_devices}) must be a multiple of "
                f"chips_per_host ({chips_per_host}) for TPU topologies."
            )

        num_hosts = max(1, num_devices // chips_per_host)

        tpu_resources = min(num_devices, chips_per_host)
        bundle = {"TPU": tpu_resources}
        bundle[format_ray_accelerator_resource(accelerator_type_str)] = 0.001

        return [bundle.copy() for _ in range(num_hosts)]

    def create_placement_group(
        self,
        *,
        bundles: List[Dict[str, float]],
        strategy: str,
        name: str,
        accelerator_type_str: Optional[str] = None,
    ) -> PlacementGroup:
        if not self._config.topology:
            return placement_group(bundles=bundles, strategy=strategy, name=name)

        if not accelerator_type_str:
            raise ValueError(
                "accelerator_type must be provided for TPU slice provisioning."
            )

        version = get_tpu_version_from_type(accelerator_type_str)

        if bundles:
            # Filter for bundles that actually specify TPU resources
            tpu_bundles = [b for b in bundles if b.get("TPU", 0) > 0]

            if not tpu_bundles:
                worker_bundle = {"TPU": 1}
            else:
                worker_bundle = tpu_bundles[0]

                # Ensure all TPU bundles are homogeneous
                if any(b != worker_bundle for b in tpu_bundles):
                    raise ValueError(
                        "Heterogeneous TPU bundles are not supported when `topology` is set. "
                        "A multi-host TPU slice requires homogeneous resource bundles across all workers. "
                        "Please use `bundle_per_worker` in `placement_group_config` to define uniform worker resources."
                    )
        else:
            # Default to 1 TPU per bundle.
            worker_bundle = {"TPU": 1}

        if self._slice_pg_wrapper is not None:
            logger.debug(
                "Existing TPU slice PG found. Shutting it down before creating a new one."
            )
            self.shutdown()

        self._slice_pg_wrapper = slice_placement_group(
            topology=self._config.topology,
            accelerator_version=version,
            resources_per_bundle=worker_bundle,
            strategy=strategy,
            name=name,
        )
        return self._slice_pg_wrapper.placement_group

    @property
    def requires_deferred_placement_group(self) -> bool:
        return bool(self._config.topology)

    @property
    def requires_remote_initialization(self) -> bool:
        return True

    def get_remote_options(self, accelerator_type_str: str = None):
        options: Dict[str, Any] = {"resources": {}}
        if accelerator_type_str:
            options["label_selector"] = {
                "ray.io/accelerator-type": accelerator_type_str
            }
        return options

    def shutdown(self):
        if self._slice_pg_wrapper is not None:
            try:
                logger.info("Shutting down TPU slice PG for server replica.")
                self._slice_pg_wrapper.shutdown()
            except Exception as e:
                logger.warning(f"Failed to shut down TPU slice PG: {e}")
            finally:
                self._slice_pg_wrapper = None

    def _derive_layout(self, topology: str, accelerator_type: str) -> TPUReplicaLayout:
        """Derive the resolved VM layout from Ray's default chips-per-host rules."""
        accel_version = get_tpu_version_from_type(accelerator_type)
        total_chips = get_num_chips_from_topology(topology)
        chips_per_vm = get_chips_per_host(topology, accel_version)
        if chips_per_vm <= 0:
            raise ValueError(
                f"Resolved chips per VM must be positive, got {chips_per_vm}"
            )
        if total_chips % chips_per_vm != 0:
            raise ValueError(
                f"Topology '{topology}' on {accelerator_type} resolves to "
                f"{total_chips} chips with {chips_per_vm} chips per VM, which does "
                "not divide evenly."
            )
        num_vms = total_chips // chips_per_vm
        return TPUReplicaLayout(
            topology=topology,
            accelerator_type=accelerator_type,
            accelerator_version=accel_version,
            total_chips=total_chips,
            chips_per_vm=chips_per_vm,
            num_vms=num_vms,
        )

    def build_batch_scheduling_plan(
        self, request: BatchSchedulingRequest
    ) -> AcquiredBatchResources:
        """Construct the TPU batch scheduling plan with atomic slice acquisition."""
        tpu_config = (
            request.accelerator_config
            if isinstance(request.accelerator_config, TPUConfig)
            else self._config
        )
        if not isinstance(tpu_config, TPUConfig) or not tpu_config.topology:
            raise ValueError(
                "TPU batch inference requires an explicit `accelerator_config` with "
                "`kind='tpu'` and `topology=...` "
                "(e.g. {'kind': 'tpu', 'topology': '4x4'}). "
                f"Got config: {tpu_config}"
            )

        # The bundle layout below assumes one Ray TPU resource per physical chip.
        raw_driver_rpc = os.environ.get(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
        try:
            driver_rpc = int(raw_driver_rpc)
        except ValueError as exc:
            raise ValueError(
                f"Invalid integer for {RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR} in driver environment: {raw_driver_rpc!r}."
            ) from exc

        if driver_rpc != 1:
            raise ValueError(
                f"TPU batch inference currently requires {RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR} == 1; "
                f"got {driver_rpc}. Multi-PJRT-device-per-chip configurations are not "
                "yet validated."
            )

        if not request.accelerator_type:
            raise ValueError(
                "`accelerator_type` (e.g. 'TPU-V6E') is required for TPU batch inference."
            )

        canonical_accel = normalize_tpu_accelerator_type(request.accelerator_type)
        if canonical_accel not in TPU_ACCELERATOR_VALUES:
            raise ValueError(
                f"Unknown or unsupported TPU accelerator type: {request.accelerator_type!r}. "
                f"Supported TPU types: {sorted(TPU_ACCELERATOR_VALUES)}."
            )

        # engine_kwargs are free-form, so bool/float spellings of 1 (True, 1.0)
        # must be rejected the same way concurrency is — before the == 1 checks.
        tp = _require_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
        pp = _require_positive_int(
            request.pipeline_parallel_size, "pipeline_parallel_size"
        )
        dp = _require_positive_int(request.data_parallel_size, "data_parallel_size")

        layout = self._derive_layout(tpu_config.topology, canonical_accel)

        if tp != layout.total_chips:
            raise ValueError(
                f"tensor_parallel_size must match the total number of physical TPU chips ({layout.total_chips}) "
                f"for topology '{layout.topology}'; got {tp}."
            )
        if pp != 1:
            raise ValueError(
                f"TPU batch inference currently supports pipeline_parallel_size=1; got {pp}."
            )
        if dp != 1:
            raise ValueError(
                f"TPU batch inference currently supports data_parallel_size=1; got {dp}."
            )

        if request.executor_backend != "ray":
            raise ValueError(
                f"TPU batch inference requires executor_backend='ray'; got {request.executor_backend!r}."
            )

        if type(request.concurrency) is not int or request.concurrency != 1:
            raise ValueError(
                f"TPU batch inference requires concurrency=1 (exactly the integer 1); got "
                f"{request.concurrency!r}. Autoscaling and multi-replica TPU execution are "
                "not supported in this release."
            )

        if request.placement_group_config is not None:
            raise ValueError(
                "placement_group_config is not supported for topology-backed TPU "
                "batch inference. The TPU slice bundle layout and placement are "
                "managed by the accelerator backend."
            )

        # Declarative runtime_env merge. Preserve unrelated user variables while
        # forcing the values the TPU engine and its child workers require.
        merged_runtime_env = copy.deepcopy(request.runtime_env or {})
        env_vars = merged_runtime_env.setdefault("env_vars", {})
        if not isinstance(env_vars, dict):
            raise ValueError("runtime_env['env_vars'] must be a dictionary.")

        for name, required in TPU_ENGINE_ENV_VARS.items():
            supplied = env_vars.get(name)
            if supplied is not None and supplied != required:
                raise ValueError(
                    f"runtime_env['env_vars']['{name}'] must be the string "
                    f"{required!r}; got {supplied!r}."
                )
        env_vars.update(TPU_ENGINE_ENV_VARS)

        # Each resolved TPU VM becomes one PG bundle. CPU is reserved for the
        # engine actor; SlicePlacementGroup still advertises per-VM TPU chips on
        # every bundle. Hardware validation must confirm the TPU executor accepts
        # that mixed {CPU, TPU} shape rather than only PGs it built itself.
        resources_per_bundle = {
            "CPU": PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST,
        }

        # Single-VM SlicePGs skip Ray's multi-host slice-name reservation, so the
        # caller must constrain placement to the requested generation/topology.
        # Multi-VM SlicePGs inject ray.io/tpu-slice-name themselves after head
        # reservation; leave that path unchanged.
        bundle_label_selector = None
        if layout.is_single_vm:
            pod_type = infer_tpu_pod_type_from_topology(
                layout.topology, layout.accelerator_type
            )
            if not pod_type:
                raise ValueError(
                    f"Failed to infer TPU pod type for topology '{layout.topology}' "
                    f"and accelerator_type '{layout.accelerator_type}'."
                )
            selector = {
                _raylet.RAY_NODE_TPU_TOPOLOGY_KEY: layout.topology,
                _raylet.RAY_NODE_TPU_POD_TYPE_KEY: pod_type,
            }
            bundle_label_selector = [dict(selector) for _ in range(layout.num_vms)]

        handle = None
        success = False
        timeout_s = _slice_ready_timeout_s()

        try:
            handle = slice_placement_group(
                topology=layout.topology,
                accelerator_version=layout.accelerator_version,
                resources_per_bundle=resources_per_bundle,
                bundle_label_selector=bundle_label_selector,
                strategy="SPREAD",
                tpu_resource_per_chip=1,
            )

            try:
                _wait_for_placement_group(handle.placement_group, timeout_s)
            except ray.exceptions.GetTimeoutError as exc:
                raise TimeoutError(
                    f"Timed out after {timeout_s}s waiting for TPU slice placement "
                    f"group readiness. Requested {layout.accelerator_type} "
                    f"topology={layout.topology} ({layout.num_vms} VMs). This "
                    "usually means the cluster has no intact topology of that shape "
                    f"available. Set {SLICE_READY_TIMEOUT_ENV_VAR} to wait longer "
                    "while capacity is provisioned."
                ) from exc

            # Idempotent for single-host (no head reservation PGs) and multi-host.
            handle.release_head_pgs()

            _validate_reserved_layout(handle, layout)

            # Schedule the Ray Data engine actor into the driver-owned SlicePG.
            # Child-task capture keeps tpu_inference workers in the same PG.
            # vLLM's create_engine_config() (via AsyncLLMEngine.from_engine_args)
            # copies get_current_placement_group() into ParallelConfig.placement_group
            # when running inside this actor; tpu_inference reuses that field and does
            # not call get_current_placement_group() itself.
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=handle.placement_group,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True,
            )

            plan = BatchSchedulingPlan(
                map_batches_kwargs={
                    "num_cpus": PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST,
                    "num_gpus": 0,
                    "resources": {},
                    "scheduling_strategy": scheduling_strategy,
                    "runtime_env": merged_runtime_env,
                },
                required_engine_env_vars=dict(TPU_ENGINE_ENV_VARS),
            )

            success = True
            return AcquiredBatchResources(plan=plan, close_handle=handle)

        finally:
            if handle is not None and not success:
                try:
                    handle.shutdown()
                except Exception:
                    logger.exception(
                        "Failed to clean up TPU slice after batch-plan construction failed."
                    )


def _validate_reserved_layout(handle: Any, layout: TPUReplicaLayout) -> None:
    """Validate the reserved SlicePlacementGroup matches the derived VM layout.

    Structural checks cover both single-host and multi-host topologies. Multi-host
    reservations additionally verify slice-name gang identity via GCS, because Ray
    only attaches ``ray.io/tpu-slice-name`` labels for multi-VM topologies.
    """
    if getattr(handle, "num_hosts", None) != layout.num_vms:
        raise RuntimeError(
            f"Reserved SlicePlacementGroup reports {handle.num_hosts} hosts, but "
            f"topology '{layout.topology}' requires {layout.num_vms} VMs."
        )
    if getattr(handle, "num_bundles", None) != layout.num_vms:
        raise RuntimeError(
            f"Reserved SlicePlacementGroup reports {handle.num_bundles} bundles, but "
            f"topology '{layout.topology}' requires {layout.num_vms}."
        )
    if getattr(handle, "chips_per_host", None) != layout.chips_per_vm:
        raise RuntimeError(
            f"Reserved SlicePlacementGroup reports {handle.chips_per_host} chips per "
            f"host, but layout requires {layout.chips_per_vm}."
        )
    # While RAY_TPU_RESOURCE_PER_CHIP == 1, logical devices equal physical chips.
    if getattr(handle, "devices_per_host", None) != layout.chips_per_vm:
        raise RuntimeError(
            f"Reserved SlicePlacementGroup reports {handle.devices_per_host} devices "
            f"per host, but layout requires {layout.chips_per_vm}."
        )

    pg = handle.placement_group
    bundle_specs = getattr(pg, "bundle_specs", None)
    if bundle_specs is None:
        raise RuntimeError(
            "Reserved placement group is missing bundle_specs; cannot validate the "
            "TPU layout."
        )
    if len(bundle_specs) != layout.num_vms:
        raise RuntimeError(
            f"Reserved placement group has {len(bundle_specs)} bundles, but topology "
            f"'{layout.topology}' requires {layout.num_vms}."
        )
    for idx, bundle in enumerate(bundle_specs):
        tpu_amount = bundle.get("TPU", 0)
        if tpu_amount != layout.chips_per_vm:
            raise RuntimeError(
                f"Reserved placement group bundle {idx} advertises TPU={tpu_amount}, "
                f"but layout requires TPU={layout.chips_per_vm}."
            )

    if not layout.is_single_vm:
        _validate_multihost_slice_identity(handle, layout)


def _validate_multihost_slice_identity(
    handle: Any, layout: TPUReplicaLayout
) -> List[Dict[str, Any]]:
    """Validate multi-host slice-name identity and per-node TPU capacity via GCS."""
    label_selectors = getattr(handle, "bundle_label_selector", [])
    slice_names = {
        selector[_raylet.RAY_NODE_TPU_SLICE_NAME_KEY]
        for selector in label_selectors
        if isinstance(selector, dict)
        and selector.get(_raylet.RAY_NODE_TPU_SLICE_NAME_KEY)
    }

    if len(slice_names) != 1:
        raise RuntimeError(
            f"Expected exactly 1 distinct TPU slice name across bundles; found "
            f"{len(slice_names)}: {slice_names}."
        )

    slice_name = next(iter(slice_names))
    selected_nodes = get_tpu_nodes_for_slice(slice_name)

    if len(selected_nodes) != layout.num_vms:
        raise RuntimeError(
            f"Slice '{slice_name}' selected by placement group contains {len(selected_nodes)} "
            f"alive nodes, but topology '{layout.topology}' requires {layout.num_vms} VMs."
        )

    for node in selected_nodes:
        resources = node.get("Resources", {})
        tpu_count = resources.get("TPU", 0)
        if tpu_count != layout.chips_per_vm:
            node_id = node.get("NodeID")
            raise RuntimeError(
                f"Node '{node_id}' in selected slice '{slice_name}' advertises {tpu_count} TPU "
                f"resources, but layout requires {layout.chips_per_vm} chips per VM."
            )

    return selected_nodes


def get_accelerator_backend(
    accelerator_config: AcceleratorConfig,
) -> AcceleratorBackend:
    """Instantiate an AcceleratorBackend for a fully resolved accelerator config.

    Callers (e.g. vLLM Batch processor config validation) must finish defaulting and
    compatibility checks before invoking this helper. This function only dispatches
    on the typed config.
    """
    if isinstance(accelerator_config, TPUConfig):
        return TPUAccelerator(accelerator_config)
    if isinstance(accelerator_config, GPUConfig):
        return GPUAccelerator()
    if isinstance(accelerator_config, CPUConfig):
        return CPUAccelerator()
    raise TypeError(f"Unsupported accelerator config: {accelerator_config!r}")
