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
    TPU_8_CHIPS_PER_HOST_TYPES,
    TPU_SINGLE_HOST_TOPOLOGIES,
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
    get_tpu_worker_resources,
    slice_placement_group,
)

logger = logging.getLogger(__name__)

# Constants for TPU batch scheduling
PARENT_ACTOR_CPU_RESERVE = 1
DEFAULT_USER_CPU_PER_HOST = 1
CPU_ACCELERATOR_TYPE_LITERAL = "CPU"

# Explicit chips_per_vm overrides are only for known ambiguous provision modes.
# Keyed by (accelerator_version, topology) → allowed chips_per_vm values.
# v6e/v5e 2x4 can be one 8-chip VM (Ray default) or two 4-chip VMs.
_AMBIGUOUS_CHIPS_PER_VM_OVERRIDES = {
    (version, "2x4"): frozenset({4, 8}) for version in TPU_8_CHIPS_PER_HOST_TYPES
}

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
    """Resolved TPU topology for one model replica.

    Separates physical SlicePG topology from vLLM/framework executor devices.
    Placement-group bundle count is derived separately from the optional worker
    template via ``get_tpu_worker_resources`` (Ray scheduling resources), not
    from ``total_framework_devices``.

    Attributes:
        topology: Canonical topology string, such as ``"4x4"`` or ``"2x2x1"``.
        accelerator_type: The canonical Ray accelerator type, such as ``"TPU-V6E"``.
        accelerator_version: The generation alone, such as ``"v6e"`` or ``"v7x"``.
        total_chips: Total physical chips across the topology.
        chips_per_vm: Physical chips on each resolved TPU VM (Ray's default
            ``chips_per_host`` / ``chips_per_vm``, or an explicit
            ``accelerator_config.chips_per_vm`` override for ambiguous shapes).
        num_vms: Number of physical TPU VMs in the topology. This is independent of
            the SlicePG bundle count when using finer TPU-per-bundle granularity.
        framework_devices_per_chip: Framework-visible devices per physical chip (1 for
            all current generations except v7x; 2 for Ironwood/v7x chiplets).
            Independent of ``RAY_TPU_RESOURCE_PER_CHIP``, which remains a Ray
            scheduling concern.
    """

    topology: str
    accelerator_type: str
    accelerator_version: str

    # Physical slice
    total_chips: int
    chips_per_vm: int
    num_vms: int

    # Framework execution
    framework_devices_per_chip: int

    @property
    def total_framework_devices(self) -> int:
        return self.total_chips * self.framework_devices_per_chip

    @property
    def is_single_vm(self) -> bool:
        return self.num_vms == 1


@dataclass(frozen=True)
class BatchSchedulingRequest:
    """Input request for batch scheduling strategy construction.

    Parallel-size fields default to vLLM's single-replica values (1). Callers that
    omit them intentionally get that default; TPU admission still validates TP
    against the topology's framework device count and rejects coerced bool/float
    spellings.
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
class _BatchOwnedTPUResources:
    """Driver-local Batch handle that releases a backend-owned SlicePG.

    ``TPUAccelerator.shutdown()`` swallows errors for Serve replica teardown.
    Batch construction cleanup needs failures to propagate so the processor
    builder can log them, so Batch returns this thin handle instead of the
    backend instance itself.

    The wrapper is retained until ``shutdown()`` succeeds so
    ``_ManagedVLLMProcessor.close()`` can retry after a transient failure.
    """

    backend: "TPUAccelerator"
    wrapper: Any

    def shutdown(self) -> None:
        if self.wrapper is None:
            return

        wrapper = self.wrapper
        wrapper.shutdown()

        # Only clear ownership after a successful shutdown so a retained
        # close handle can retry the same SlicePG.
        if self.backend._slice_pg_wrapper is wrapper:
            self.backend._slice_pg_wrapper = None
        self.wrapper = None


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
            "use Ray's default chips-per-VM resolution unless chips_per_vm is set."
        ),
    )
    chips_per_vm: Optional[int] = Field(
        default=None,
        description=(
            "Optional override for physical chips per TPU VM on ambiguous "
            "provisionings such as v6e '2x4' (default 8 chips / 1 VM, or 4 chips / "
            "2 VMs). This requests a physical realization that must match how the "
            "cluster is provisioned; SlicePG head reservation does not yet make "
            "selection deterministic in a mixed cluster. When omitted, Ray's "
            "default chips-per-host rules apply."
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

    def default_batch_executor_backend(
        self,
        *,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ) -> str:
        """Default vLLM distributed executor backend for Batch.

        GPU legacy behavior: ``uni`` when TP×PP == 1, otherwise ``ray``.
        Accelerator-specific subclasses may override (e.g. TPU always ``ray``).
        """
        return "uni" if tensor_parallel_size * pipeline_parallel_size == 1 else "ray"

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
        topology = self._config.topology.strip().lower()
        version = get_tpu_version_from_type(accelerator_type_str)
        total_chips = get_num_chips_from_topology(topology)
        chips_per_host = self._resolve_chips_per_vm(
            self._config.chips_per_vm,
            topology=topology,
            accelerator_version=version,
            total_chips=total_chips,
            default_chips_per_vm=get_chips_per_host(topology, version),
        )
        # Serve passes TP×PP as num_devices (framework devices). Convert to
        # physical chips before packing hosts so v7x (2 devices/chip) and
        # chips_per_vm overrides share one physical host model.
        framework_devices_per_chip = self._resolve_framework_devices_per_chip(version)
        if num_devices % framework_devices_per_chip != 0:
            raise ValueError(
                f"num_devices ({num_devices}) must be a multiple of "
                f"framework_devices_per_chip ({framework_devices_per_chip}) for {version}."
            )
        num_chips = num_devices // framework_devices_per_chip

        if num_chips > chips_per_host and num_chips % chips_per_host != 0:
            raise ValueError(
                f"Physical chip count ({num_chips}) must be a multiple of "
                f"chips_per_host ({chips_per_host}) for TPU topologies."
            )

        num_hosts = max(1, num_chips // chips_per_host)

        tpu_resources = min(num_chips, chips_per_host)
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

        worker_bundle = self._resolve_topology_worker_bundle(bundles)
        topology = self._config.topology.strip().lower()
        version = get_tpu_version_from_type(accelerator_type_str)
        chips_per_vm = self._resolve_chips_per_vm(
            self._config.chips_per_vm,
            topology=topology,
            accelerator_version=version,
            total_chips=get_num_chips_from_topology(topology),
            default_chips_per_vm=get_chips_per_host(topology, version),
        )
        self._create_slice_pg_handle(
            accelerator_type=accelerator_type_str,
            resources_per_bundle=worker_bundle,
            strategy=strategy,
            name=name,
            chips_per_vm=chips_per_vm,
        )
        return self._slice_pg_wrapper.placement_group

    def _resolve_topology_worker_bundle(
        self, bundles: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Resolve a homogeneous TPU worker-bundle template from Serve/Data PG bundles.

        Preserves existing Serve semantics for positive-TPU bundles while ensuring
        no-TPU templates preserve CPU/custom resources with a ``{"TPU": 1}`` fallback.
        """
        if not bundles:
            return {"TPU": 1}

        tpu_bundles = [b for b in bundles if b.get("TPU", 0) > 0]
        if tpu_bundles:
            worker_bundle = tpu_bundles[0]
            if any(b != worker_bundle for b in tpu_bundles):
                raise ValueError(
                    "Heterogeneous TPU bundles are not supported when `topology` is set. "
                    "A multi-host TPU slice requires homogeneous resource bundles across all workers. "
                    "Please use `bundle_per_worker` in `placement_group_config` to define uniform worker resources."
                )
            return dict(worker_bundle)

        # No positive-TPU bundles: preserve CPU/custom resources and add TPU: 1.
        cleaned_bundles = [
            {k: v for k, v in b.items() if v != 0 and v != 0.0} for b in bundles
        ]
        template = cleaned_bundles[0]
        if any(b != template for b in cleaned_bundles):
            raise ValueError(
                "Heterogeneous placement_group_config bundles are not supported "
                "when `topology` is set; got "
                f"{bundles!r}."
            )
        return {**template, "TPU": 1}

    def _create_slice_pg_handle(
        self,
        *,
        accelerator_type: str,
        resources_per_bundle: Dict[str, float],
        strategy: str,
        name: str = "",
        bundle_label_selector: Optional[List[Dict[str, str]]] = None,
        tpu_resource_per_chip: Optional[int] = None,
        chips_per_vm: Optional[int] = None,
    ):
        """Create and own a topology-backed SlicePlacementGroup.

        Both Serve (deferred replica PG) and Data (eager driver PG) call this
        private primitive so ``slice_placement_group`` stays encapsulated here.
        """
        if not self._config.topology:
            raise ValueError(
                "TPU slice placement requires accelerator_config.topology to be set."
            )
        if self._slice_pg_wrapper is not None:
            logger.debug(
                "Existing TPU slice PG found. Shutting it down before creating a new one."
            )
            self.shutdown()

        # Canonicalize like SlicePlacementGroup / Batch layout derivation so
        # topology math and any caller labels always agree.
        topology = self._config.topology.strip().lower()
        version = get_tpu_version_from_type(accelerator_type)
        slice_kwargs: Dict[str, Any] = {
            "topology": topology,
            "accelerator_version": version,
            "resources_per_bundle": resources_per_bundle,
            "strategy": strategy,
            "bundle_label_selector": bundle_label_selector,
            "tpu_resource_per_chip": tpu_resource_per_chip,
        }
        if chips_per_vm is not None:
            slice_kwargs["chips_per_vm"] = chips_per_vm
        if name:
            slice_kwargs["name"] = name
        self._slice_pg_wrapper = slice_placement_group(**slice_kwargs)
        return self._slice_pg_wrapper

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
                logger.info("Shutting down TPU slice placement group.")
                self._slice_pg_wrapper.shutdown()
            except Exception as e:
                logger.warning(f"Failed to shut down TPU slice PG: {e}")
            finally:
                self._slice_pg_wrapper = None

    def default_batch_executor_backend(
        self,
        *,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ) -> str:
        # Topology-backed TPU always uses the Ray executor so one SlicePG owns
        # both the parent actor and every TPU worker, including single-VM shapes.
        return "ray"

    def _derive_layout(
        self,
        topology: str,
        accelerator_type: str,
        chips_per_vm: Optional[int] = None,
    ) -> TPUReplicaLayout:
        """Derive physical VM layout and framework device counts for one replica."""
        # Match SlicePlacementGroup / node-label discovery: topology strings are
        # compared as exact node labels, so Batch must store the canonical form.
        if not isinstance(topology, str):
            raise ValueError(
                f"TPU topology must be a non-empty string; got {topology!r}."
            )
        canonical_topology = topology.strip().lower()
        if not canonical_topology:
            raise ValueError("TPU topology must be non-empty.")

        accel_version = get_tpu_version_from_type(accelerator_type)
        total_chips = get_num_chips_from_topology(canonical_topology)
        default_chips_per_vm = get_chips_per_host(canonical_topology, accel_version)
        resolved_chips_per_vm = self._resolve_chips_per_vm(
            chips_per_vm,
            topology=canonical_topology,
            accelerator_version=accel_version,
            total_chips=total_chips,
            default_chips_per_vm=default_chips_per_vm,
        )
        num_vms = total_chips // resolved_chips_per_vm
        framework_devices_per_chip = self._resolve_framework_devices_per_chip(
            accel_version
        )
        return TPUReplicaLayout(
            topology=canonical_topology,
            accelerator_type=accelerator_type,
            accelerator_version=accel_version,
            total_chips=total_chips,
            chips_per_vm=resolved_chips_per_vm,
            num_vms=num_vms,
            framework_devices_per_chip=framework_devices_per_chip,
        )

    @staticmethod
    def _resolve_framework_devices_per_chip(accelerator_version: str) -> int:
        """Framework-visible devices per physical chip for vLLM TP sizing.

        This is the vLLM/framework execution-device count, not the TPU core
        count Ray uses for pod-type naming (e.g. v4-8). Among currently
        accepted generations, only v7x exposes two framework devices per
        physical chip; every other generation maps one chip to one device.
        Independent of ``RAY_TPU_RESOURCE_PER_CHIP``.
        """
        if accelerator_version.strip().lower() == "v7x":
            return 2
        return 1

    @staticmethod
    def _resolve_chips_per_vm(
        chips_per_vm: Optional[int],
        *,
        topology: str,
        accelerator_version: str,
        total_chips: int,
        default_chips_per_vm: int,
    ) -> int:
        """Resolve chips-per-VM, allowing overrides only for known ambiguous shapes."""
        if chips_per_vm is None:
            chips_per_vm = default_chips_per_vm
        else:
            if isinstance(chips_per_vm, bool) or not isinstance(chips_per_vm, int):
                raise ValueError(
                    f"chips_per_vm must be a positive integer; got {chips_per_vm!r}."
                )
            if chips_per_vm <= 0:
                raise ValueError(
                    f"chips_per_vm must be a positive integer; got {chips_per_vm}."
                )
            if chips_per_vm != default_chips_per_vm:
                allowed = _AMBIGUOUS_CHIPS_PER_VM_OVERRIDES.get(
                    (accelerator_version, topology)
                )
                if allowed is None or chips_per_vm not in allowed:
                    raise ValueError(
                        f"chips_per_vm={chips_per_vm} is not a supported override for "
                        f"topology '{topology}' on {accelerator_version}. "
                        f"Ray's default is {default_chips_per_vm} chips per VM. "
                        "Explicit overrides are only supported for ambiguous "
                        "single-host shapes such as v6e '2x4' "
                        f"(allowed values: {sorted(allowed) if allowed else 'n/a'})."
                    )

        if chips_per_vm <= 0:
            raise ValueError(
                f"Resolved chips per VM must be positive, got {chips_per_vm}"
            )
        if total_chips % chips_per_vm != 0:
            raise ValueError(
                f"Topology '{topology}' on {accelerator_version} resolves to "
                f"{total_chips} chips with {chips_per_vm} chips per VM, which does "
                "not divide evenly."
            )
        # Defensive: allowlisted ambiguous overrides should already be single-host
        # topologies with multiple valid host packings.
        if (
            chips_per_vm != default_chips_per_vm
            and topology not in TPU_SINGLE_HOST_TOPOLOGIES
        ):
            raise ValueError(
                f"chips_per_vm overrides are only supported for single-host topologies "
                f"{TPU_SINGLE_HOST_TOPOLOGIES}; got topology '{topology}'."
            )
        return chips_per_vm

    @staticmethod
    def _apply_batch_cpu_floor(bundle: Dict[str, float]) -> Dict[str, float]:
        """Preserve user resources while ensuring Batch parent CPU admission."""
        out: Dict[str, float] = {}
        for key, value in bundle.items():
            if value == 0 or value == 0.0:
                continue
            out[key] = value
        if out.get("GPU", 0) > 0:
            raise ValueError(
                "GPU resources are not supported in topology-backed TPU Batch "
                f"placement_group_config bundles; got GPU={out['GPU']!r}."
            )
        floor = float(PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST)
        out["CPU"] = max(float(out.get("CPU", 0.0)), floor)
        return out

    def _resolve_batch_worker_bundle(
        self,
        placement_group_config: Optional[Dict[str, Any]],
        layout: TPUReplicaLayout,
    ) -> Dict[str, float]:
        """Resolve the homogeneous TPU worker-resource template for Batch.

        Default (no placement_group_config) intentionally omits TPU so Ray fills
        chips-per-VM. Explicit placement_group_config supplies a single template
        (via bundle_per_worker or bundles) that sets worker granularity (e.g. TPU:1)
        with Batch parent CPU floor applied.
        """
        if placement_group_config is None:
            return {
                "CPU": float(PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST),
            }

        bundle_per_worker = placement_group_config.get("bundle_per_worker")
        if bundle_per_worker is not None:
            source_bundles = [dict(bundle_per_worker)]
        elif (
            "bundles" in placement_group_config
            and placement_group_config["bundles"] is not None
        ):
            source_bundles = [
                dict(bundle) for bundle in placement_group_config["bundles"]
            ]
        else:
            raise ValueError(
                "placement_group_config must specify bundle_per_worker or bundles."
            )

        self._validate_batch_tpu_template_bundles(source_bundles, layout)

        has_positive_tpu = [bundle.get("TPU", 0) > 0 for bundle in source_bundles]
        if any(has_positive_tpu) and not all(has_positive_tpu):
            raise ValueError(
                "Topology-backed TPU Batch placement_group_config bundles "
                "cannot mix TPU-bearing and non-TPU bundles."
            )

        worker_bundle = self._resolve_topology_worker_bundle(source_bundles)
        return self._apply_batch_cpu_floor(worker_bundle)

    @staticmethod
    def _validate_tpu_per_bundle(tpu_per_bundle: Any, layout: TPUReplicaLayout) -> int:
        """Reject TPU granularities that cannot land evenly on each physical VM."""
        if isinstance(tpu_per_bundle, bool) or not isinstance(
            tpu_per_bundle, (int, float)
        ):
            raise ValueError(
                f"TPU resources per bundle must be a positive number; got {tpu_per_bundle!r}."
            )
        if float(tpu_per_bundle) != int(tpu_per_bundle):
            raise ValueError(
                f"TPU resources per bundle must be an integer; got {tpu_per_bundle!r}."
            )
        tpu_i = int(tpu_per_bundle)
        if tpu_i <= 0:
            raise ValueError(f"TPU resources per bundle must be positive; got {tpu_i}.")
        if tpu_i > layout.chips_per_vm:
            raise ValueError(
                f"TPU resources per bundle ({tpu_i}) exceed physical chips per VM "
                f"({layout.chips_per_vm}) for topology '{layout.topology}'."
            )
        if layout.chips_per_vm % tpu_i != 0:
            raise ValueError(
                f"TPU resources per bundle ({tpu_i}) must evenly divide physical "
                f"chips per VM ({layout.chips_per_vm}) for topology '{layout.topology}'."
            )
        return tpu_i

    def _validate_batch_tpu_template_bundles(
        self,
        bundles: List[Dict[str, float]],
        layout: TPUReplicaLayout,
    ) -> None:
        """Reject invalid Batch templates before shared fallback can mask them."""
        for bundle in bundles:
            gpu = bundle.get("GPU", 0)
            if gpu > 0:
                raise ValueError(
                    "GPU resources are not supported in topology-backed TPU Batch "
                    f"placement_group_config bundles; got GPU={gpu!r}."
                )
            if "TPU" in bundle:
                self._validate_tpu_per_bundle(bundle["TPU"], layout)

    @staticmethod
    def _resolve_batch_slice_strategy(
        *,
        requested_strategy: str,
        layout: TPUReplicaLayout,
        num_bundles: int,
    ) -> str:
        """Enforce topology-safe SlicePG strategies for Batch execution layouts."""
        strategy = requested_strategy
        if layout.is_single_vm and num_bundles > 1:
            if strategy not in ("PACK", "STRICT_PACK"):
                raise ValueError(
                    "Single-VM TPU topologies with multiple worker bundles require "
                    "PACK/STRICT_PACK so every bundle remains on the same TPU VM; "
                    f"got strategy={strategy!r}."
                )
            # PACK prefers one node but is not a hard invariant; STRICT_PACK is.
            return "STRICT_PACK"

        if not layout.is_single_vm:
            if strategy == "STRICT_PACK":
                raise ValueError(
                    "STRICT_PACK cannot represent a multi-VM TPU topology "
                    f"({layout.num_vms} VMs for topology '{layout.topology}')."
                )
            if strategy == "STRICT_SPREAD" and num_bundles > layout.num_vms:
                raise ValueError(
                    "STRICT_SPREAD requires one node per bundle, but this topology "
                    f"has {layout.num_vms} physical TPU VMs and {num_bundles} bundles."
                )
        return strategy

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

        # Keep the backend config topology in sync with the request (Serve may
        # construct TPUAccelerator once; Batch always passes the request config).
        self._config = tpu_config

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

        layout = self._derive_layout(
            tpu_config.topology,
            canonical_accel,
            chips_per_vm=tpu_config.chips_per_vm,
        )

        if tp != layout.total_framework_devices:
            raise ValueError(
                f"tensor_parallel_size must match the total number of framework "
                f"devices ({layout.total_framework_devices} = {layout.total_chips} physical "
                f"chips × {layout.framework_devices_per_chip} device(s)/chip) for topology "
                f"'{layout.topology}' on {layout.accelerator_version}; got {tp}."
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

        # Resolve execution-layout worker template and placement strategy.
        resources_per_bundle = self._resolve_batch_worker_bundle(
            request.placement_group_config,
            layout,
        )

        requested_strategy = (
            request.placement_group_config.get("strategy")
            if request.placement_group_config
            else None
        ) or "PACK"

        expected_num_bundles, expected_bundle_resources = get_tpu_worker_resources(
            topology=layout.topology,
            accelerator_type=layout.accelerator_type,
            resources_per_worker=resources_per_bundle,
            num_slices=1,
            chips_per_vm=layout.chips_per_vm,
            tpu_resource_per_chip=1,
        )
        strategy = self._resolve_batch_slice_strategy(
            requested_strategy=requested_strategy,
            layout=layout,
            num_bundles=expected_num_bundles,
        )

        # Single-VM SlicePGs skip Ray's multi-host slice-name reservation, so the
        # caller must constrain placement to the requested generation/topology.
        # Selector count must match the *execution* bundle count, not num_vms.
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
            bundle_label_selector = [
                dict(selector) for _ in range(expected_num_bundles)
            ]

        handle = None
        success = False
        timeout_s = _slice_ready_timeout_s()

        try:
            handle = self._create_slice_pg_handle(
                accelerator_type=layout.accelerator_type,
                resources_per_bundle=resources_per_bundle,
                strategy=strategy,
                bundle_label_selector=bundle_label_selector,
                tpu_resource_per_chip=1,
                chips_per_vm=layout.chips_per_vm,
            )

            try:
                _wait_for_placement_group(handle.placement_group, timeout_s)
            except ray.exceptions.GetTimeoutError as exc:
                raise TimeoutError(
                    f"Timed out after {timeout_s}s waiting for TPU slice placement "
                    f"group readiness. Requested {layout.accelerator_type} "
                    f"topology={layout.topology} ({layout.num_vms} VMs, "
                    f"{expected_num_bundles} bundles). This usually means the "
                    "cluster has no intact topology of that shape available. Set "
                    f"{SLICE_READY_TIMEOUT_ENV_VAR} to wait longer while capacity "
                    "is provisioned."
                ) from exc

            # Idempotent for single-host (no head reservation PGs) and multi-host.
            handle.release_head_pgs()

            _validate_reserved_layout(
                handle,
                layout,
                expected_num_bundles=expected_num_bundles,
                expected_bundle_resources=expected_bundle_resources,
            )

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
            return AcquiredBatchResources(
                plan=plan,
                close_handle=_BatchOwnedTPUResources(backend=self, wrapper=handle),
            )

        finally:
            if handle is not None and not success:
                # Propagate Batch construction-cleanup failures (unlike Serve's
                # swallow-on-shutdown path on the backend itself).
                try:
                    handle.shutdown()
                except Exception:
                    logger.exception(
                        "Failed to clean up TPU slice after batch-plan construction failed."
                    )
                finally:
                    self._slice_pg_wrapper = None


def _validate_reserved_layout(
    handle: Any,
    layout: TPUReplicaLayout,
    *,
    expected_num_bundles: int,
    expected_bundle_resources: Dict[str, float],
) -> None:
    """Validate reserved SlicePG against physical topology and execution layout.

    Physical checks cover VM count and per-VM chip capacity. Execution checks
    cover PG bundle count and per-bundle TPU/CPU amounts. Multi-host reservations
    additionally verify slice-name gang identity via GCS.
    """
    if getattr(handle, "num_hosts", None) != layout.num_vms:
        raise RuntimeError(
            f"Reserved SlicePlacementGroup reports {handle.num_hosts} hosts, but "
            f"topology '{layout.topology}' requires {layout.num_vms} VMs."
        )
    if getattr(handle, "num_bundles", None) != expected_num_bundles:
        raise RuntimeError(
            f"Reserved SlicePlacementGroup reports {handle.num_bundles} bundles, but "
            f"execution layout requires {expected_num_bundles}."
        )
    if getattr(handle, "chips_per_host", None) != layout.chips_per_vm:
        raise RuntimeError(
            f"Reserved SlicePlacementGroup reports {handle.chips_per_host} chips per "
            f"host, but layout requires {layout.chips_per_vm}."
        )
    # SlicePlacementGroup.devices_per_host represents Ray logical TPU
    # resources per host. With RAY_TPU_RESOURCE_PER_CHIP=1 this equals
    # the physical chip count; it is independent of framework devices
    # (TPUReplicaLayout.framework_devices_per_chip / total_framework_devices).
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
    if len(bundle_specs) != expected_num_bundles:
        raise RuntimeError(
            f"Reserved placement group has {len(bundle_specs)} bundles, but "
            f"execution layout requires {expected_num_bundles}."
        )

    expected_tpu = float(expected_bundle_resources.get("TPU", 0))
    min_cpu = float(PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST)
    for idx, bundle in enumerate(bundle_specs):
        tpu_amount = float(bundle.get("TPU", 0))
        if tpu_amount != expected_tpu:
            raise RuntimeError(
                f"Reserved placement group bundle {idx} advertises TPU={tpu_amount}, "
                f"but execution layout requires TPU={expected_tpu}."
            )
        cpu_amount = float(bundle.get("CPU", 0))
        if cpu_amount < min_cpu:
            raise RuntimeError(
                f"Reserved placement group bundle {idx} advertises CPU={cpu_amount}, "
                f"but Batch requires at least CPU={min_cpu} for the engine parent."
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
