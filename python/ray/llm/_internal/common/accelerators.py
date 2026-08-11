"""Shared accelerator configurations and backend abstractions for LLM serving and batch inference."""

import copy
import logging
import os
from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

import ray
import ray.util.accelerators.accelerators as accelerators
from ray._private.accelerators.tpu import get_num_chips_from_topology
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.tpu import (
    RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR,
    get_tpu_version_from_type,
    resolve_chips_per_vm,
    slice_placement_group,
)

logger = logging.getLogger(__name__)

# Constants for TPU batch scheduling
PARENT_ACTOR_CPU_RESERVE = 1
DEFAULT_USER_CPU_PER_HOST = 1
CPU_ACCELERATOR_TYPE_LITERAL = "CPU"

# Bound driver-side wait for an eagerly acquired TPU SlicePG.
DEFAULT_PG_READY_TIMEOUT_S = 180.0

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


def _wait_for_placement_group(pg: PlacementGroup, timeout_s: float) -> None:
    """Block until the placement group is scheduled, or raise ``GetTimeoutError``."""
    ray.get(pg.ready(), timeout=timeout_s)


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


def _vllm_tp_multiplier(accelerator_version: str) -> int:
    """vLLM TP size multiplier per physical TPU chip.

    This is the vLLM/framework execution-device count, not the TPU core count
    Ray uses for pod-type naming. Among currently accepted generations, only
    v7x exposes two framework devices per physical chip.
    """
    return 2 if accelerator_version.strip().lower() == "v7x" else 1


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
            "Physical TPU topology string (e.g. '4x4'). Required for "
            "topology-backed TPU batch inference."
        ),
    )
    chips_per_vm: Optional[int] = Field(
        default=None,
        description=(
            "Optional physical chips-per-VM override for TPU topologies with "
            "multiple supported VM packings. Must match cluster provisioning."
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

    def build_batch_scheduling_options(
        self,
        *,
        accelerator_type: Optional[str],
        engine_kwargs: Dict[str, Any],
        placement_group_config: Optional[Dict[str, Any]],
        runtime_env: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Optional[Callable[[], None]]]:
        """Return ``(map_batches_kwargs, optional driver close_fn)``.

        ``map_batches_kwargs`` is a plain, picklable dict safe to embed in the
        lazy Ray Data dataset DAG. ``close_fn`` (when not ``None``) is a
        driver-local callable that releases any resources acquired here; it must
        never enter the dataset graph. Backends that do not support Batch raise
        ``NotImplementedError`` so Serve-only accelerators need not implement it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement Batch scheduling options."
        )


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

    def build_batch_scheduling_options(
        self,
        *,
        accelerator_type: Optional[str],
        engine_kwargs: Dict[str, Any],
        placement_group_config: Optional[Dict[str, Any]],
        runtime_env: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Optional[Callable[[], None]]]:
        ray_remote_args: Dict[str, Any] = {}
        if accelerator_type:
            ray_remote_args["accelerator_type"] = accelerator_type

        tp = engine_kwargs.get("tensor_parallel_size", 1)
        pp = engine_kwargs.get("pipeline_parallel_size", 1)
        num_bundles_per_replica = tp * pp
        pg_config = (
            copy.deepcopy(placement_group_config) if placement_group_config else None
        )
        if pg_config is not None:
            bundle_per_worker = pg_config.pop("bundle_per_worker", None)
            if bundle_per_worker is not None:
                pg_config["bundles"] = [
                    bundle_per_worker.copy() for _ in range(num_bundles_per_replica)
                ]

        engine_kwargs.setdefault(
            "distributed_executor_backend",
            "uni" if num_bundles_per_replica == 1 else "ray",
        )
        executor_backend = engine_kwargs["distributed_executor_backend"]

        map_batches_kwargs: Dict[str, Any] = {
            "runtime_env": copy.deepcopy(runtime_env),
        }

        if executor_backend == "ray":
            map_batches_kwargs["ray_remote_args_fn"] = partial(
                _gpu_ray_scheduling_strategy_fn,
                num_bundles_per_replica,
                accelerator_type,
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
        return map_batches_kwargs, None


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
        chips_per_host = resolve_chips_per_vm(
            topology, version, self._config.chips_per_vm
        )
        # Serve passes TP×PP as num_devices. Convert to physical chips before
        # packing hosts so v7x (TP multiplier 2) and chips_per_vm overrides
        # share one physical host model.
        tp_multiplier = _vllm_tp_multiplier(version)
        if num_devices % tp_multiplier != 0:
            raise ValueError(
                f"num_devices ({num_devices}) must be a multiple of "
                f"the vLLM TP multiplier ({tp_multiplier}) for {version}."
            )
        num_chips = num_devices // tp_multiplier

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
        # Pass the configured chips_per_vm through to Ray (may be None so Ray
        # owns the default). Reject non-positive / non-int overrides early.
        chips_per_vm = self._config.chips_per_vm
        if chips_per_vm is not None:
            _require_positive_int(chips_per_vm, "chips_per_vm")
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
    ) -> Dict[str, float]:
        """Resolve the homogeneous TPU worker-resource template for Batch.

        Default (no placement_group_config) intentionally omits TPU so Ray fills
        chips-per-VM. Explicit placement_group_config supplies a single template
        (via bundle_per_worker or bundles) that sets worker granularity (e.g. TPU:1)
        with Batch parent CPU floor applied. Positive TPU-per-VM fit is enforced by
        Ray in ``get_tpu_worker_resources``.
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

        self._validate_batch_tpu_template_bundles(source_bundles)

        has_positive_tpu = [bundle.get("TPU", 0) > 0 for bundle in source_bundles]
        if any(has_positive_tpu) and not all(has_positive_tpu):
            raise ValueError(
                "Topology-backed TPU Batch placement_group_config bundles "
                "cannot mix TPU-bearing and non-TPU bundles."
            )

        worker_bundle = self._resolve_topology_worker_bundle(source_bundles)
        return self._apply_batch_cpu_floor(worker_bundle)

    @staticmethod
    def _validate_batch_tpu_template_bundles(
        bundles: List[Dict[str, float]],
    ) -> None:
        """Reject GPU / invalid explicit TPU values before shared fallback can mask them.

        Positive TPU-per-VM fit (fits on a VM, divides evenly) is validated by Ray
        in ``get_tpu_worker_resources``. Explicit ``TPU`` keys that are non-positive
        or non-integer are rejected here so the omit-TPU ``TPU:1`` fallback cannot
        hide them.
        """
        for bundle in bundles:
            gpu = bundle.get("GPU", 0)
            if gpu > 0:
                raise ValueError(
                    "GPU resources are not supported in topology-backed TPU Batch "
                    f"placement_group_config bundles; got GPU={gpu!r}."
                )
            if "TPU" not in bundle:
                continue
            tpu = bundle["TPU"]
            if isinstance(tpu, bool) or not isinstance(tpu, (int, float)):
                raise ValueError(
                    f"TPU resources per bundle must be a positive number; got {tpu!r}."
                )
            if float(tpu) != int(tpu):
                raise ValueError(
                    f"TPU resources per bundle must be an integer; got {tpu!r}."
                )
            if int(tpu) <= 0:
                raise ValueError(
                    f"TPU resources per bundle must be positive; got {int(tpu)}."
                )

    def build_batch_scheduling_options(
        self,
        *,
        accelerator_type: Optional[str],
        engine_kwargs: Dict[str, Any],
        placement_group_config: Optional[Dict[str, Any]],
        runtime_env: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Optional[Callable[[], None]]]:
        """Reserve a TPU slice and return ``(map_batches_kwargs, close_fn)``.

        The driver reserves one ``SlicePlacementGroup`` while the processor is
        built, waits under a bounded timeout, releases head reservation markers,
        and trusts SlicePG for physical packing, labels, and strategy. ``close_fn``
        is the SlicePG shutdown callable.
        """
        tpu_config = self._config
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
                f"Invalid integer for {RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR} in "
                f"driver environment: {raw_driver_rpc!r}."
            ) from exc

        if driver_rpc != 1:
            raise ValueError(
                f"TPU batch inference currently requires "
                f"{RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR} == 1; got {driver_rpc}. "
                "Multi-PJRT-device-per-chip configurations are not yet validated."
            )

        if not accelerator_type:
            raise ValueError(
                "`accelerator_type` (e.g. 'TPU-V6E') is required for TPU batch inference."
            )

        canonical_accel = normalize_tpu_accelerator_type(accelerator_type)
        if canonical_accel not in TPU_ACCELERATOR_VALUES:
            raise ValueError(
                f"Unknown or unsupported TPU accelerator type: {accelerator_type!r}. "
                f"Supported TPU types: {sorted(TPU_ACCELERATOR_VALUES)}."
            )
        version = get_tpu_version_from_type(canonical_accel)

        engine_kwargs.setdefault("distributed_executor_backend", "ray")
        executor_backend = engine_kwargs["distributed_executor_backend"]
        if executor_backend != "ray":
            raise ValueError(
                "TPU batch inference requires distributed_executor_backend='ray'; "
                f"got {executor_backend!r}."
            )

        # engine_kwargs are free-form, so bool/float spellings of 1 (True, 1.0)
        # must be rejected before the == 1 checks.
        tp = _require_positive_int(
            engine_kwargs.get("tensor_parallel_size", 1), "tensor_parallel_size"
        )
        pp = _require_positive_int(
            engine_kwargs.get("pipeline_parallel_size", 1), "pipeline_parallel_size"
        )
        dp = _require_positive_int(
            engine_kwargs.get("data_parallel_size", 1), "data_parallel_size"
        )

        topology = tpu_config.topology.strip().lower()
        # Validate chips_per_vm packing early via the shared Core helper.
        resolve_chips_per_vm(topology, version, tpu_config.chips_per_vm)
        total_chips = get_num_chips_from_topology(topology)

        tp_multiplier = _vllm_tp_multiplier(version)
        expected_tp = total_chips * tp_multiplier
        if tp != expected_tp:
            raise ValueError(
                f"tensor_parallel_size must be {expected_tp} for topology "
                f"'{topology}' on {version} ({total_chips} physical chips × "
                f"vLLM TP multiplier {tp_multiplier}); got {tp}."
            )
        if pp != 1:
            raise ValueError(
                f"TPU batch inference currently supports pipeline_parallel_size=1; got {pp}."
            )
        if dp != 1:
            raise ValueError(
                f"TPU batch inference currently supports data_parallel_size=1; got {dp}."
            )

        # Declarative runtime_env merge. Preserve unrelated user variables while
        # forcing the values the TPU engine and its child workers require.
        merged_runtime_env = copy.deepcopy(runtime_env or {})
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

        resources_per_bundle = self._resolve_batch_worker_bundle(
            placement_group_config,
        )
        strategy = (
            placement_group_config.get("strategy") if placement_group_config else None
        ) or "PACK"

        handle = None
        success = False
        try:
            handle = self._create_slice_pg_handle(
                accelerator_type=canonical_accel,
                resources_per_bundle=resources_per_bundle,
                strategy=strategy,
                tpu_resource_per_chip=1,
                chips_per_vm=tpu_config.chips_per_vm,
            )

            try:
                _wait_for_placement_group(
                    handle.placement_group, DEFAULT_PG_READY_TIMEOUT_S
                )
            except ray.exceptions.GetTimeoutError as exc:
                raise TimeoutError(
                    f"Timed out after {DEFAULT_PG_READY_TIMEOUT_S}s waiting for TPU "
                    f"slice placement group readiness. Requested {canonical_accel} "
                    f"topology={topology} ({handle.num_hosts} hosts, "
                    f"{handle.num_bundles} bundles). This usually means the cluster "
                    "has no intact topology of that shape available."
                ) from exc

            handle.release_head_pgs()

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=handle.placement_group,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True,
            )

            map_batches_kwargs = {
                "num_cpus": PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST,
                "num_gpus": 0,
                "resources": {},
                "scheduling_strategy": scheduling_strategy,
                "runtime_env": merged_runtime_env,
            }
            success = True
            return map_batches_kwargs, handle.shutdown

        finally:
            if handle is not None and not success:
                try:
                    handle.shutdown()
                except Exception:
                    logger.exception(
                        "Failed to clean up TPU slice after batch scheduling "
                        "construction failed."
                    )
                finally:
                    self._slice_pg_wrapper = None


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
