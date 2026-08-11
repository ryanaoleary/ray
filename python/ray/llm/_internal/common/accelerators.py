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
from ray._private.accelerators.tpu import (
    get_chips_per_host,
    get_num_chips_from_topology,
)
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.tpu import (
    RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR,
    get_tpu_version_from_type,
    slice_placement_group,
)

logger = logging.getLogger(__name__)

# Constants for TPU batch scheduling
PARENT_ACTOR_CPU_RESERVE = 1
DEFAULT_USER_CPU_PER_HOST = 1
CPU_ACCELERATOR_TYPE_LITERAL = "CPU"

# Bound driver-side wait for an eagerly acquired TPU SlicePG.
DEFAULT_PG_READY_TIMEOUT_S = 180.0

# Required env for the TPU distributed executor backend.
TPU_MULTIHOST_BACKEND_ENV_VAR = "TPU_MULTIHOST_BACKEND"
TPU_ENGINE_ENV_VARS = {
    TPU_MULTIHOST_BACKEND_ENV_VAR: "ray",
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
    topology: Optional[str] = None
    chips_per_vm: Optional[int] = None


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
        """Return ``(map_batches_kwargs, optional close_fn)`` for batch inference.

        Implementations may populate accelerator-specific defaults in
        ``engine_kwargs``; callers should pass a private mutable copy. Backends
        without Batch support raise ``NotImplementedError``.
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
    """TPU backend shared by Ray Serve and Ray Data batch inference."""

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
        chips_per_host = (
            self._config.chips_per_vm
            if self._config.chips_per_vm is not None
            else get_chips_per_host(topology, version)
        )
        if chips_per_host <= 0:
            raise ValueError(
                f"Resolved chips per host must be positive, got {chips_per_host}"
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

        # Serve semantics: homogeneous positive-TPU bundles, else TPU:1.
        if bundles:
            tpu_bundles = [b for b in bundles if b.get("TPU", 0) > 0]
            if not tpu_bundles:
                worker_bundle = {"TPU": 1}
            else:
                worker_bundle = tpu_bundles[0]
                if any(b != worker_bundle for b in tpu_bundles):
                    raise ValueError(
                        "Heterogeneous TPU bundles are not supported when `topology` is set. "
                        "A multi-host TPU slice requires homogeneous resource bundles across all workers. "
                        "Please use `bundle_per_worker` in `placement_group_config` to define uniform worker resources."
                    )
        else:
            worker_bundle = {"TPU": 1}

        self._create_slice_pg_handle(
            accelerator_type=accelerator_type_str,
            resources_per_bundle=worker_bundle,
            strategy=strategy,
            name=name,
        )
        return self._slice_pg_wrapper.placement_group

    def _create_slice_pg_handle(
        self,
        *,
        accelerator_type: str,
        resources_per_bundle: Dict[str, float],
        strategy: str,
        name: str = "",
    ):
        """Create and own a topology-backed SlicePlacementGroup."""
        if not self._config.topology:
            raise ValueError(
                "TPU slice placement requires accelerator_config.topology to be set."
            )
        if self._slice_pg_wrapper is not None:
            logger.debug(
                "Existing TPU slice PG found. Shutting it down before creating a new one."
            )
            self.shutdown()

        slice_kwargs: Dict[str, Any] = {
            "topology": self._config.topology.strip().lower(),
            "accelerator_version": get_tpu_version_from_type(accelerator_type),
            "resources_per_bundle": resources_per_bundle,
            "strategy": strategy,
            "name": name,
        }
        if self._config.chips_per_vm is not None:
            slice_kwargs["chips_per_vm"] = self._config.chips_per_vm
        self._slice_pg_wrapper = slice_placement_group(**slice_kwargs)
        return self._slice_pg_wrapper

    @property
    def requires_deferred_placement_group(self) -> bool:
        """
        If a TPU topology is specified, we defer PG creation so the replica can
        provision a `SlicePlacementGroup` at runtime. This ensures multi-host
        TPU slices are gang-scheduled atomically according to their physical
        topology rather than fragmented across the cluster.
        """
        return bool(self._config.topology)

    @property
    def requires_remote_initialization(self) -> bool:
        return True

    def get_remote_options(self, accelerator_type_str: str = None):
        # The PlacementGroupSchedulingStrategy natively handles routing the task to
        # the correct hardware. We omit TPU resource requests to avoid consuming
        # chips that the model engine workers must use.
        options: Dict[str, Any] = {"resources": {}}
        if accelerator_type_str:
            # Pin the task to the TPU accelerator to avoid scheduling on a CPU bundle.
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

    def _resolve_batch_worker_bundle(
        self,
        placement_group_config: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Resolve the Batch worker-resource template for one SlicePG bundle.

        Default omits TPU so Ray fills chips-per-VM. Explicit configs supply a
        homogeneous template. Always apply the parent-actor CPU floor so the
        Ray Data engine actor and user map work can admit onto bundle 0.
        """
        cpu_floor = float(PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST)
        if placement_group_config is None:
            return {"CPU": cpu_floor}

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
        if not source_bundles:
            raise ValueError(
                "placement_group_config bundles must be non-empty when provided."
            )

        has_positive_tpu = [bundle.get("TPU", 0) > 0 for bundle in source_bundles]
        if any(has_positive_tpu) and not all(has_positive_tpu):
            raise ValueError(
                "TPU Batch placement_group_config bundles cannot mix TPU-bearing "
                "and non-TPU bundles."
            )

        for bundle in source_bundles:
            if bundle.get("GPU", 0) > 0:
                raise ValueError(
                    "GPU resources are not supported in TPU Batch "
                    f"placement_group_config bundles; got GPU={bundle['GPU']!r}."
                )
            if "TPU" not in bundle:
                continue
            tpu = bundle["TPU"]
            if isinstance(tpu, bool) or not isinstance(tpu, (int, float)):
                raise ValueError(
                    f"TPU resources per bundle must be a positive number; got {tpu!r}."
                )
            if float(tpu) != int(tpu) or int(tpu) <= 0:
                raise ValueError(
                    f"TPU resources per bundle must be a positive integer; got {tpu!r}."
                )

        if any(has_positive_tpu):
            worker_bundle = dict(source_bundles[0])
            if any(b != source_bundles[0] for b in source_bundles):
                raise ValueError(
                    "Heterogeneous TPU bundles are not supported when `topology` is set."
                )
        else:
            # No positive TPU: preserve CPU/custom resources and add TPU:1 so
            # SlicePG still materializes chip-bearing bundles.
            cleaned = [
                {k: v for k, v in b.items() if v != 0 and v != 0.0}
                for b in source_bundles
            ]
            if any(b != cleaned[0] for b in cleaned):
                raise ValueError(
                    "Heterogeneous placement_group_config bundles are not supported "
                    f"when `topology` is set; got {source_bundles!r}."
                )
            worker_bundle = {**cleaned[0], "TPU": 1}

        out = {k: v for k, v in worker_bundle.items() if v != 0 and v != 0.0}
        out["CPU"] = max(float(out.get("CPU", 0.0)), cpu_floor)
        return out

    def build_batch_scheduling_options(
        self,
        *,
        accelerator_type: Optional[str],
        engine_kwargs: Dict[str, Any],
        placement_group_config: Optional[Dict[str, Any]],
        runtime_env: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Optional[Callable[[], None]]]:
        """Return Batch map_batches kwargs; SlicePG only when topology is set."""
        if self._config.topology:
            return self._build_topology_batch_scheduling_options(
                accelerator_type=accelerator_type,
                engine_kwargs=engine_kwargs,
                placement_group_config=placement_group_config,
                runtime_env=runtime_env,
            )
        return self._build_single_host_batch_scheduling_options(
            accelerator_type=accelerator_type,
            engine_kwargs=engine_kwargs,
            placement_group_config=placement_group_config,
            runtime_env=runtime_env,
        )

    def _build_single_host_batch_scheduling_options(
        self,
        *,
        accelerator_type: Optional[str],
        engine_kwargs: Dict[str, Any],
        placement_group_config: Optional[Dict[str, Any]],
        runtime_env: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Optional[Callable[[], None]]]:
        """Schedule via TPU resource requests (no SlicePG).

        The caller is responsible for sizing ``tensor_parallel_size`` /
        ``placement_group_config`` to match available single-host TPU chips.
        For multi-host topologies, set ``accelerator_config.topology`` instead.
        """
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

        tp = _require_positive_int(
            engine_kwargs.get("tensor_parallel_size", 1), "tensor_parallel_size"
        )
        pp = _require_positive_int(
            engine_kwargs.get("pipeline_parallel_size", 1), "pipeline_parallel_size"
        )
        num_bundles = tp * pp
        pg_config = (
            copy.deepcopy(placement_group_config) if placement_group_config else None
        )
        if pg_config is not None:
            bundle_per_worker = pg_config.pop("bundle_per_worker", None)
            if bundle_per_worker is not None:
                pg_config["bundles"] = [
                    bundle_per_worker.copy() for _ in range(num_bundles)
                ]

        engine_kwargs.setdefault(
            "distributed_executor_backend",
            "uni" if num_bundles == 1 else "ray",
        )
        executor_backend = engine_kwargs["distributed_executor_backend"]

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

        map_batches_kwargs: Dict[str, Any] = {
            "runtime_env": merged_runtime_env,
            "accelerator_type": canonical_accel,
            "num_gpus": 0,
        }
        if executor_backend == "ray":
            map_batches_kwargs["ray_remote_args_fn"] = partial(
                _tpu_ray_scheduling_strategy_fn,
                num_bundles,
                canonical_accel,
                pg_config,
                self,
            )
            map_batches_kwargs["resources"] = {}
        elif not pg_config:
            map_batches_kwargs["resources"] = {"TPU": float(num_bundles)}
        else:
            resource_counter = Counter()
            for bundle in pg_config["bundles"]:
                resource_counter.update(bundle)
            total_cpus = resource_counter.pop("CPU", 0)
            resource_counter.pop("GPU", None)
            if total_cpus:
                map_batches_kwargs["num_cpus"] = total_cpus
            if resource_counter:
                map_batches_kwargs["resources"] = dict(resource_counter)
            else:
                map_batches_kwargs["resources"] = {}
        return map_batches_kwargs, None

    def _build_topology_batch_scheduling_options(
        self,
        *,
        accelerator_type: Optional[str],
        engine_kwargs: Dict[str, Any],
        placement_group_config: Optional[Dict[str, Any]],
        runtime_env: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Optional[Callable[[], None]]]:
        """Eagerly reserve one topology-backed SlicePG for Batch."""
        tpu_config = self._config
        assert tpu_config.topology is not None

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
                f"Topology-backed TPU batch inference requires "
                f"{RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR} == 1; got {driver_rpc}."
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
        if engine_kwargs["distributed_executor_backend"] != "ray":
            raise ValueError(
                "Topology-backed TPU batch inference requires "
                "distributed_executor_backend='ray'; "
                f"got {engine_kwargs['distributed_executor_backend']!r}."
            )

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
                "Topology-backed TPU batch inference currently supports "
                f"pipeline_parallel_size=1; got {pp}."
            )
        if dp != 1:
            raise ValueError(
                "Topology-backed TPU batch inference currently supports "
                f"data_parallel_size=1; got {dp}."
            )

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

        resources_per_bundle = self._resolve_batch_worker_bundle(placement_group_config)
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
                    f"{handle.num_bundles} bundles)."
                ) from exc

            handle.release_head_pgs()
            map_batches_kwargs = {
                "num_cpus": PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST,
                "num_gpus": 0,
                "resources": {},
                "scheduling_strategy": PlacementGroupSchedulingStrategy(
                    placement_group=handle.placement_group,
                    placement_group_bundle_index=0,
                    placement_group_capture_child_tasks=True,
                ),
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


def _tpu_ray_scheduling_strategy_fn(
    num_bundles_per_replica: int,
    accelerator_type: Optional[str] = None,
    placement_group_config: Optional[Dict[str, Any]] = None,
    backend: Optional["TPUAccelerator"] = None,
) -> Dict[str, Any]:
    """Dynamic PG creation for single-host TPU Batch (no topology / SlicePG)."""

    def _get_bundle() -> Dict[str, float]:
        bundle: Dict[str, float] = {"TPU": 1, "CPU": 1}
        if accelerator_type:
            bundle[f"accelerator_type:{accelerator_type}"] = 0.001
        return bundle

    if placement_group_config:
        placement_group_config = copy.deepcopy(placement_group_config)
        bundles = placement_group_config.get("bundles") or []
        if accelerator_type:
            for bundle in bundles:
                bundle[f"accelerator_type:{accelerator_type}"] = 0.001
        strategy = placement_group_config.get("strategy") or "PACK"
        if backend is not None:
            pg = backend.create_placement_group(
                bundles=bundles,
                strategy=strategy,
                name="",
                accelerator_type_str=accelerator_type,
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
                accelerator_type_str=accelerator_type,
            )
        else:
            pg = ray.util.placement_group(bundles, strategy="PACK")
    return dict(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            pg, placement_group_capture_child_tasks=True
        )
    )


def get_accelerator_backend(
    accelerator_config: AcceleratorConfig,
) -> AcceleratorBackend:
    """Return the backend implementation for a resolved accelerator config."""
    if isinstance(accelerator_config, TPUConfig):
        return TPUAccelerator(accelerator_config)
    if isinstance(accelerator_config, GPUConfig):
        return GPUAccelerator()
    if isinstance(accelerator_config, CPUConfig):
        return CPUAccelerator()
    raise TypeError(f"Unsupported accelerator config: {accelerator_config!r}")
