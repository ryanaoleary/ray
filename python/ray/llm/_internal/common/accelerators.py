"""Shared accelerator configurations and backend abstractions for LLM serving and batch inference."""

import copy
import logging
import math
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator
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

# Bound driver-side wait for an eagerly acquired TPU placement group.
DEFAULT_PG_READY_TIMEOUT_S = 180.0

# Env vars injected into the Batch engine actor runtime so vLLM's TPU path
# uses Ray for process orchestration and reports one resource unit per chip.
# Users should not need to set these manually; conflicting values are rejected.
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


def validate_tpu_accelerator_type(value: str) -> str:
    """Normalize and validate a TPU accelerator type; raise ValueError if unknown."""
    canonical = normalize_tpu_accelerator_type(value)
    if canonical not in TPU_ACCELERATOR_VALUES:
        raise ValueError(
            f"Unknown or unsupported TPU accelerator type: {value!r}. "
            f"Supported TPU types: {sorted(TPU_ACCELERATOR_VALUES)}."
        )
    return canonical


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
            "TPU slice topology (e.g. '4x4', '2x4'). Required for topology-backed "
            "Batch and for deferred Serve SlicePG placement."
        ),
    )
    chips_per_vm: Optional[int] = Field(
        default=None,
        description=(
            "Optional chips-per-VM override for ambiguous topologies. Example: v6e "
            "'2x4' defaults to one 8-chip VM; set chips_per_vm=4 for two 4-chip VMs. "
            "When unset, Ray's get_chips_per_host default is used."
        ),
    )

    @field_validator("topology", mode="before")
    @classmethod
    def _normalize_topology(cls, value):
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"topology must be a string; got {value!r}.")
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("topology must be a non-empty string.")
        return normalized

    @field_validator("chips_per_vm", mode="before")
    @classmethod
    def _reject_bool_chips_per_vm(cls, value):
        # bool is a subclass of int; reject before Pydantic coerces True to 1.
        if isinstance(value, bool):
            raise ValueError(f"chips_per_vm must be a positive integer; got {value!r}.")
        return value

    @model_validator(mode="after")
    def _validate_chips_per_vm(self) -> "TPUConfig":
        if self.chips_per_vm is not None and not self.topology:
            raise ValueError("chips_per_vm requires topology to be specified.")
        if self.chips_per_vm is not None and self.chips_per_vm <= 0:
            raise ValueError(
                "chips_per_vm must be a positive integer; "
                f"got {self.chips_per_vm!r}."
            )
        if self.topology is not None:
            try:
                total_chips = get_num_chips_from_topology(self.topology)
            except Exception as exc:
                raise ValueError(
                    f"Invalid TPU topology {self.topology!r}. Expected a chip "
                    f"topology such as '4x4' or '2x2x1'."
                ) from exc
            if total_chips <= 0:
                raise ValueError(
                    f"Invalid TPU topology {self.topology!r}. Expected a chip "
                    f"topology such as '4x4' or '2x2x1'."
                )
            if self.chips_per_vm is not None and total_chips % self.chips_per_vm != 0:
                raise ValueError(
                    f"chips_per_vm ({self.chips_per_vm}) must divide the topology "
                    f"chip count ({total_chips} for '{self.topology}')."
                )
        return self


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
    # Stateless backend; no instance state.
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
    # Stateless backend; no instance state.
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

        # Serve passes TP*PP as num_devices and treats them as the chip count for
        # host packing (same as master Serve accelerators).
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
        strategy: Optional[str],
        name: str = "",
    ):
        """Create and own a topology-backed TPU placement group.

        Passes ``strategy`` through unmodified (aside from defaulting unset
        strategy to PACK). Serve always supplies an explicit strategy from
        deployment config (historical default PACK). Batch resolves its own
        default (SPREAD when topology is set) before calling this helper.
        """
        if not self._config.topology:
            raise ValueError(
                "TPU placement requires accelerator_config.topology to be set."
            )
        if self._slice_pg_wrapper is not None:
            logger.debug(
                "Existing TPU slice PG found. Shutting it down before creating a new one."
            )
            self.shutdown()

        topology = self._config.topology.strip().lower()
        version = get_tpu_version_from_type(accelerator_type)
        slice_kwargs: Dict[str, Any] = {
            "topology": topology,
            "accelerator_version": version,
            "resources_per_bundle": resources_per_bundle,
            # Serve historical default when callers omit strategy.
            "strategy": strategy or "PACK",
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
        provision a topology-backed placement group at runtime. This ensures
        multi-host TPU slices are gang-scheduled atomically according to their
        physical topology rather than fragmented across the cluster.
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

    def shutdown(self) -> None:
        """Release the owned SlicePG. Swallows errors for Serve replica teardown."""
        if self._slice_pg_wrapper is None:
            return
        try:
            logger.info("Shutting down TPU slice placement group.")
            self._slice_pg_wrapper.shutdown()
        except Exception as e:
            logger.warning(f"Failed to shut down TPU slice PG: {e}")
        finally:
            self._slice_pg_wrapper = None

    def _resolve_batch_worker_bundle(
        self,
        placement_group_config: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Resolve the Batch worker-resource template for one TPU bundle.

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
            # Strategy-only (or empty) config: use the default CPU-floor template.
            return {"CPU": cpu_floor}
        if not source_bundles:
            raise ValueError(
                "placement_group_config bundles must be non-empty when provided."
            )

        # Validate resource types before any numeric comparisons.
        for bundle in source_bundles:
            gpu = bundle.get("GPU", 0)
            if (
                isinstance(gpu, bool)
                or not isinstance(gpu, (int, float))
                or not math.isfinite(gpu)
            ):
                raise ValueError(
                    f"GPU resources per bundle must be a finite number; got {gpu!r}."
                )
            if gpu > 0:
                raise ValueError(
                    "GPU resources are not supported in TPU Batch "
                    f"placement_group_config bundles; got GPU={bundle['GPU']!r}."
                )
            if "TPU" in bundle:
                tpu = bundle["TPU"]
                if (
                    isinstance(tpu, bool)
                    or not isinstance(tpu, (int, float))
                    or not math.isfinite(tpu)
                ):
                    raise ValueError(
                        "TPU resources per bundle must be a positive integer; "
                        f"got {tpu!r}."
                    )
                if float(tpu) != int(tpu) or int(tpu) <= 0:
                    raise ValueError(
                        "TPU resources per bundle must be a positive integer; "
                        f"got {tpu!r}."
                    )

        has_positive_tpu = [bundle.get("TPU", 0) > 0 for bundle in source_bundles]
        if any(has_positive_tpu) and not all(has_positive_tpu):
            raise ValueError(
                "TPU Batch placement_group_config bundles cannot mix TPU-bearing "
                "and non-TPU bundles."
            )

        if len(source_bundles) > 1:
            logger.warning(
                "placement_group_config specified %d bundles, but topology-backed TPU "
                "scheduling derives the bundle count from topology %r. Using bundles[0] "
                "as a homogeneous per-worker template; the extra %d entries only "
                "participate in the homogeneity check.",
                len(source_bundles),
                self._config.topology,
                len(source_bundles) - 1,
            )

        if any(has_positive_tpu):
            worker_bundle = dict(source_bundles[0])
            if any(b != source_bundles[0] for b in source_bundles):
                raise ValueError(
                    "Heterogeneous TPU bundles are not supported when `topology` is set. "
                    "Use `bundle_per_worker` in `placement_group_config` for a uniform "
                    "per-worker resource template."
                )
        else:
            # No positive TPU: keep CPU/custom resources and omit TPU so
            # SlicePlacementGroup fills chips-per-VM (same as the default path).
            cleaned = [
                {k: v for k, v in b.items() if v != 0 and v != 0.0}
                for b in source_bundles
            ]
            if any(b != cleaned[0] for b in cleaned):
                raise ValueError(
                    "Heterogeneous placement_group_config bundles are not supported "
                    f"when `topology` is set; got {source_bundles!r}."
                )
            worker_bundle = cleaned[0]

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
        """Eagerly reserve one topology-backed TPU placement group for Batch.

        Validates config, waits for SlicePG readiness, releases head PGs, and
        returns map kwargs plus a ``close_fn`` that tears down the slice.
        Do not use ``self.shutdown`` as ``close_fn``: that path swallows errors
        for Serve replica teardown, while Batch must surface failures so
        ``Processor.close()`` can retry.
        """
        if not self._config.topology:
            raise ValueError(
                "TPU batch inference requires accelerator_config.topology. "
                "Omit accelerator_config (or use GPUConfig) for GPU scheduling."
            )

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
        canonical_accel = validate_tpu_accelerator_type(accelerator_type)
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

        topology = self._config.topology.strip().lower()
        total_chips = get_num_chips_from_topology(topology)
        if tp != total_chips:
            raise ValueError(
                f"tensor_parallel_size must be {total_chips} for topology "
                f"'{topology}' on {version} ({total_chips} physical chips / "
                f"vLLM devices); got {tp}."
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
        # Topology-backed Batch defaults to SPREAD (matches SlicePlacementGroup
        # and JaxTrainer TPU examples). Serve keeps its own PACK default.
        # Users can still override via placement_group_config["strategy"].
        strategy = (placement_group_config or {}).get("strategy") or "SPREAD"

        handle = self._create_slice_pg_handle(
            accelerator_type=canonical_accel,
            resources_per_bundle=resources_per_bundle,
            strategy=strategy,
        )
        try:
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
            # Head PGs are temporary reservation markers that atomically claim a
            # slice label before the worker PG is scheduled. Once the worker PG is
            # ready they are redundant; releasing them here frees those markers for
            # other jobs. Serve's create_placement_group path does not call this
            # today (pre-existing asymmetry — Batch readiness is eager at build time).
            handle.release_head_pgs()
        except Exception:
            try:
                handle.shutdown()
            except Exception:
                logger.exception(
                    "Failed to clean up TPU slice after batch scheduling "
                    "construction failed."
                )
            finally:
                self._slice_pg_wrapper = None
            raise

        def close_fn() -> None:
            owned = self._slice_pg_wrapper
            if owned is None:
                return
            # Clear only after a successful shutdown so a failed close can retry.
            owned.shutdown()
            self._slice_pg_wrapper = None

        map_batches_kwargs = {
            # Bundle 0 CPU is sized exactly for the Ray Data engine actor + user
            # map work. This is only safe because vLLM's Ray TPU executor requests
            # num_cpus=0 for its workers (verified against vLLM 0.26.0,
            # vllm/v1/executor/ray_executor.py non-GPU branch). If that changes,
            # child tasks captured into this PG will queue forever rather than
            # fail loudly.
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
        return map_batches_kwargs, close_fn


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
