"""Comprehensive hermetic unit tests for TPU batch scheduling and lifecycle.

All tests run on CPU CI without TPU hardware or placement group engines.
"""

import gc
import logging
import pickle
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

import ray
from ray._private.accelerators.tpu import infer_tpu_pod_type_from_topology
from ray.data import ActorPoolStrategy
from ray.data.llm import (
    build_processor,
    vLLMEngineProcessorConfig,
)
from ray.llm._internal.batch.processor.base import Processor
from ray.llm._internal.batch.processor.vllm_engine_proc import _ManagedVLLMProcessor
from ray.llm._internal.common.accelerators import (
    DEFAULT_PG_READY_TIMEOUT_S,
    DEFAULT_USER_CPU_PER_HOST,
    PARENT_ACTOR_CPU_RESERVE,
    RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR,
    SLICE_READY_TIMEOUT_ENV_VAR,
    TPU_ACCELERATOR_VALUES,
    TPU_ENGINE_ENV_VARS,
    AcceleratorBackend,
    AnyAcceleratorConfig,
    CPUAccelerator,
    CPUConfig,
    GPUAccelerator,
    GPUConfig,
    TPUAccelerator,
    TPUConfig,
    _slice_ready_timeout_s,
    _vllm_tp_multiplier,
    get_accelerator_backend,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# -------------------------------------------------------------------------
# Test Fakes and Helpers
# -------------------------------------------------------------------------

_ACCEL = "ray.llm._internal.common.accelerators"


def _tpu_options(
    backend: TPUAccelerator,
    *,
    accelerator_type: str,
    tensor_parallel_size: int,
    executor_backend: str = "ray",
    placement_group_config: Optional[Dict[str, Any]] = None,
    runtime_env: Optional[Dict[str, Any]] = None,
    concurrency: Any = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
) -> Tuple[Dict[str, Any], Optional[Callable[[], None]]]:
    """Thin wrapper around the Batch scheduling hook used by tests."""
    return backend.build_batch_scheduling_options(
        accelerator_type=accelerator_type,
        engine_kwargs={
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "data_parallel_size": data_parallel_size,
            "distributed_executor_backend": executor_backend,
        },
        placement_group_config=placement_group_config,
        runtime_env=runtime_env,
        concurrency=concurrency,
    )


def _gpu_options(
    *,
    accelerator_type: Optional[str] = None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    executor_backend: Optional[str] = None,
    placement_group_config: Optional[Dict[str, Any]] = None,
    runtime_env: Optional[Dict[str, Any]] = None,
    concurrency: Any = 1,
) -> Tuple[Dict[str, Any], Optional[Callable[[], None]]]:
    engine_kwargs: Dict[str, Any] = {
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
    }
    if executor_backend is not None:
        engine_kwargs["distributed_executor_backend"] = executor_backend
    return GPUAccelerator().build_batch_scheduling_options(
        accelerator_type=accelerator_type,
        engine_kwargs=engine_kwargs,
        placement_group_config=placement_group_config,
        runtime_env=runtime_env,
        concurrency=concurrency,
    )


class FakePlacementGroup:
    def __init__(
        self,
        pg_id: str = "pg-fake-123",
        *,
        num_bundles: int = 4,
        chips_per_host: int = 4,
        cpu_per_host: float = 2.0,
    ):
        self.id = pg_id
        self.bundle_specs = [
            {"CPU": cpu_per_host, "TPU": float(chips_per_host)}
            for _ in range(num_bundles)
        ]

    def ready(self):
        return None


class FakeSlicePlacementGroupHandle:
    def __init__(
        self,
        topology: str = "4x4",
        num_hosts: int = 4,
        chips_per_host: int = 4,
        slice_name: Optional[str] = "tpu-slice-0",
    ):
        self.topology = topology
        self.num_hosts = num_hosts
        self.num_bundles = num_hosts
        self.chips_per_host = chips_per_host
        self.devices_per_host = chips_per_host
        self.placement_group = FakePlacementGroup(
            num_bundles=num_hosts, chips_per_host=chips_per_host
        )
        # Single-host SlicePlacementGroups do not attach slice-name labels.
        if slice_name is None or num_hosts <= 1:
            self.bundle_label_selector: List[Dict[str, str]] = [
                {} for _ in range(num_hosts)
            ]
        else:
            self.bundle_label_selector = [
                {"ray.io/tpu-slice-name": slice_name} for _ in range(num_hosts)
            ]
        self.released_head_pgs = 0
        self.shutdown_calls = 0

    def release_head_pgs(self):
        self.released_head_pgs += 1

    def shutdown(self):
        self.shutdown_calls += 1


def _install_tpu_slice_fakes(
    monkeypatch,
    fake_handle: FakeSlicePlacementGroupHandle,
    *,
    on_slice=None,
    on_wait=None,
):
    """Patch the public TPU seams used by ``TPUAccelerator.build_batch_scheduling_options``."""
    from ray._private.accelerators.tpu import (
        get_chips_per_host,
        get_num_chips_from_topology,
    )
    from ray.util.tpu import get_tpu_worker_resources

    def _slice(*args, **kwargs):
        if on_slice is not None:
            on_slice(*args, **kwargs)
        # Keep the fake handle consistent with Ray's SlicePG arithmetic so
        # reserved-layout validation sees the execution bundle count.
        topology = kwargs.get("topology", fake_handle.topology)
        version = kwargs.get("accelerator_version", "v6e")
        resources = kwargs.get("resources_per_bundle") or {}
        tpu_rpc = kwargs.get("tpu_resource_per_chip") or 1
        chips_per_vm_override = kwargs.get("chips_per_vm")
        num_bundles, bundle_resources = get_tpu_worker_resources(
            topology=topology,
            accelerator_type=f"TPU-{version.upper()}",
            resources_per_worker=resources,
            num_slices=1,
            chips_per_vm=chips_per_vm_override,
            tpu_resource_per_chip=tpu_rpc,
        )
        chips_per_vm = (
            chips_per_vm_override
            if chips_per_vm_override is not None
            else get_chips_per_host(topology, version)
        )
        total_chips = get_num_chips_from_topology(topology)
        num_vms = total_chips // chips_per_vm
        fake_handle.topology = topology
        fake_handle.num_hosts = num_vms
        fake_handle.num_bundles = num_bundles
        fake_handle.chips_per_host = chips_per_vm
        fake_handle.devices_per_host = chips_per_vm
        fake_handle.placement_group = FakePlacementGroup(
            num_bundles=num_bundles,
            chips_per_host=int(bundle_resources.get("TPU", chips_per_vm)),
            cpu_per_host=float(bundle_resources.get("CPU", 2.0)),
        )
        if num_vms <= 1:
            fake_handle.bundle_label_selector = [{} for _ in range(num_bundles)]
        else:
            fake_handle.bundle_label_selector = [
                {"ray.io/tpu-slice-name": "tpu-slice-0"} for _ in range(num_bundles)
            ]
        return fake_handle

    def _wait(pg, timeout_s):
        if on_wait is not None:
            on_wait(pg, timeout_s)

    monkeypatch.setattr(f"{_ACCEL}.slice_placement_group", _slice)
    monkeypatch.setattr(f"{_ACCEL}._wait_for_placement_group", _wait)
    return fake_handle


@pytest.fixture
def mock_tpu_slice_environment(monkeypatch):
    """Hermetic fixture providing a mocked TPU slice placement group and node lookup."""
    # Pin the driver precondition so the suite cannot inherit it from the host.
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="4x4", num_hosts=4, chips_per_host=4
    )
    _install_tpu_slice_fakes(monkeypatch, fake_handle)
    monkeypatch.setattr(
        "ray.llm._internal.batch.processor.vllm_engine_proc.download_model_files",
        lambda *args, **kwargs: "/tmp/mock-model",
    )
    return fake_handle


# -------------------------------------------------------------------------
# Config normalization and validation tests
# -------------------------------------------------------------------------


def test_dict_accelerator_config_normalizes():
    """Verify public dict accelerator_config normalizes to internal TPUConfig."""
    config = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config={"kind": "tpu", "topology": "4x4"},
    )
    assert isinstance(config.accelerator_config, TPUConfig)
    assert config.accelerator_config.kind == "tpu"
    assert config.accelerator_config.topology == "4x4"


def test_canonicalize_tpu_spelling():
    """Verify TPU spelling variants are normalized to uppercase hyphenated format."""
    for raw in ["TPU-V6E", "tpu-v6e", "TPU_V6E", " TPU-V6E "]:
        cfg = vLLMEngineProcessorConfig(
            model_source="test-model",
            accelerator_type=raw,
            accelerator_config=TPUConfig(topology="4x4"),
        )
        assert cfg.accelerator_type == "TPU-V6E"


def test_raw_concurrency_validation():
    """Verify strict raw concurrency validation on TPU configs prior to type coercion."""
    # 1. Valid cases: omitted, 1
    cfg_default = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config=TPUConfig(topology="4x4"),
    )
    assert cfg_default.concurrency == 1

    cfg_one = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config=TPUConfig(topology="4x4"),
        concurrency=1,
    )
    assert cfg_one.concurrency == 1

    # 2. Invalid cases: True, 1.0, "1", (1, 1), 2
    for invalid_concurrency in [True, 1.0, "1", (1, 1), 2]:
        with pytest.raises(
            ValueError, match="TPU batch inference requires concurrency=1"
        ):
            vLLMEngineProcessorConfig(
                model_source="test-model",
                accelerator_type="TPU-V6E",
                accelerator_config=TPUConfig(topology="4x4"),
                concurrency=invalid_concurrency,
            )

    # 3. GPU configs retain normal concurrency behavior (including autoscaling tuples)
    cfg_gpu = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="A100",
        accelerator_config=GPUConfig(),
        concurrency=(1, 4),
    )
    assert cfg_gpu.concurrency == (1, 4)


def test_config_normalization_typed_tpu():
    config = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config={"kind": "tpu", "topology": "4x4"},
    )
    assert isinstance(config.accelerator_config, TPUConfig)
    assert config.accelerator_config.topology == "4x4"


def test_config_default_inference():
    # 1. TPU without topology fails at config construction
    with pytest.raises(ValueError, match="requires accelerator_config"):
        vLLMEngineProcessorConfig(
            model_source="test-model",
            accelerator_type="TPU-V6E",
        )
    with pytest.raises(ValueError, match="requires accelerator_config"):
        vLLMEngineProcessorConfig(
            model_source="test-model",
            accelerator_type="TPU-V6E",
            accelerator_config={"kind": "tpu"},
        )

    # 2. GPU / absent accelerator_type with absent accelerator_config infers GPUConfig()
    cfg_gpu = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="A100",
    )
    assert isinstance(cfg_gpu.accelerator_config, GPUConfig)


def test_validation_matrix_tpu():
    # 1. Valid: Known TPU + TPUConfig(topology="4x4")
    cfg = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config=TPUConfig(topology="4x4"),
    )
    assert cfg.accelerator_type == "TPU-V6E"

    # 2. Known TPU + GPUConfig -> rejects
    with pytest.raises(ValueError, match="GPUConfig cannot be used with TPU"):
        vLLMEngineProcessorConfig(
            model_source="test-model",
            accelerator_type="TPU-V6E",
            accelerator_config=GPUConfig(),
        )

    # 3. Non-TPU type + TPUConfig -> rejects
    with pytest.raises(ValueError, match="TPUConfig requires a TPU accelerator_type"):
        vLLMEngineProcessorConfig(
            model_source="test-model",
            accelerator_type="A100",
            accelerator_config=TPUConfig(topology="4x4"),
        )

    # 4. Explicit CPU type -> rejects
    with pytest.raises(
        ValueError, match="Explicit 'CPU' accelerator type is not supported"
    ):
        vLLMEngineProcessorConfig(
            model_source="test-model",
            accelerator_type="CPU",
        )

    # 5. CPUConfig -> rejects
    with pytest.raises(ValueError, match="CPUConfig is not supported"):
        vLLMEngineProcessorConfig(
            model_source="test-model",
            accelerator_config=CPUConfig(),
        )

    # 6. Unknown TPU* type -> rejects (does not fall back to GPU)
    with pytest.raises(ValueError, match="Unknown or unsupported TPU accelerator type"):
        vLLMEngineProcessorConfig(
            model_source="test-model",
            accelerator_type="TPU-UNKNOWN-99",
        )

    # 7. TPU + placement_group_config -> accepted (Serve-compatible granularity)
    cfg_tpu_pg = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config=TPUConfig(topology="4x4"),
        placement_group_config={"bundle_per_worker": {"TPU": 1}},
    )
    assert cfg_tpu_pg.placement_group_config == {
        "bundle_per_worker": {"TPU": 1},
    }

    # 8. GPU + placement_group_config -> valid and not rejected
    cfg_gpu_pg = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="A100",
        accelerator_config=GPUConfig(),
        placement_group_config={"bundle_per_worker": {"CPU": 1, "GPU": 1}},
    )
    assert cfg_gpu_pg.placement_group_config is not None

    # 9. TPU with placement_group_config=None -> valid
    cfg_tpu_none_pg = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config=TPUConfig(topology="4x4"),
        placement_group_config=None,
    )
    assert cfg_tpu_none_pg.placement_group_config is None


# -------------------------------------------------------------------------
# Builder and backend integration tests
# -------------------------------------------------------------------------


def test_builder_to_backend_tpu(mock_tpu_slice_environment):
    fake_handle = mock_tpu_slice_environment
    config = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config=TPUConfig(topology="4x4"),
        concurrency=1,
        engine_kwargs={
            "tensor_parallel_size": 16,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
        },
    )

    processor = build_processor(config)
    assert isinstance(processor, _ManagedVLLMProcessor)
    assert callable(processor._close_fn)

    # Ensure config.engine_kwargs is not mutated
    assert "distributed_executor_backend" not in config.engine_kwargs

    # Check stage construction kwargs
    vllm_stage = processor.get_stage_by_name("vLLMEngineStage")
    kwargs = vllm_stage.map_batches_kwargs
    assert kwargs["num_cpus"] == PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST
    assert kwargs["num_gpus"] == 0
    assert kwargs["resources"] == {}
    assert "accelerator_type" not in kwargs
    assert kwargs["runtime_env"]["env_vars"]["TPU_MULTIHOST_BACKEND"] == "ray"
    assert kwargs["runtime_env"]["env_vars"]["RAY_TPU_RESOURCE_PER_CHIP"] == "1"

    compute = kwargs["compute"]
    assert isinstance(compute, ActorPoolStrategy)
    assert compute.min_size == 1
    assert compute.max_size == 1

    strategy = kwargs["scheduling_strategy"]
    assert isinstance(strategy, PlacementGroupSchedulingStrategy)
    assert strategy.placement_group_bundle_index == 0
    assert strategy.placement_group_capture_child_tasks is True

    # The parent actor is scheduled into the SlicePG with child capture so
    # vLLM from_engine_args can publish the current PG into ParallelConfig.
    # Required TPU env vars are merged into runtime_env (not a separate UDF check).
    fn_kwargs = vllm_stage.fn_constructor_kwargs
    assert "reuse_current_placement_group" not in fn_kwargs
    assert "required_env_vars" not in fn_kwargs

    processor.close()
    assert fake_handle.shutdown_calls == 1
    assert processor._close_fn is None


def test_builder_returns_ordinary_processor_for_gpu(monkeypatch):
    monkeypatch.setattr(
        "ray.llm._internal.batch.processor.vllm_engine_proc.download_model_files",
        lambda *args, **kwargs: "/tmp/mock-model",
    )
    config = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="A100",
        accelerator_config=GPUConfig(),
        concurrency=1,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
        },
    )
    processor = build_processor(config)
    assert type(processor) is Processor


@pytest.mark.parametrize(
    "topology, tensor_parallel_size, chips_per_vm",
    [
        ("1x1", 1, 1),
        ("2x4", 8, 8),
    ],
)
def test_builder_defaults_tpu_executor_backend_to_ray(
    monkeypatch, topology, tensor_parallel_size, chips_per_vm
):
    """TPU topologies always default to the Ray executor, including single-chip 1x1."""
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    monkeypatch.setattr(
        "ray.llm._internal.batch.processor.vllm_engine_proc.download_model_files",
        lambda *args, **kwargs: "/tmp/mock-model",
    )
    fake_handle = FakeSlicePlacementGroupHandle(
        topology=topology,
        num_hosts=1,
        chips_per_host=chips_per_vm,
        slice_name=None,
    )
    _install_tpu_slice_fakes(monkeypatch, fake_handle)

    config = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config={"kind": "tpu", "topology": topology},
        concurrency=1,
        engine_kwargs={
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": 1,
        },
    )
    assert "distributed_executor_backend" not in config.engine_kwargs

    processor = build_processor(config)
    assert isinstance(processor, _ManagedVLLMProcessor)
    fn_kwargs = processor.get_stage_by_name("vLLMEngineStage").fn_constructor_kwargs
    assert fn_kwargs["engine_kwargs"]["distributed_executor_backend"] == "ray"
    # Caller config must remain unmodified.
    assert "distributed_executor_backend" not in config.engine_kwargs
    processor.close()
    assert fake_handle.shutdown_calls == 1


def test_builder_rejects_explicit_tpu_uni_executor(mock_tpu_slice_environment):
    config = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config={"kind": "tpu", "topology": "1x1"},
        concurrency=1,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "distributed_executor_backend": "uni",
        },
    )
    with pytest.raises(ValueError, match="executor_backend='ray'"):
        build_processor(config)
    assert mock_tpu_slice_environment.shutdown_calls == 0


@pytest.mark.parametrize(
    "tensor_parallel_size, expected_backend",
    [
        (1, "uni"),
        (2, "ray"),
    ],
)
def test_builder_preserves_gpu_executor_backend_defaults(
    monkeypatch, tensor_parallel_size, expected_backend
):
    monkeypatch.setattr(
        "ray.llm._internal.batch.processor.vllm_engine_proc.download_model_files",
        lambda *args, **kwargs: "/tmp/mock-model",
    )
    config = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="A100",
        accelerator_config=GPUConfig(),
        concurrency=1,
        engine_kwargs={
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": 1,
        },
    )
    processor = build_processor(config)
    assert type(processor) is Processor
    fn_kwargs = processor.get_stage_by_name("vLLMEngineStage").fn_constructor_kwargs
    assert (
        fn_kwargs["engine_kwargs"]["distributed_executor_backend"] == expected_backend
    )
    assert "distributed_executor_backend" not in config.engine_kwargs


def test_builder_cleanup_on_construction_failure(
    mock_tpu_slice_environment, monkeypatch, caplog
):
    """Verify builder cleans up acquired resources if stage construction fails and logs errors."""
    fake_handle = mock_tpu_slice_environment

    # Force a failure during stage building
    def failing_stage(*args, **kwargs):
        raise RuntimeError("Stage construction failed")

    monkeypatch.setattr(
        "ray.llm._internal.batch.processor.vllm_engine_proc.vLLMEngineStage",
        failing_stage,
    )

    config = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config=TPUConfig(topology="4x4"),
        concurrency=1,
        engine_kwargs={"tensor_parallel_size": 16},
    )

    with pytest.raises(RuntimeError, match="Stage construction failed"):
        build_processor(config)

    # Cleanup must be called exactly once
    assert fake_handle.shutdown_calls == 1

    # Also test when handle.shutdown itself fails
    fake_handle_failing = FakeSlicePlacementGroupHandle()

    def failing_shutdown():
        fake_handle_failing.shutdown_calls += 1
        raise ConnectionError("GCS unreachable during cleanup")

    fake_handle_failing.shutdown = failing_shutdown
    _install_tpu_slice_fakes(monkeypatch, fake_handle_failing)

    import ray.llm._internal.batch.processor.vllm_engine_proc as proc_mod

    log_records = []

    class RecordHandler(logging.Handler):
        def emit(self, record):
            log_records.append(record)

    handler = RecordHandler()
    proc_mod.logger.addHandler(handler)
    try:
        with pytest.raises(RuntimeError, match="Stage construction failed"):
            build_processor(config)
    finally:
        proc_mod.logger.removeHandler(handler)

    assert fake_handle_failing.shutdown_calls == 1
    assert any(
        "Failed to release accelerator batch resources after processor construction failed"
        in r.getMessage()
        for r in log_records
    )


# -------------------------------------------------------------------------
# Direct backend validation and ordering tests
# -------------------------------------------------------------------------


def test_env_var_constant():
    assert RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR == "RAY_TPU_RESOURCE_PER_CHIP"


def test_direct_backend_validation_failures(monkeypatch):
    backend = TPUAccelerator(TPUConfig(topology="4x4"))

    # 1. Missing topology
    with pytest.raises(
        ValueError, match="TPU batch inference requires an explicit `accelerator_config"
    ):
        _tpu_options(
            TPUAccelerator(TPUConfig()),
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            placement_group_config=None,
            runtime_env=None,
            concurrency=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        )

    # 2. Missing accelerator_type
    with pytest.raises(ValueError, match="`accelerator_type`.*is required"):
        _tpu_options(
            backend,
            accelerator_type=None,
            tensor_parallel_size=16,
            executor_backend="ray",
            placement_group_config=None,
            runtime_env=None,
            concurrency=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        )

    # 3. TP mismatch (15 != 16)
    with pytest.raises(ValueError, match="tensor_parallel_size must be"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=15,
            executor_backend="ray",
            placement_group_config=None,
            runtime_env=None,
            concurrency=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        )

    # 4. PP != 1
    with pytest.raises(ValueError, match="pipeline_parallel_size=1"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            placement_group_config=None,
            runtime_env=None,
            concurrency=1,
            pipeline_parallel_size=2,
            data_parallel_size=1,
        )

    # 4b. PP True / 1.0 must not silently pass the == 1 check
    for bad_pp in (True, 1.0):
        with pytest.raises(
            ValueError, match="pipeline_parallel_size must be a positive integer"
        ):
            _tpu_options(
                backend,
                accelerator_type="TPU-V6E",
                tensor_parallel_size=16,
                executor_backend="ray",
                concurrency=1,
                pipeline_parallel_size=bad_pp,
                data_parallel_size=1,
            )

    # 5. DP != 1
    with pytest.raises(ValueError, match="data_parallel_size=1"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            placement_group_config=None,
            runtime_env=None,
            concurrency=1,
            pipeline_parallel_size=1,
            data_parallel_size=2,
        )

    # 5b. DP True / 1.0 must not silently pass the == 1 check
    for bad_dp in (True, 1.0):
        with pytest.raises(
            ValueError, match="data_parallel_size must be a positive integer"
        ):
            _tpu_options(
                backend,
                accelerator_type="TPU-V6E",
                tensor_parallel_size=16,
                executor_backend="ray",
                concurrency=1,
                pipeline_parallel_size=1,
                data_parallel_size=bad_dp,
            )

    # 6. Concurrency bool True
    with pytest.raises(ValueError, match="concurrency=1"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            placement_group_config=None,
            runtime_env=None,
            concurrency=True,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        )

    # 7. Concurrency float 1.0
    with pytest.raises(ValueError, match="concurrency=1"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            placement_group_config=None,
            runtime_env=None,
            concurrency=1.0,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        )

    # 8. Invalid TPU-per-bundle granularity (does not divide chips/VM)
    with pytest.raises(ValueError, match="evenly divide"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            placement_group_config={"bundle_per_worker": {"TPU": 3}},
            runtime_env=None,
            concurrency=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        )

    # 9. Runtime env backend conflict
    with pytest.raises(ValueError, match="TPU_MULTIHOST_BACKEND"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            placement_group_config=None,
            runtime_env={"env_vars": {"TPU_MULTIHOST_BACKEND": "grpc"}},
            concurrency=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        )

    # 10. Runtime env resource-per-chip mismatch (integer 1 or non-string)
    with pytest.raises(ValueError, match="must be the string '1'"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            placement_group_config=None,
            runtime_env={"env_vars": {"RAY_TPU_RESOURCE_PER_CHIP": 1}},
            concurrency=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        )

    # 11. Driver resource per chip mismatch
    monkeypatch.setenv("RAY_TPU_RESOURCE_PER_CHIP", "2")
    with pytest.raises(ValueError, match="requires RAY_TPU_RESOURCE_PER_CHIP == 1"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            concurrency=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        )


@pytest.mark.parametrize(
    "accelerator_type, topology, total_chips, chips_per_vm, num_vms",
    [
        ("TPU-V6E", "1x1", 1, 1, 1),
        ("TPU-V6E", "2x2", 4, 4, 1),
        ("TPU-V6E", "2x4", 8, 8, 1),
        ("TPU-V6E", "4x4", 16, 4, 4),
    ],
)
def test_default_topology_layouts(
    monkeypatch, accelerator_type, topology, total_chips, chips_per_vm, num_vms
):
    """Ray's default chips-per-VM resolution drives single- and multi-VM layouts."""
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    slice_kwargs = []
    fake_handle = FakeSlicePlacementGroupHandle(
        topology=topology,
        num_hosts=num_vms,
        chips_per_host=chips_per_vm,
        slice_name=None if num_vms == 1 else "tpu-slice-0",
    )
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        on_slice=lambda *args, **kwargs: slice_kwargs.append(kwargs),
    )

    kwargs, close_fn = _tpu_options(
        TPUAccelerator(TPUConfig(topology=topology)),
        accelerator_type=accelerator_type,
        tensor_parallel_size=total_chips,
        executor_backend="ray",
        concurrency=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )

    assert kwargs["num_cpus"] == PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST
    assert kwargs["num_gpus"] == 0
    assert kwargs["resources"] == {}
    strategy = kwargs["scheduling_strategy"]
    assert strategy.placement_group_bundle_index == 0
    assert strategy.placement_group_capture_child_tasks is True
    assert len(fake_handle.placement_group.bundle_specs) == num_vms
    for bundle in fake_handle.placement_group.bundle_specs:
        assert bundle["TPU"] == float(chips_per_vm)
    assert len(slice_kwargs) == 1
    if num_vms == 1:
        # Single-VM SlicePGs skip slice-name reservation; pin generation/topology.
        expected_pod_type = infer_tpu_pod_type_from_topology(topology, accelerator_type)
        assert slice_kwargs[0]["bundle_label_selector"] == [
            {
                "ray.io/tpu-topology": topology,
                "ray.io/tpu-pod-type": expected_pod_type,
            }
        ]
    else:
        # Multi-VM path must not inject caller hardware labels.
        assert slice_kwargs[0]["bundle_label_selector"] is None
    close_fn()


@pytest.mark.parametrize("topology", ["2x4", "2X4", " 2x4 "])
def test_single_vm_slice_allocation_uses_topology_pod_type_labels(
    monkeypatch, topology
):
    """v6e 2x4 must constrain SlicePG placement via canonical TPU node labels."""
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    slice_kwargs = []
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="2x4", num_hosts=1, chips_per_host=8, slice_name=None
    )
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        on_slice=lambda *args, **kwargs: slice_kwargs.append(kwargs),
    )

    kwargs, close_fn = _tpu_options(
        TPUAccelerator(TPUConfig(topology=topology)),
        accelerator_type="TPU-V6E",
        tensor_parallel_size=8,
        executor_backend="ray",
        concurrency=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )

    assert len(slice_kwargs) == 1
    assert slice_kwargs[0]["topology"] == "2x4"
    assert slice_kwargs[0]["accelerator_version"] == "v6e"
    assert slice_kwargs[0]["resources_per_bundle"] == {
        "CPU": float(PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST),
    }
    assert slice_kwargs[0]["bundle_label_selector"] == [
        {
            "ray.io/tpu-topology": "2x4",
            "ray.io/tpu-pod-type": "v6e-8",
        }
    ]
    assert slice_kwargs[0]["strategy"] == "PACK"
    assert slice_kwargs[0]["tpu_resource_per_chip"] == 1
    # chips_per_vm omitted so Ray owns the default for this topology.
    assert "chips_per_vm" not in slice_kwargs[0]
    close_fn()


@pytest.mark.parametrize(
    "topology, tp, tpu_per_bundle, expected_bundles, chips_per_vm, num_vms",
    [
        ("2x4", 8, None, 1, 8, 1),
        ("2x4", 8, 8, 1, 8, 1),
        ("2x4", 8, 4, 2, 8, 1),
        ("2x4", 8, 2, 4, 8, 1),
        ("2x4", 8, 1, 8, 8, 1),
        ("4x4", 16, None, 4, 4, 4),
        ("4x4", 16, 4, 4, 4, 4),
        ("4x4", 16, 2, 8, 4, 4),
        ("4x4", 16, 1, 16, 4, 4),
    ],
)
def test_bundle_granularity_matrix(
    monkeypatch,
    topology,
    tp,
    tpu_per_bundle,
    expected_bundles,
    chips_per_vm,
    num_vms,
):
    """Physical VM count stays fixed while TPU-per-bundle changes PG workers."""
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    slice_kwargs = []
    fake_handle = FakeSlicePlacementGroupHandle(
        topology=topology,
        num_hosts=num_vms,
        chips_per_host=chips_per_vm,
        slice_name=None if num_vms == 1 else "tpu-slice-0",
    )
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        on_slice=lambda *args, **kwargs: slice_kwargs.append(kwargs),
    )

    pg_config = None
    if tpu_per_bundle is not None:
        pg_config = {"bundle_per_worker": {"TPU": tpu_per_bundle}}

    kwargs, close_fn = _tpu_options(
        TPUAccelerator(TPUConfig(topology=topology)),
        accelerator_type="TPU-V6E",
        tensor_parallel_size=tp,
        executor_backend="ray",
        placement_group_config=pg_config,
        concurrency=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )

    assert len(slice_kwargs) == 1
    assert fake_handle.num_hosts == num_vms
    assert fake_handle.num_bundles == expected_bundles
    assert len(fake_handle.placement_group.bundle_specs) == expected_bundles
    expected_tpu = float(chips_per_vm if tpu_per_bundle is None else tpu_per_bundle)
    for bundle in fake_handle.placement_group.bundle_specs:
        assert bundle["TPU"] == expected_tpu
        assert bundle["CPU"] >= float(
            PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST
        )

    if num_vms == 1:
        selectors = slice_kwargs[0]["bundle_label_selector"]
        assert selectors is not None
        assert len(selectors) == expected_bundles
        assert all(
            s
            == {
                "ray.io/tpu-topology": topology,
                "ray.io/tpu-pod-type": infer_tpu_pod_type_from_topology(
                    topology, "TPU-V6E"
                ),
            }
            for s in selectors
        )
        assert slice_kwargs[0]["strategy"] == (
            "STRICT_PACK" if expected_bundles > 1 else "PACK"
        )
    else:
        assert slice_kwargs[0]["bundle_label_selector"] is None

    close_fn()


def test_batch_cpu_floor_and_gpu_rejection(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="4x4", num_hosts=4, chips_per_host=4
    )
    slice_kwargs = []
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        on_slice=lambda *args, **kwargs: slice_kwargs.append(kwargs),
    )
    backend = TPUAccelerator(TPUConfig(topology="4x4"))

    kwargs, close_fn = _tpu_options(
        backend,
        accelerator_type="TPU-V6E",
        tensor_parallel_size=16,
        executor_backend="ray",
        placement_group_config={"bundle_per_worker": {"CPU": 1, "TPU": 1}},
        concurrency=1,
    )
    assert slice_kwargs[-1]["resources_per_bundle"]["CPU"] == float(
        PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST
    )
    assert slice_kwargs[-1]["resources_per_bundle"]["TPU"] == 1
    close_fn()

    for bad_pg in (
        {"bundle_per_worker": {"GPU": 1, "TPU": 1}},
        {"bundle_per_worker": {"GPU": 1}},
        {"bundle_per_worker": {"GPU": 1, "TPU": 0}},
    ):
        with pytest.raises(ValueError, match="GPU resources are not supported"):
            _tpu_options(
                backend,
                accelerator_type="TPU-V6E",
                tensor_parallel_size=16,
                executor_backend="ray",
                placement_group_config=bad_pg,
                concurrency=1,
            )


@pytest.mark.parametrize(
    "bad_bundle, match",
    [
        ({"TPU": 0}, "must be positive"),
        ({"TPU": -1}, "must be positive"),
        ({"TPU": 1.5}, "must be an integer"),
    ],
)
def test_batch_rejects_invalid_explicit_tpu_templates(monkeypatch, bad_bundle, match):
    """Explicit invalid TPU values must not fall back to Serve's TPU:1 default."""
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    backend = TPUAccelerator(TPUConfig(topology="4x4"))
    with pytest.raises(ValueError, match=match):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            placement_group_config={"bundle_per_worker": bad_bundle},
            concurrency=1,
        )


def test_batch_cpu_only_template_preserves_tpu1_fallback(monkeypatch):
    """Omitting TPU preserves CPU/custom fields while adding Batch's TPU:1 fallback.

    Chips-per-VM fill is only used when ``placement_group_config`` is ``None``.
    """
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    slice_kwargs = []
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="4x4", num_hosts=4, chips_per_host=4
    )
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        on_slice=lambda *args, **kwargs: slice_kwargs.append(kwargs),
    )
    backend = TPUAccelerator(TPUConfig(topology="4x4"))

    kwargs, close_fn = _tpu_options(
        backend,
        accelerator_type="TPU-V6E",
        tensor_parallel_size=16,
        executor_backend="ray",
        placement_group_config={"bundle_per_worker": {"CPU": 4}},
        concurrency=1,
    )
    assert slice_kwargs[-1]["resources_per_bundle"]["TPU"] == 1
    assert slice_kwargs[-1]["resources_per_bundle"]["CPU"] == 4.0
    close_fn()

    kwargs, close_fn = _tpu_options(
        backend,
        accelerator_type="TPU-V6E",
        tensor_parallel_size=16,
        executor_backend="ray",
        placement_group_config={
            "bundle_per_worker": {"CPU": 4, "special": 1},
        },
        concurrency=1,
    )
    assert slice_kwargs[-1]["resources_per_bundle"] == {
        "CPU": 4.0,
        "special": 1,
        "TPU": 1,
    }
    close_fn()

    # Explicit bundles list without TPU also preserves homogeneous extras.
    kwargs, close_fn = _tpu_options(
        backend,
        accelerator_type="TPU-V6E",
        tensor_parallel_size=16,
        executor_backend="ray",
        placement_group_config={
            "bundles": [{"CPU": 3, "custom": 2} for _ in range(16)],
        },
        concurrency=1,
    )
    assert slice_kwargs[-1]["resources_per_bundle"] == {
        "CPU": 3.0,
        "custom": 2,
        "TPU": 1,
    }
    close_fn()

    # CPU below the Batch floor is still raised, never lowered.
    kwargs, close_fn = _tpu_options(
        backend,
        accelerator_type="TPU-V6E",
        tensor_parallel_size=16,
        executor_backend="ray",
        placement_group_config={"bundle_per_worker": {"CPU": 1}},
        concurrency=1,
    )
    assert slice_kwargs[-1]["resources_per_bundle"]["CPU"] == float(
        PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST
    )
    assert slice_kwargs[-1]["resources_per_bundle"]["TPU"] == 1
    close_fn()


def test_single_vm_multi_bundle_upgrades_pack_to_strict_pack(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    slice_kwargs = []
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="2x4", num_hosts=1, chips_per_host=8, slice_name=None
    )
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        on_slice=lambda *args, **kwargs: slice_kwargs.append(kwargs),
    )
    kwargs, close_fn = _tpu_options(
        TPUAccelerator(TPUConfig(topology="2x4")),
        accelerator_type="TPU-V6E",
        tensor_parallel_size=8,
        executor_backend="ray",
        placement_group_config={
            "bundle_per_worker": {"TPU": 1},
            "strategy": "PACK",
        },
        concurrency=1,
    )
    assert fake_handle.num_bundles == 8
    assert slice_kwargs[-1]["strategy"] == "STRICT_PACK"
    close_fn()


def test_single_vm_multi_bundle_rejects_spread(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    backend = TPUAccelerator(TPUConfig(topology="2x4"))
    with pytest.raises(ValueError, match="PACK/STRICT_PACK"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=8,
            executor_backend="ray",
            placement_group_config={
                "bundle_per_worker": {"TPU": 1},
                "strategy": "SPREAD",
            },
            concurrency=1,
        )


def test_multi_vm_rejects_strict_pack_and_impossible_strict_spread(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    backend = TPUAccelerator(TPUConfig(topology="4x4"))
    with pytest.raises(ValueError, match="STRICT_PACK cannot represent"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            placement_group_config={
                "bundle_per_worker": {"TPU": 4},
                "strategy": "STRICT_PACK",
            },
            concurrency=1,
        )
    with pytest.raises(ValueError, match="STRICT_SPREAD requires one node per bundle"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            placement_group_config={
                "bundle_per_worker": {"TPU": 1},
                "strategy": "STRICT_SPREAD",
            },
            concurrency=1,
        )


def test_heterogeneous_tpu_bundles_rejected_for_batch(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    backend = TPUAccelerator(TPUConfig(topology="4x4"))
    with pytest.raises(ValueError, match="Heterogeneous TPU bundles"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            placement_group_config={
                "bundles": [{"TPU": 1}, {"TPU": 4}],
            },
            concurrency=1,
        )


@pytest.mark.parametrize(
    "version, expected",
    [
        ("v2", 1),
        ("v3", 1),
        ("v4", 1),
        ("v5p", 1),
        ("v5litepod", 1),
        ("v6e", 1),
        ("v7x", 2),
    ],
)
def test_vllm_tp_multiplier_by_tpu_generation(version, expected):
    """Lock the all-generation vLLM TP multiplier contract (v7x-only = 2)."""
    assert _vllm_tp_multiplier(version) == expected


def test_v7x_batch_requires_tp_equal_to_physical_chips_times_multiplier(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    slice_kwargs = []
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="2x2x1", num_hosts=1, chips_per_host=4, slice_name=None
    )
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        on_slice=lambda *args, **kwargs: slice_kwargs.append(kwargs),
    )
    backend = TPUAccelerator(TPUConfig(topology="2x2x1"))

    with pytest.raises(ValueError, match="tensor_parallel_size must be 8"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V7X",
            tensor_parallel_size=4,
            concurrency=1,
        )

    kwargs, close_fn = _tpu_options(
        backend,
        accelerator_type="TPU-V7X",
        tensor_parallel_size=8,
        executor_backend="ray",
        concurrency=1,
    )
    # Bundle layout still follows Ray scheduling resources (rpc=1 → TPU:4 / VM),
    # not framework device count.
    assert fake_handle.num_hosts == 1
    assert fake_handle.num_bundles == 1
    assert fake_handle.placement_group.bundle_specs[0]["TPU"] == 4.0
    # chips_per_vm omitted so Ray owns the Ironwood host default.
    assert "chips_per_vm" not in slice_kwargs[-1]
    close_fn()


def test_v7x_2x2x2_default_and_per_chip_bundle_layout(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    slice_kwargs = []
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="2x2x2", num_hosts=2, chips_per_host=4, slice_name="tpu-slice-0"
    )
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        on_slice=lambda *args, **kwargs: slice_kwargs.append(kwargs),
    )
    backend = TPUAccelerator(TPUConfig(topology="2x2x2"))

    kwargs, close_fn = _tpu_options(
        backend,
        accelerator_type="TPU-V7X",
        tensor_parallel_size=16,
        executor_backend="ray",
        concurrency=1,
    )
    assert fake_handle.num_hosts == 2
    assert fake_handle.num_bundles == 2
    assert slice_kwargs[-1]["strategy"] == "PACK"
    close_fn()

    kwargs, close_fn = _tpu_options(
        backend,
        accelerator_type="TPU-V7X",
        tensor_parallel_size=16,
        executor_backend="ray",
        placement_group_config={"bundle_per_worker": {"TPU": 1}},
        concurrency=1,
    )
    assert fake_handle.num_hosts == 2
    assert fake_handle.num_bundles == 8
    assert slice_kwargs[-1]["strategy"] == "PACK"
    close_fn()


def test_serve_create_placement_group_forwards_chips_per_vm(monkeypatch):
    """Shared TPUConfig.chips_per_vm must reach SlicePG from the Serve PG path."""
    slice_kwargs = []
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="2x4", num_hosts=2, chips_per_host=4, slice_name="tpu-slice-0"
    )
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        on_slice=lambda *args, **kwargs: slice_kwargs.append(kwargs),
    )
    backend = TPUAccelerator(TPUConfig(topology="2x4", chips_per_vm=4))
    pg = backend.create_placement_group(
        bundles=[{"TPU": 4, "CPU": 1}],
        strategy="PACK",
        name="serve-tpu-replica",
        accelerator_type_str="TPU-V6E",
    )
    assert pg is fake_handle.placement_group
    assert len(slice_kwargs) == 1
    assert slice_kwargs[0]["chips_per_vm"] == 4
    assert slice_kwargs[0]["resources_per_bundle"]["TPU"] == 4
    backend.shutdown()


def test_serve_default_bundles_respects_chips_per_vm():
    """Serve default_bundles must pack hosts with the resolved chips_per_vm."""
    backend = TPUAccelerator(TPUConfig(topology="2x4", chips_per_vm=4))
    bundles = backend.default_bundles(
        num_devices=8,
        accelerator_type_str="TPU-V6E",
    )
    assert bundles == [
        {"TPU": 4, "accelerator_type:TPU-V6E": 0.001},
        {"TPU": 4, "accelerator_type:TPU-V6E": 0.001},
    ]


def test_serve_default_bundles_then_create_pg_with_chips_per_vm(monkeypatch):
    """Serve placement_bundles path: default_bundles + create_placement_group."""
    slice_kwargs = []
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="2x4", num_hosts=2, chips_per_host=4, slice_name="tpu-slice-0"
    )
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        on_slice=lambda *args, **kwargs: slice_kwargs.append(kwargs),
    )
    backend = TPUAccelerator(TPUConfig(topology="2x4", chips_per_vm=4))
    bundles = backend.default_bundles(
        num_devices=8,
        accelerator_type_str="TPU-V6E",
    )
    assert len(bundles) == 2
    assert all(b["TPU"] == 4 for b in bundles)

    pg = backend.create_placement_group(
        bundles=bundles,
        strategy="PACK",
        name="serve-tpu-replica",
        accelerator_type_str="TPU-V6E",
    )
    assert pg is fake_handle.placement_group
    assert slice_kwargs[0]["chips_per_vm"] == 4
    assert slice_kwargs[0]["resources_per_bundle"]["TPU"] == 4
    backend.shutdown()


def test_serve_default_bundles_v7x_converts_framework_devices_to_chips():
    """Serve num_devices is TP (framework devices); pack hosts by physical chips."""
    backend = TPUAccelerator(TPUConfig(topology="2x2x1"))
    bundles = backend.default_bundles(
        num_devices=8,  # framework devices on 4 physical chips
        accelerator_type_str="TPU-V7X",
    )
    assert bundles == [
        {"TPU": 4, "accelerator_type:TPU-V7X": 0.001},
    ]


def test_batch_rejects_mixed_tpu_and_non_tpu_bundles(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    backend = TPUAccelerator(TPUConfig(topology="4x4"))
    for bad_bundles in (
        [{"TPU": 1}, {"CPU": 4}],
        [{"CPU": 2, "TPU": 1}, {"CPU": 8, "special": 1}],
    ):
        with pytest.raises(ValueError, match="cannot mix"):
            _tpu_options(
                backend,
                accelerator_type="TPU-V6E",
                tensor_parallel_size=16,
                executor_backend="ray",
                placement_group_config={"bundles": bad_bundles},
                concurrency=1,
            )


def test_batch_accepts_homogeneous_tpu_and_omit_tpu_bundles(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    slice_kwargs = []
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="4x4", num_hosts=4, chips_per_host=4
    )
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        on_slice=lambda *args, **kwargs: slice_kwargs.append(kwargs),
    )
    backend = TPUAccelerator(TPUConfig(topology="4x4"))

    kwargs, close_fn = _tpu_options(
        backend,
        accelerator_type="TPU-V6E",
        tensor_parallel_size=16,
        executor_backend="ray",
        placement_group_config={
            "bundles": [{"CPU": 2, "TPU": 1} for _ in range(16)],
        },
        concurrency=1,
    )
    assert slice_kwargs[-1]["resources_per_bundle"]["TPU"] == 1
    assert slice_kwargs[-1]["resources_per_bundle"]["CPU"] == float(
        PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST
    )
    close_fn()

    kwargs, close_fn = _tpu_options(
        backend,
        accelerator_type="TPU-V6E",
        tensor_parallel_size=16,
        executor_backend="ray",
        placement_group_config={
            "bundles": [{"CPU": 4, "special": 1} for _ in range(16)],
        },
        concurrency=1,
    )
    assert slice_kwargs[-1]["resources_per_bundle"] == {
        "CPU": 4.0,
        "special": 1,
        "TPU": 1,
    }
    close_fn()


def test_v7x_2x2x1_per_chip_uses_strict_pack(monkeypatch):
    """Single-VM Ironwood + TPU1 hits devices_per_chip and STRICT_PACK together."""
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    slice_kwargs = []
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="2x2x1", num_hosts=1, chips_per_host=4, slice_name=None
    )
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        on_slice=lambda *args, **kwargs: slice_kwargs.append(kwargs),
    )
    kwargs, close_fn = _tpu_options(
        TPUAccelerator(TPUConfig(topology="2x2x1")),
        accelerator_type="TPU-V7X",
        tensor_parallel_size=8,
        executor_backend="ray",
        placement_group_config={"bundle_per_worker": {"TPU": 1}},
        concurrency=1,
    )
    assert fake_handle.num_hosts == 1
    assert fake_handle.num_bundles == 4
    assert slice_kwargs[-1]["strategy"] == "STRICT_PACK"
    selectors = slice_kwargs[-1]["bundle_label_selector"]
    assert selectors is not None
    assert len(selectors) == 4
    assert all(
        s
        == {
            "ray.io/tpu-topology": "2x2x1",
            "ray.io/tpu-pod-type": "v7x-8",
        }
        for s in selectors
    )
    close_fn()


def test_v6e_2x4_chips_per_vm_four_is_multi_vm_layout(monkeypatch):
    """chips_per_vm=4 makes 2x4 a 2-VM slice with slice-name gang scheduling."""
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    slice_kwargs = []
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="2x4", num_hosts=2, chips_per_host=4, slice_name="tpu-slice-0"
    )
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        on_slice=lambda *args, **kwargs: slice_kwargs.append(kwargs),
    )

    kwargs, close_fn = _tpu_options(
        TPUAccelerator(TPUConfig(topology="2x4", chips_per_vm=4)),
        accelerator_type="TPU-V6E",
        tensor_parallel_size=8,
        executor_backend="ray",
        concurrency=1,
    )

    assert len(slice_kwargs) == 1
    assert slice_kwargs[0]["chips_per_vm"] == 4
    assert fake_handle.num_hosts == 2
    assert fake_handle.num_bundles == 2
    assert fake_handle.chips_per_host == 4
    for bundle in fake_handle.placement_group.bundle_specs:
        assert bundle["TPU"] == 4.0
    # Multi-VM path: SlicePG injects slice-name; Batch must not set single-VM selectors.
    assert slice_kwargs[0]["bundle_label_selector"] is None
    assert slice_kwargs[0]["strategy"] == "PACK"
    close_fn()


def test_v6e_2x4_chips_per_vm_four_per_chip_granularity(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    slice_kwargs = []
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="2x4", num_hosts=2, chips_per_host=4, slice_name="tpu-slice-0"
    )
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        on_slice=lambda *args, **kwargs: slice_kwargs.append(kwargs),
    )

    kwargs, close_fn = _tpu_options(
        TPUAccelerator(TPUConfig(topology="2x4", chips_per_vm=4)),
        accelerator_type="TPU-V6E",
        tensor_parallel_size=8,
        executor_backend="ray",
        placement_group_config={"bundle_per_worker": {"TPU": 1}},
        concurrency=1,
    )

    assert fake_handle.num_hosts == 2
    assert fake_handle.num_bundles == 8
    assert slice_kwargs[0]["chips_per_vm"] == 4
    assert slice_kwargs[0]["strategy"] == "PACK"
    assert slice_kwargs[0]["bundle_label_selector"] is None
    for bundle in fake_handle.placement_group.bundle_specs:
        assert bundle["TPU"] == 1.0
    close_fn()


def test_processor_config_accepts_chips_per_vm_dict():
    cfg = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config={
            "kind": "tpu",
            "topology": "2x4",
            "chips_per_vm": 4,
        },
        concurrency=1,
        engine_kwargs={"tensor_parallel_size": 8},
    )
    assert isinstance(cfg.accelerator_config, TPUConfig)
    assert cfg.accelerator_config.chips_per_vm == 4


def test_single_vm_v4_pod_type_uses_cores_not_chips(monkeypatch):
    """Pod type is cores-based; v4 has two cores per chip (2x2x1 → v4-8)."""
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    slice_kwargs = []
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="2x2x1", num_hosts=1, chips_per_host=4, slice_name=None
    )
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        on_slice=lambda *args, **kwargs: slice_kwargs.append(kwargs),
    )

    kwargs, close_fn = _tpu_options(
        TPUAccelerator(TPUConfig(topology="2x2x1")),
        accelerator_type="TPU-V4",
        tensor_parallel_size=4,
        executor_backend="ray",
        concurrency=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )

    assert slice_kwargs[0]["topology"] == "2x2x1"
    assert slice_kwargs[0]["accelerator_version"] == "v4"
    assert slice_kwargs[0]["bundle_label_selector"] == [
        {
            "ray.io/tpu-topology": "2x2x1",
            "ray.io/tpu-pod-type": "v4-8",
        }
    ]
    close_fn()


def test_slice_allocation_kwargs_and_head_release_ordering(monkeypatch):
    """Reserve with fixed kwargs, wait, release head markers, then validate hosts."""
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    call_order = []
    slice_kwargs = []
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="4x4", num_hosts=4, chips_per_host=4
    )

    original_release = fake_handle.release_head_pgs

    def tracking_release():
        call_order.append("release_head")
        original_release()

    fake_handle.release_head_pgs = tracking_release

    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        on_slice=lambda *args, **kwargs: (
            call_order.append("slice_pg"),
            slice_kwargs.append(kwargs),
        ),
        on_wait=lambda pg, timeout_s: call_order.append(("wait", timeout_s)),
    )

    kwargs, close_fn = _tpu_options(
        TPUAccelerator(TPUConfig(topology="4x4")),
        accelerator_type="TPU-V6E",
        tensor_parallel_size=16,
        executor_backend="ray",
        concurrency=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )

    assert len(slice_kwargs) == 1
    assert slice_kwargs[0]["topology"] == "4x4"
    assert slice_kwargs[0]["accelerator_version"] == "v6e"
    assert slice_kwargs[0]["resources_per_bundle"] == {
        "CPU": float(PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST),
    }
    assert slice_kwargs[0]["bundle_label_selector"] is None
    assert slice_kwargs[0]["strategy"] == "PACK"
    assert slice_kwargs[0]["tpu_resource_per_chip"] == 1
    assert "chips_per_vm" not in slice_kwargs[0]
    assert call_order == [
        "slice_pg",
        ("wait", DEFAULT_PG_READY_TIMEOUT_S),
        "release_head",
    ]
    assert fake_handle.released_head_pgs == 1
    assert fake_handle.shutdown_calls == 0
    close_fn()
    assert fake_handle.shutdown_calls == 1


def test_runtime_env_merge_matrix(monkeypatch):
    """Required TPU env vars are forced; unrelated user vars are preserved."""
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    fake_handle = FakeSlicePlacementGroupHandle()
    _install_tpu_slice_fakes(monkeypatch, fake_handle)
    backend = TPUAccelerator(TPUConfig(topology="4x4"))

    kwargs, close_fn = _tpu_options(
        backend,
        accelerator_type="TPU-V6E",
        tensor_parallel_size=16,
        executor_backend="ray",
        runtime_env={
            "pip": ["numpy"],
            "env_vars": {
                "USER_VAR": "keep-me",
                "TPU_MULTIHOST_BACKEND": "ray",
                "RAY_TPU_RESOURCE_PER_CHIP": "1",
            },
        },
        concurrency=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )
    env_vars = kwargs["runtime_env"]["env_vars"]
    assert env_vars["USER_VAR"] == "keep-me"
    assert env_vars == {
        "USER_VAR": "keep-me",
        **TPU_ENGINE_ENV_VARS,
    }
    assert kwargs["runtime_env"]["pip"] == ["numpy"]

    with pytest.raises(ValueError, match="runtime_env\\['env_vars'\\] must be"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            runtime_env={"env_vars": ["not-a-dict"]},
            concurrency=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        )


def test_slice_ready_timeout_is_configurable(monkeypatch):
    monkeypatch.delenv(SLICE_READY_TIMEOUT_ENV_VAR, raising=False)
    assert _slice_ready_timeout_s() == DEFAULT_PG_READY_TIMEOUT_S

    monkeypatch.setenv(SLICE_READY_TIMEOUT_ENV_VAR, "12.5")
    assert _slice_ready_timeout_s() == 12.5

    monkeypatch.setenv(SLICE_READY_TIMEOUT_ENV_VAR, "0")
    with pytest.raises(ValueError, match="must be a finite positive number"):
        _slice_ready_timeout_s()

    monkeypatch.setenv(SLICE_READY_TIMEOUT_ENV_VAR, "-1")
    with pytest.raises(ValueError, match="must be a finite positive number"):
        _slice_ready_timeout_s()

    monkeypatch.setenv(SLICE_READY_TIMEOUT_ENV_VAR, "nan")
    with pytest.raises(ValueError, match="must be a finite positive number"):
        _slice_ready_timeout_s()

    monkeypatch.setenv(SLICE_READY_TIMEOUT_ENV_VAR, "inf")
    with pytest.raises(ValueError, match="must be a finite positive number"):
        _slice_ready_timeout_s()

    monkeypatch.setenv(SLICE_READY_TIMEOUT_ENV_VAR, "not-a-number")
    with pytest.raises(ValueError, match="must be a number of seconds"):
        _slice_ready_timeout_s()

    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    monkeypatch.setenv(SLICE_READY_TIMEOUT_ENV_VAR, "7")
    fake_handle = FakeSlicePlacementGroupHandle()

    def failing_timeout(pg, timeout_s):
        assert timeout_s == 7.0
        raise ray.exceptions.GetTimeoutError("timeout")

    _install_tpu_slice_fakes(monkeypatch, fake_handle, on_wait=failing_timeout)
    with pytest.raises(TimeoutError, match="Timed out after 7"):
        _tpu_options(
            TPUAccelerator(TPUConfig(topology="4x4")),
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            concurrency=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        )
    assert fake_handle.shutdown_calls == 1
    assert fake_handle.released_head_pgs == 0


def test_ordering_and_cleanup_on_failures(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    backend = TPUAccelerator(TPUConfig(topology="4x4"))

    # Timeout case -> translates to TimeoutError and cleans up exactly once
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="4x4", num_hosts=4, chips_per_host=4
    )

    def failing_timeout(pg, timeout_s):
        raise ray.exceptions.GetTimeoutError("timeout")

    _install_tpu_slice_fakes(monkeypatch, fake_handle, on_wait=failing_timeout)
    with pytest.raises(TimeoutError, match="Timed out after"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            concurrency=1,
        )
    assert fake_handle.shutdown_calls == 1
    assert fake_handle.released_head_pgs == 0

    # Non-timeout error in wait -> preserves original error type
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="4x4", num_hosts=4, chips_per_host=4
    )

    def failing_runtime(pg, timeout_s):
        raise RuntimeError("GCS crash")

    _install_tpu_slice_fakes(monkeypatch, fake_handle, on_wait=failing_runtime)
    with pytest.raises(RuntimeError, match="GCS crash"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            concurrency=1,
        )
    assert fake_handle.shutdown_calls == 1


def test_managed_processor_lifecycle_no_finalizer():
    fake_handle = FakeSlicePlacementGroupHandle()
    close_calls = {"n": 0}

    def close_fn():
        close_calls["n"] += 1
        fake_handle.shutdown()

    proc_config = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config=TPUConfig(topology="4x4"),
    )
    managed_proc = _ManagedVLLMProcessor(
        config=proc_config,
        stages=[],
        close_fn=close_fn,
    )

    # Context manager usage
    with managed_proc as p:
        assert p._closed is False

    assert managed_proc._closed is True
    assert close_calls["n"] == 1
    assert fake_handle.shutdown_calls == 1

    # Idempotent close
    managed_proc.close()
    assert close_calls["n"] == 1

    # Closed processor rejects execution
    class _ClosedDataset:
        pass

    with pytest.raises(RuntimeError, match="Processor is closed"):
        managed_proc(_ClosedDataset())  # type: ignore[arg-type]

    # Deleting processor does NOT trigger close (no finalizer)
    close_calls_del = {"n": 0}

    def close_fn_del():
        close_calls_del["n"] += 1

    proc_to_del = _ManagedVLLMProcessor(
        config=proc_config,
        stages=[],
        close_fn=close_fn_del,
    )
    del proc_to_del
    gc.collect()
    assert close_calls_del["n"] == 0


def test_managed_processor_close_exception_retains_callable():
    """If close_fn raises, close keeps the callable for another try."""
    first_attempt = True
    calls = {"n": 0}

    def failing_close():
        nonlocal first_attempt
        calls["n"] += 1
        if first_attempt:
            first_attempt = False
            raise ConnectionError("Transient network failure")

    proc_config = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config=TPUConfig(topology="4x4"),
    )
    proc = _ManagedVLLMProcessor(
        config=proc_config,
        stages=[],
        close_fn=failing_close,
    )

    with pytest.raises(ConnectionError, match="Transient network failure"):
        proc.close()
    assert proc._close_fn is not None

    proc.close()
    assert proc._close_fn is None
    assert calls["n"] == 2


def test_production_kwargs_round_trips_through_pickle(mock_tpu_slice_environment):
    """map_batches kwargs must survive embedding in the lazy Dataset graph.

    The driver-owned close callable stays on the processor, not in the kwargs.
    """
    kwargs, close_fn = _tpu_options(
        TPUAccelerator(TPUConfig(topology="4x4")),
        accelerator_type="TPU-V6E",
        tensor_parallel_size=16,
        executor_backend="ray",
        concurrency=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )

    strategy = kwargs.pop("scheduling_strategy")
    assert isinstance(strategy, PlacementGroupSchedulingStrategy)
    assert strategy.placement_group_bundle_index == 0
    assert strategy.placement_group_capture_child_tasks is True

    # Pickle the plain DAG-safe fields (strategy holds a live PG handle).
    loaded = pickle.loads(pickle.dumps(kwargs))
    assert loaded["num_cpus"] == PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST
    assert loaded["num_gpus"] == 0
    assert loaded["resources"] == {}
    assert loaded["runtime_env"]["env_vars"] == {
        "TPU_MULTIHOST_BACKEND": "ray",
        "RAY_TPU_RESOURCE_PER_CHIP": "1",
    }
    assert callable(close_fn)
    close_fn()


def test_fresh_interpreter_imports():
    """Data and Serve must both reach the shared accelerator module cleanly."""
    cmd = [
        sys.executable,
        "-c",
        "import ray.data.llm; import ray.llm._internal.serve.core.configs.llm_config",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Import failed in fresh subprocess: {res.stderr}"


def test_neutral_backend_factory():
    """Factory dispatches on a fully resolved typed config only."""
    assert isinstance(
        get_accelerator_backend(TPUConfig(topology="4x4")),
        TPUAccelerator,
    )
    assert isinstance(get_accelerator_backend(GPUConfig()), GPUAccelerator)
    assert isinstance(get_accelerator_backend(CPUConfig()), CPUAccelerator)

    with pytest.raises(TypeError, match="Unsupported accelerator config"):
        get_accelerator_backend(object())  # type: ignore[arg-type]


def test_serve_shares_the_accelerator_definitions():
    """Serve and Data must resolve the same accelerator classes, not parallel copies."""
    from ray.llm._internal.serve.core.configs import llm_config as serve_llm_config
    from ray.llm._internal.serve.engines.vllm import vllm_models

    assert serve_llm_config.TPUConfig is TPUConfig
    assert serve_llm_config.AnyAcceleratorConfig is AnyAcceleratorConfig
    assert serve_llm_config.TPU_ACCELERATOR_VALUES is TPU_ACCELERATOR_VALUES
    assert vllm_models.TPUAccelerator is TPUAccelerator
    assert vllm_models.AcceleratorBackend is AcceleratorBackend


# -------------------------------------------------------------------------
# GPU backend plan boundary tests
# -------------------------------------------------------------------------


def test_gpu_build_batch_scheduling_plan_uni():
    """Single-replica GPU plans pin GPUs directly and never reuse a driver PG."""
    kwargs, close_fn = _gpu_options(
        accelerator_type="A100",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        executor_backend="uni",
        runtime_env={"env_vars": {"USER_VAR": "gpu"}},
        concurrency=1,
    )

    assert close_fn is None
    kwargs = kwargs
    assert kwargs["accelerator_type"] == "A100"
    assert kwargs["num_gpus"] == 1
    assert "ray_remote_args_fn" not in kwargs
    assert kwargs["runtime_env"]["env_vars"]["USER_VAR"] == "gpu"


def test_gpu_build_batch_scheduling_plan_ray_executor(monkeypatch):
    """Multi-bundle GPU plans defer PG creation and inject accelerator tokens (F-8)."""
    captured = {}

    def fake_create(
        self,
        *,
        bundles,
        strategy,
        name,
        accelerator_type_str=None,
    ):
        captured["bundles"] = [dict(b) for b in bundles]
        captured["strategy"] = strategy
        return FakePlacementGroup()

    monkeypatch.setattr(GPUAccelerator, "create_placement_group", fake_create)

    kwargs, close_fn = _gpu_options(
        accelerator_type="A100",
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        executor_backend="ray",
        concurrency=1,
    )

    assert close_fn is None
    kwargs = kwargs
    assert kwargs["accelerator_type"] == "A100"
    assert kwargs["num_gpus"] == 0
    ray_remote_args_fn = kwargs["ray_remote_args_fn"]
    assert callable(ray_remote_args_fn)

    remote_args = ray_remote_args_fn()
    strategy = remote_args["scheduling_strategy"]
    assert isinstance(strategy, PlacementGroupSchedulingStrategy)
    assert strategy.placement_group_capture_child_tasks is True
    assert captured["strategy"] == "PACK"
    assert len(captured["bundles"]) == 4
    for bundle in captured["bundles"]:
        assert bundle["GPU"] == 1
        assert bundle["CPU"] == 1
        assert bundle["accelerator_type:A100"] == 0.001


def test_gpu_build_batch_scheduling_plan_custom_placement_group_uni():
    """Custom GPU placement bundles collapse into actor resource requests for uni."""
    kwargs, close_fn = _gpu_options(
        accelerator_type="A100",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        executor_backend="uni",
        placement_group_config={
            "bundle_per_worker": {"GPU": 1, "CPU": 2, "custom": 1},
        },
        concurrency=1,
    )

    kwargs = kwargs
    assert kwargs["num_gpus"] == 2
    assert kwargs["num_cpus"] == 4
    assert kwargs["resources"] == {"custom": 2}
    assert "ray_remote_args_fn" not in kwargs


def test_gpu_ray_callback_custom_bundles_inject_accelerator_token(monkeypatch):
    """Custom GPU bundles also receive accelerator_type tokens when the callback runs."""
    captured = {}

    def fake_create(
        self,
        *,
        bundles,
        strategy,
        name,
        accelerator_type_str=None,
    ):
        captured["bundles"] = [dict(b) for b in bundles]
        captured["strategy"] = strategy
        return FakePlacementGroup()

    monkeypatch.setattr(GPUAccelerator, "create_placement_group", fake_create)

    kwargs, close_fn = _gpu_options(
        accelerator_type="A100",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        executor_backend="ray",
        placement_group_config={
            "bundle_per_worker": {"GPU": 1, "CPU": 2},
            "strategy": "STRICT_PACK",
        },
        concurrency=1,
    )

    remote_args = kwargs["ray_remote_args_fn"]()
    assert isinstance(
        remote_args["scheduling_strategy"], PlacementGroupSchedulingStrategy
    )
    assert captured["strategy"] == "STRICT_PACK"
    assert len(captured["bundles"]) == 2
    for bundle in captured["bundles"]:
        assert bundle["GPU"] == 1
        assert bundle["CPU"] == 2
        assert bundle["accelerator_type:A100"] == 0.001


def test_gpu_ray_callback_treats_null_bundles_as_empty(monkeypatch):
    """bundles: None must not raise when the callback materializes the PG."""
    captured = {}

    def fake_create(
        self,
        *,
        bundles,
        strategy,
        name,
        accelerator_type_str=None,
    ):
        captured["bundles"] = list(bundles)
        return FakePlacementGroup()

    monkeypatch.setattr(GPUAccelerator, "create_placement_group", fake_create)

    kwargs, close_fn = _gpu_options(
        accelerator_type="A100",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        executor_backend="ray",
        placement_group_config={"bundles": None, "strategy": "PACK"},
        concurrency=1,
    )
    kwargs["ray_remote_args_fn"]()
    assert captured["bundles"] == []


def test_tpu_batch_rejects_strategy_only_placement_group_config():
    """Internal direct callers passing only strategy must fail closed."""
    backend = TPUAccelerator(TPUConfig(topology="4x4"))
    with pytest.raises(
        ValueError,
        match="placement_group_config must specify bundle_per_worker or bundles",
    ):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="ray",
            placement_group_config={"strategy": "PACK"},
            concurrency=1,
        )


def test_resolve_topology_worker_bundle_preserves_custom_resources_on_tpu_fallback():
    """Shared _resolve_topology_worker_bundle helper preserves CPU/custom and adds TPU:1."""
    backend = TPUAccelerator()

    # Empty bundles list -> TPU: 1
    assert backend._resolve_topology_worker_bundle([]) == {"TPU": 1}

    # Positive TPU bundle -> preserves bundle
    assert backend._resolve_topology_worker_bundle([{"CPU": 2, "TPU": 1}]) == {
        "CPU": 2,
        "TPU": 1,
    }

    # No TPU present -> preserves CPU/custom resources and adds TPU: 1
    assert backend._resolve_topology_worker_bundle(
        [{"CPU": 4, "accelerator_type:TPU-V6E": 0.001}]
    ) == {
        "CPU": 4,
        "accelerator_type:TPU-V6E": 0.001,
        "TPU": 1,
    }

    # Heterogeneous no-TPU bundles -> raises ValueError
    with pytest.raises(ValueError, match="Heterogeneous"):
        backend._resolve_topology_worker_bundle([{"CPU": 2}, {"CPU": 4}])
