"""Comprehensive hermetic unit tests for TPU batch scheduling and lifecycle.

All tests run on CPU CI without TPU hardware or placement group engines.
"""

import gc
import logging
import pickle
import subprocess
import sys
from typing import Any, Dict, List, Optional

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
from ray.llm._internal.batch.stages.vllm_engine_stage import vLLMEngineStageUDF
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
    BatchSchedulingRequest,
    CPUAccelerator,
    CPUConfig,
    GPUAccelerator,
    GPUConfig,
    TPUAccelerator,
    TPUConfig,
    _slice_ready_timeout_s,
    get_accelerator_backend,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# -------------------------------------------------------------------------
# Test Fakes and Helpers
# -------------------------------------------------------------------------

_ACCEL = "ray.llm._internal.common.accelerators"


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


def _create_mock_v6e_nodes(
    num_hosts: int = 4, tpu_per_host: int = 4, slice_name: str = "tpu-slice-0"
) -> List[Dict[str, Any]]:
    nodes = []
    for i in range(num_hosts):
        nodes.append(
            {
                "NodeID": f"node-tpu-{i}",
                "Alive": True,
                "Labels": {"ray.io/tpu-slice-name": slice_name},
                "Resources": {"CPU": 32.0, "TPU": float(tpu_per_host)},
            }
        )
    return nodes


def _install_tpu_slice_fakes(
    monkeypatch,
    fake_handle: FakeSlicePlacementGroupHandle,
    *,
    nodes: Optional[List[Dict[str, Any]]] = None,
    on_slice=None,
    on_wait=None,
    on_nodes=None,
):
    """Patch the public TPU seams used by ``TPUAccelerator.build_batch_scheduling_plan``."""
    from ray._private.accelerators.tpu import (
        get_chips_per_host,
        get_num_chips_from_topology,
    )
    from ray.util.tpu import get_tpu_worker_resources

    selected_nodes = nodes

    def _slice(*args, **kwargs):
        if on_slice is not None:
            on_slice(*args, **kwargs)
        # Keep the fake handle consistent with Ray's SlicePG arithmetic so
        # reserved-layout validation sees the execution bundle count.
        topology = kwargs.get("topology", fake_handle.topology)
        version = kwargs.get("accelerator_version", "v6e")
        resources = kwargs.get("resources_per_bundle") or {}
        tpu_rpc = kwargs.get("tpu_resource_per_chip") or 1
        num_bundles, bundle_resources = get_tpu_worker_resources(
            topology=topology,
            accelerator_type=f"TPU-{version.upper()}",
            resources_per_worker=resources,
            tpu_resource_per_chip=tpu_rpc,
        )
        chips_per_vm = get_chips_per_host(topology, version)
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

    def _nodes(slice_name, nodes=None):
        if on_nodes is not None:
            return on_nodes(slice_name, nodes)
        if selected_nodes is not None:
            return selected_nodes
        # Prefer currently configured host count / chips after slice creation.
        return _create_mock_v6e_nodes(
            num_hosts=fake_handle.num_hosts, tpu_per_host=fake_handle.chips_per_host
        )

    monkeypatch.setattr(f"{_ACCEL}.slice_placement_group", _slice)
    monkeypatch.setattr(f"{_ACCEL}._wait_for_placement_group", _wait)
    monkeypatch.setattr(f"{_ACCEL}.get_tpu_nodes_for_slice", _nodes)
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
        "bundle_per_worker": {"CPU": 0.0, "GPU": 0.0, "TPU": 1},
        "bundles": None,
        "strategy": "PACK",
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
    assert isinstance(processor._close_handle.backend, TPUAccelerator)
    assert processor._close_handle.backend._slice_pg_wrapper is fake_handle

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
    # vLLM from_engine_args can publish the current PG into ParallelConfig;
    # required_env_vars fail fast if the TPU runtime env did not propagate.
    fn_kwargs = vllm_stage.fn_constructor_kwargs
    assert "reuse_current_placement_group" not in fn_kwargs
    assert fn_kwargs["required_env_vars"] == {
        "TPU_MULTIHOST_BACKEND": "ray",
        "RAY_TPU_RESOURCE_PER_CHIP": "1",
    }

    close_backend = processor._close_handle.backend
    processor.close()
    assert fake_handle.shutdown_calls == 1
    assert close_backend._slice_pg_wrapper is None
    assert processor._close_handle is None


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
        "Failed to release TPU batch resources after processor construction failed"
        in r.getMessage()
        for r in log_records
    )


# -------------------------------------------------------------------------
# Direct backend validation and ordering tests
# -------------------------------------------------------------------------


def test_env_var_constant():
    assert RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR == "RAY_TPU_RESOURCE_PER_CHIP"


def test_direct_backend_validation_failures(monkeypatch):
    backend = TPUAccelerator()

    # 1. Missing topology
    with pytest.raises(
        ValueError, match="TPU batch inference requires an explicit `accelerator_config"
    ):
        backend.build_batch_scheduling_plan(
            BatchSchedulingRequest(
                accelerator_type="TPU-V6E",
                accelerator_config=TPUConfig(),
                tensor_parallel_size=16,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                executor_backend="ray",
                placement_group_config=None,
                runtime_env=None,
                concurrency=1,
            )
        )

    # 2. Missing accelerator_type
    with pytest.raises(ValueError, match="`accelerator_type`.*is required"):
        backend.build_batch_scheduling_plan(
            BatchSchedulingRequest(
                accelerator_type=None,
                accelerator_config=TPUConfig(topology="4x4"),
                tensor_parallel_size=16,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                executor_backend="ray",
                placement_group_config=None,
                runtime_env=None,
                concurrency=1,
            )
        )

    # 3. TP mismatch (15 != 16)
    with pytest.raises(ValueError, match="tensor_parallel_size must match"):
        backend.build_batch_scheduling_plan(
            BatchSchedulingRequest(
                accelerator_type="TPU-V6E",
                accelerator_config=TPUConfig(topology="4x4"),
                tensor_parallel_size=15,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                executor_backend="ray",
                placement_group_config=None,
                runtime_env=None,
                concurrency=1,
            )
        )

    # 4. PP != 1
    with pytest.raises(ValueError, match="pipeline_parallel_size=1"):
        backend.build_batch_scheduling_plan(
            BatchSchedulingRequest(
                accelerator_type="TPU-V6E",
                accelerator_config=TPUConfig(topology="4x4"),
                tensor_parallel_size=16,
                pipeline_parallel_size=2,
                data_parallel_size=1,
                executor_backend="ray",
                placement_group_config=None,
                runtime_env=None,
                concurrency=1,
            )
        )

    # 4b. PP True / 1.0 must not silently pass the == 1 check
    for bad_pp in (True, 1.0):
        with pytest.raises(
            ValueError, match="pipeline_parallel_size must be a positive integer"
        ):
            backend.build_batch_scheduling_plan(
                BatchSchedulingRequest(
                    accelerator_type="TPU-V6E",
                    accelerator_config=TPUConfig(topology="4x4"),
                    tensor_parallel_size=16,
                    pipeline_parallel_size=bad_pp,
                    data_parallel_size=1,
                    executor_backend="ray",
                    concurrency=1,
                )
            )

    # 5. DP != 1
    with pytest.raises(ValueError, match="data_parallel_size=1"):
        backend.build_batch_scheduling_plan(
            BatchSchedulingRequest(
                accelerator_type="TPU-V6E",
                accelerator_config=TPUConfig(topology="4x4"),
                tensor_parallel_size=16,
                pipeline_parallel_size=1,
                data_parallel_size=2,
                executor_backend="ray",
                placement_group_config=None,
                runtime_env=None,
                concurrency=1,
            )
        )

    # 5b. DP True / 1.0 must not silently pass the == 1 check
    for bad_dp in (True, 1.0):
        with pytest.raises(
            ValueError, match="data_parallel_size must be a positive integer"
        ):
            backend.build_batch_scheduling_plan(
                BatchSchedulingRequest(
                    accelerator_type="TPU-V6E",
                    accelerator_config=TPUConfig(topology="4x4"),
                    tensor_parallel_size=16,
                    pipeline_parallel_size=1,
                    data_parallel_size=bad_dp,
                    executor_backend="ray",
                    concurrency=1,
                )
            )

    # 6. Concurrency bool True
    with pytest.raises(ValueError, match="concurrency=1"):
        backend.build_batch_scheduling_plan(
            BatchSchedulingRequest(
                accelerator_type="TPU-V6E",
                accelerator_config=TPUConfig(topology="4x4"),
                tensor_parallel_size=16,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                executor_backend="ray",
                placement_group_config=None,
                runtime_env=None,
                concurrency=True,
            )
        )

    # 7. Concurrency float 1.0
    with pytest.raises(ValueError, match="concurrency=1"):
        backend.build_batch_scheduling_plan(
            BatchSchedulingRequest(
                accelerator_type="TPU-V6E",
                accelerator_config=TPUConfig(topology="4x4"),
                tensor_parallel_size=16,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                executor_backend="ray",
                placement_group_config=None,
                runtime_env=None,
                concurrency=1.0,
            )
        )

    # 8. Invalid TPU-per-bundle granularity (does not divide chips/VM)
    with pytest.raises(ValueError, match="evenly divide"):
        backend.build_batch_scheduling_plan(
            BatchSchedulingRequest(
                accelerator_type="TPU-V6E",
                accelerator_config=TPUConfig(topology="4x4"),
                tensor_parallel_size=16,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                executor_backend="ray",
                placement_group_config={"bundle_per_worker": {"TPU": 3}},
                runtime_env=None,
                concurrency=1,
            )
        )

    # 9. Runtime env backend conflict
    with pytest.raises(ValueError, match="TPU_MULTIHOST_BACKEND"):
        backend.build_batch_scheduling_plan(
            BatchSchedulingRequest(
                accelerator_type="TPU-V6E",
                accelerator_config=TPUConfig(topology="4x4"),
                tensor_parallel_size=16,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                executor_backend="ray",
                placement_group_config=None,
                runtime_env={"env_vars": {"TPU_MULTIHOST_BACKEND": "grpc"}},
                concurrency=1,
            )
        )

    # 10. Runtime env resource-per-chip mismatch (integer 1 or non-string)
    with pytest.raises(ValueError, match="must be the string '1'"):
        backend.build_batch_scheduling_plan(
            BatchSchedulingRequest(
                accelerator_type="TPU-V6E",
                accelerator_config=TPUConfig(topology="4x4"),
                tensor_parallel_size=16,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                executor_backend="ray",
                placement_group_config=None,
                runtime_env={"env_vars": {"RAY_TPU_RESOURCE_PER_CHIP": 1}},
                concurrency=1,
            )
        )

    # 11. Driver resource per chip mismatch
    monkeypatch.setenv("RAY_TPU_RESOURCE_PER_CHIP", "2")
    with pytest.raises(ValueError, match="requires RAY_TPU_RESOURCE_PER_CHIP == 1"):
        backend.build_batch_scheduling_plan(
            BatchSchedulingRequest(
                accelerator_type="TPU-V6E",
                accelerator_config=TPUConfig(topology="4x4"),
                tensor_parallel_size=16,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                executor_backend="ray",
                concurrency=1,
            )
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
    node_lookups = []
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
        on_nodes=lambda slice_name, nodes=None: (
            node_lookups.append(slice_name),
            _create_mock_v6e_nodes(num_hosts=num_vms, tpu_per_host=chips_per_vm),
        )[1],
    )

    acquired = TPUAccelerator().build_batch_scheduling_plan(
        BatchSchedulingRequest(
            accelerator_type=accelerator_type,
            accelerator_config=TPUConfig(topology=topology),
            tensor_parallel_size=total_chips,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            executor_backend="ray",
            concurrency=1,
        )
    )

    kwargs = acquired.plan.map_batches_kwargs
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
        assert node_lookups == []
        # Single-VM SlicePGs skip slice-name reservation; pin generation/topology.
        expected_pod_type = infer_tpu_pod_type_from_topology(topology, accelerator_type)
        assert slice_kwargs[0]["bundle_label_selector"] == [
            {
                "ray.io/tpu-topology": topology,
                "ray.io/tpu-pod-type": expected_pod_type,
            }
        ]
    else:
        assert node_lookups == ["tpu-slice-0"]
        # Multi-VM path must not inject caller hardware labels.
        assert slice_kwargs[0]["bundle_label_selector"] is None
    acquired.close_handle.shutdown()


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

    acquired = TPUAccelerator(TPUConfig(topology=topology)).build_batch_scheduling_plan(
        BatchSchedulingRequest(
            accelerator_type="TPU-V6E",
            accelerator_config=TPUConfig(topology=topology),
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            executor_backend="ray",
            concurrency=1,
        )
    )

    assert slice_kwargs == [
        {
            "topology": "2x4",
            "accelerator_version": "v6e",
            "resources_per_bundle": {
                "CPU": PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST,
            },
            "bundle_label_selector": [
                {
                    "ray.io/tpu-topology": "2x4",
                    "ray.io/tpu-pod-type": "v6e-8",
                }
            ],
            "strategy": "SPREAD",
            "tpu_resource_per_chip": 1,
        }
    ]
    acquired.close_handle.shutdown()


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

    acquired = TPUAccelerator(TPUConfig(topology=topology)).build_batch_scheduling_plan(
        BatchSchedulingRequest(
            accelerator_type="TPU-V6E",
            accelerator_config=TPUConfig(topology=topology),
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            executor_backend="ray",
            placement_group_config=pg_config,
            concurrency=1,
        )
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
            "SPREAD" if tpu_per_bundle is None else "PACK"
        )
    else:
        assert slice_kwargs[0]["bundle_label_selector"] is None

    acquired.close_handle.shutdown()


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

    acquired = backend.build_batch_scheduling_plan(
        BatchSchedulingRequest(
            accelerator_type="TPU-V6E",
            accelerator_config=TPUConfig(topology="4x4"),
            tensor_parallel_size=16,
            executor_backend="ray",
            placement_group_config={"bundle_per_worker": {"CPU": 1, "TPU": 1}},
            concurrency=1,
        )
    )
    assert slice_kwargs[-1]["resources_per_bundle"]["CPU"] == float(
        PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST
    )
    assert slice_kwargs[-1]["resources_per_bundle"]["TPU"] == 1
    acquired.close_handle.shutdown()

    with pytest.raises(ValueError, match="GPU resources are not supported"):
        backend.build_batch_scheduling_plan(
            BatchSchedulingRequest(
                accelerator_type="TPU-V6E",
                accelerator_config=TPUConfig(topology="4x4"),
                tensor_parallel_size=16,
                executor_backend="ray",
                placement_group_config={
                    "bundle_per_worker": {"GPU": 1, "TPU": 1},
                },
                concurrency=1,
            )
        )


def test_heterogeneous_tpu_bundles_rejected_for_batch(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    backend = TPUAccelerator(TPUConfig(topology="4x4"))
    with pytest.raises(ValueError, match="Heterogeneous TPU bundles"):
        backend.build_batch_scheduling_plan(
            BatchSchedulingRequest(
                accelerator_type="TPU-V6E",
                accelerator_config=TPUConfig(topology="4x4"),
                tensor_parallel_size=16,
                executor_backend="ray",
                placement_group_config={
                    "bundles": [{"TPU": 1}, {"TPU": 4}],
                },
                concurrency=1,
            )
        )


def test_derive_layout_rejects_whitespace_only_topology():
    with pytest.raises(ValueError, match="TPU topology must be non-empty"):
        TPUAccelerator()._derive_layout("   ", "TPU-V6E")


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

    acquired = TPUAccelerator().build_batch_scheduling_plan(
        BatchSchedulingRequest(
            accelerator_type="TPU-V4",
            accelerator_config=TPUConfig(topology="2x2x1"),
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            executor_backend="ray",
            concurrency=1,
        )
    )

    assert slice_kwargs[0]["topology"] == "2x2x1"
    assert slice_kwargs[0]["accelerator_version"] == "v4"
    assert slice_kwargs[0]["bundle_label_selector"] == [
        {
            "ray.io/tpu-topology": "2x2x1",
            "ray.io/tpu-pod-type": "v4-8",
        }
    ]
    acquired.close_handle.shutdown()


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
        on_nodes=lambda slice_name, nodes=None: (
            call_order.append(("nodes", slice_name)),
            _create_mock_v6e_nodes(num_hosts=4, tpu_per_host=4),
        )[1],
    )

    acquired = TPUAccelerator().build_batch_scheduling_plan(
        BatchSchedulingRequest(
            accelerator_type="TPU-V6E",
            accelerator_config=TPUConfig(topology="4x4"),
            tensor_parallel_size=16,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            executor_backend="ray",
            concurrency=1,
        )
    )

    assert slice_kwargs == [
        {
            "topology": "4x4",
            "accelerator_version": "v6e",
            "resources_per_bundle": {
                "CPU": PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST,
            },
            "bundle_label_selector": None,
            "strategy": "SPREAD",
            "tpu_resource_per_chip": 1,
        }
    ]
    assert call_order == [
        "slice_pg",
        ("wait", DEFAULT_PG_READY_TIMEOUT_S),
        "release_head",
        ("nodes", "tpu-slice-0"),
    ]
    assert fake_handle.released_head_pgs == 1
    assert fake_handle.shutdown_calls == 0
    acquired.close_handle.shutdown()
    assert fake_handle.shutdown_calls == 1


def test_runtime_env_merge_matrix(monkeypatch):
    """Required TPU env vars are forced; unrelated user vars are preserved."""
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    fake_handle = FakeSlicePlacementGroupHandle()
    _install_tpu_slice_fakes(monkeypatch, fake_handle)
    backend = TPUAccelerator()

    acquired = backend.build_batch_scheduling_plan(
        BatchSchedulingRequest(
            accelerator_type="TPU-V6E",
            accelerator_config=TPUConfig(topology="4x4"),
            tensor_parallel_size=16,
            pipeline_parallel_size=1,
            data_parallel_size=1,
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
        )
    )
    env_vars = acquired.plan.map_batches_kwargs["runtime_env"]["env_vars"]
    assert env_vars["USER_VAR"] == "keep-me"
    assert env_vars == {
        "USER_VAR": "keep-me",
        **TPU_ENGINE_ENV_VARS,
    }
    assert acquired.plan.required_engine_env_vars == dict(TPU_ENGINE_ENV_VARS)
    assert acquired.plan.map_batches_kwargs["runtime_env"]["pip"] == ["numpy"]

    with pytest.raises(ValueError, match="runtime_env\\['env_vars'\\] must be"):
        backend.build_batch_scheduling_plan(
            BatchSchedulingRequest(
                accelerator_type="TPU-V6E",
                accelerator_config=TPUConfig(topology="4x4"),
                tensor_parallel_size=16,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                executor_backend="ray",
                runtime_env={"env_vars": ["not-a-dict"]},
                concurrency=1,
            )
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
        TPUAccelerator().build_batch_scheduling_plan(
            BatchSchedulingRequest(
                accelerator_type="TPU-V6E",
                accelerator_config=TPUConfig(topology="4x4"),
                tensor_parallel_size=16,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                executor_backend="ray",
                concurrency=1,
            )
        )
    assert fake_handle.shutdown_calls == 1
    assert fake_handle.released_head_pgs == 0


def test_ordering_and_cleanup_on_failures(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    backend = TPUAccelerator()
    request = BatchSchedulingRequest(
        accelerator_type="TPU-V6E",
        accelerator_config=TPUConfig(topology="4x4"),
        tensor_parallel_size=16,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        executor_backend="ray",
        placement_group_config=None,
        runtime_env=None,
        concurrency=1,
    )

    # Timeout case -> translates to TimeoutError and cleans up exactly once
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="4x4", num_hosts=4, chips_per_host=4
    )

    def failing_timeout(pg, timeout_s):
        raise ray.exceptions.GetTimeoutError("timeout")

    _install_tpu_slice_fakes(monkeypatch, fake_handle, on_wait=failing_timeout)
    with pytest.raises(TimeoutError, match="Timed out after"):
        backend.build_batch_scheduling_plan(request)
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
        backend.build_batch_scheduling_plan(request)
    assert fake_handle.shutdown_calls == 1

    # Node count mismatch -> raises RuntimeError and cleans up exactly once
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="4x4", num_hosts=4, chips_per_host=4
    )
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        nodes=_create_mock_v6e_nodes(num_hosts=3, tpu_per_host=4),
    )
    with pytest.raises(RuntimeError, match="contains 3 alive nodes"):
        backend.build_batch_scheduling_plan(request)
    assert fake_handle.shutdown_calls == 1
    assert fake_handle.released_head_pgs == 1

    # Chip count mismatch -> raises RuntimeError and cleans up exactly once
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="4x4", num_hosts=4, chips_per_host=4
    )
    _install_tpu_slice_fakes(
        monkeypatch,
        fake_handle,
        nodes=_create_mock_v6e_nodes(num_hosts=4, tpu_per_host=2),
    )
    with pytest.raises(RuntimeError, match="advertises 2.0 TPU resources"):
        backend.build_batch_scheduling_plan(request)
    assert fake_handle.shutdown_calls == 1


# -------------------------------------------------------------------------
# Base processor and managed processor lifecycle tests
# -------------------------------------------------------------------------


def test_base_processor_lifecycle():
    """Verify base Processor lifecycle methods and context manager support."""
    proc_config = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="A100",
        accelerator_config=GPUConfig(),
    )
    base_proc = Processor(config=proc_config, stages=[])

    # 1. Base close() and shutdown() are safe no-ops
    base_proc.close()
    base_proc.shutdown()

    # 2. Context manager on base processor
    with base_proc as p:
        assert p is base_proc


def test_managed_processor_lifecycle_no_finalizer():
    fake_handle = FakeSlicePlacementGroupHandle()
    proc_config = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config=TPUConfig(topology="4x4"),
    )
    managed_proc = _ManagedVLLMProcessor(
        config=proc_config,
        stages=[],
        close_handle=fake_handle,
    )

    # Context manager usage
    with managed_proc as p:
        assert p._closed is False

    assert managed_proc._closed is True
    assert fake_handle.shutdown_calls == 1

    # Idempotent close
    managed_proc.close()
    assert fake_handle.shutdown_calls == 1

    # Closed processor rejects execution
    ds = ray.data.from_items([{"prompt": "test"}])
    with pytest.raises(RuntimeError, match="Processor is closed"):
        managed_proc(ds)

    # shutdown() dispatches through close() and releases the handle once
    fake_handle_shutdown = FakeSlicePlacementGroupHandle()
    managed_via_shutdown = _ManagedVLLMProcessor(
        config=proc_config,
        stages=[],
        close_handle=fake_handle_shutdown,
    )
    managed_via_shutdown.shutdown()
    assert managed_via_shutdown._closed is True
    assert fake_handle_shutdown.shutdown_calls == 1
    managed_via_shutdown.shutdown()
    assert fake_handle_shutdown.shutdown_calls == 1

    # Deleting processor does NOT trigger handle shutdown (no finalizer)
    fake_handle_del = FakeSlicePlacementGroupHandle()
    proc_to_del = _ManagedVLLMProcessor(
        config=proc_config,
        stages=[],
        close_handle=fake_handle_del,
    )
    del proc_to_del
    gc.collect()
    assert fake_handle_del.shutdown_calls == 0


def test_managed_processor_shutdown_exception_retains_handle():
    """If a custom handle's shutdown raises, close keeps the handle for another try.

    Production SlicePlacementGroup.shutdown() typically logs PG removal failures and
    returns successfully, so this exercises defensive handle retention rather than
    guaranteed retry of Ray core PG removal.
    """
    fake_handle = FakeSlicePlacementGroupHandle()
    proc_config = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config=TPUConfig(topology="4x4"),
    )

    # Simulate shutdown failure on first attempt
    first_attempt = True

    def failing_shutdown():
        nonlocal first_attempt
        if first_attempt:
            first_attempt = False
            raise ConnectionError("Transient network failure")
        fake_handle.shutdown_calls += 1

    fake_handle.shutdown = failing_shutdown

    proc = _ManagedVLLMProcessor(
        config=proc_config,
        stages=[],
        close_handle=fake_handle,
    )

    # First close attempt fails and leaves handle present
    with pytest.raises(ConnectionError, match="Transient network failure"):
        proc.close()
    assert proc._close_handle is not None

    # Second close attempt retries and succeeds
    proc.close()
    assert proc._close_handle is None
    assert fake_handle.shutdown_calls == 1


# -------------------------------------------------------------------------
# UDF runtime diagnostic tests
# -------------------------------------------------------------------------


@pytest.fixture
def recording_engine_wrapper(monkeypatch):
    """Replace the vLLM engine wrapper so UDF construction needs no vLLM install."""
    monkeypatch.setattr(
        "ray.llm._internal.batch.stages.vllm_engine_stage.download_model_files",
        lambda *args, **kwargs: "/tmp/mock-model",
    )
    calls = []

    class DummySchedulerConfig:
        max_num_seqs = 16

    class DummyWrapper:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)
            self.max_pending_requests = 16

        def get_scheduler_config(self):
            return DummySchedulerConfig()

    monkeypatch.setattr(
        "ray.llm._internal.batch.stages.vllm_engine_stage.vLLMEngineWrapper",
        DummyWrapper,
    )
    return calls


def _build_udf(**kwargs):
    return vLLMEngineStageUDF(
        data_column="__data",
        expected_input_keys=["prompt"],
        batch_size=1,
        max_concurrent_batches=1,
        model="test-model",
        engine_kwargs={},
        **kwargs,
    )


def test_udf_rejects_missing_required_env_vars(monkeypatch, recording_engine_wrapper):
    """A required environment variable that did not propagate fails before engine init."""
    monkeypatch.delenv("TPU_MULTIHOST_BACKEND", raising=False)

    with pytest.raises(RuntimeError, match="TPU_MULTIHOST_BACKEND='ray'"):
        _build_udf(required_env_vars={"TPU_MULTIHOST_BACKEND": "ray"})
    assert recording_engine_wrapper == []

    # A wrong value is rejected the same way as a missing one.
    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "grpc")
    with pytest.raises(RuntimeError, match="TPU_MULTIHOST_BACKEND='ray'"):
        _build_udf(required_env_vars={"TPU_MULTIHOST_BACKEND": "ray"})
    assert recording_engine_wrapper == []


def test_udf_accepts_satisfied_required_env_vars(monkeypatch, recording_engine_wrapper):
    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")
    monkeypatch.setenv("RAY_TPU_RESOURCE_PER_CHIP", "1")

    _build_udf(
        required_env_vars={
            "TPU_MULTIHOST_BACKEND": "ray",
            "RAY_TPU_RESOURCE_PER_CHIP": "1",
        },
    )

    assert len(recording_engine_wrapper) == 1
    assert "reuse_current_placement_group" not in recording_engine_wrapper[0]


def test_udf_gpu_path_checks_no_env_vars(monkeypatch, recording_engine_wrapper):
    """The GPU path carries no requirements and must not inspect the environment."""
    monkeypatch.delenv("TPU_MULTIHOST_BACKEND", raising=False)
    monkeypatch.delenv("RAY_TPU_RESOURCE_PER_CHIP", raising=False)

    _build_udf(required_env_vars=None)

    assert len(recording_engine_wrapper) == 1
    assert "reuse_current_placement_group" not in recording_engine_wrapper[0]


# -------------------------------------------------------------------------
# Transport, factory, and compatibility export tests
# -------------------------------------------------------------------------


def test_production_plan_round_trips_through_pickle(mock_tpu_slice_environment):
    """The real backend plan must survive embedding in the lazy Dataset graph.

    Only ``AcquiredBatchResources.plan`` is serialized. The driver-owned close
    handle stays out of the plan so the slice cannot be released from a worker.
    """
    acquired = TPUAccelerator().build_batch_scheduling_plan(
        BatchSchedulingRequest(
            accelerator_type="TPU-V6E",
            accelerator_config=TPUConfig(topology="4x4"),
            tensor_parallel_size=16,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            executor_backend="ray",
            concurrency=1,
        )
    )

    loaded = pickle.loads(pickle.dumps(acquired.plan))

    kwargs = loaded.map_batches_kwargs
    assert kwargs["num_cpus"] == PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST
    assert kwargs["num_gpus"] == 0
    assert kwargs["resources"] == {}
    assert kwargs["runtime_env"]["env_vars"] == {
        "TPU_MULTIHOST_BACKEND": "ray",
        "RAY_TPU_RESOURCE_PER_CHIP": "1",
    }
    strategy = kwargs["scheduling_strategy"]
    assert strategy.placement_group_bundle_index == 0
    assert strategy.placement_group_capture_child_tasks is True
    assert loaded.required_engine_env_vars == kwargs["runtime_env"]["env_vars"]
    assert not hasattr(loaded, "close_handle")
    assert not hasattr(loaded, "reuse_current_placement_group")


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
    acquired = GPUAccelerator().build_batch_scheduling_plan(
        BatchSchedulingRequest(
            accelerator_type="A100",
            accelerator_config=GPUConfig(),
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            executor_backend="uni",
            runtime_env={"env_vars": {"USER_VAR": "gpu"}},
            concurrency=1,
        )
    )

    assert acquired.close_handle is None
    assert acquired.plan.required_engine_env_vars is None
    kwargs = acquired.plan.map_batches_kwargs
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

    acquired = GPUAccelerator().build_batch_scheduling_plan(
        BatchSchedulingRequest(
            accelerator_type="A100",
            accelerator_config=GPUConfig(),
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=1,
            executor_backend="ray",
            concurrency=1,
        )
    )

    assert acquired.close_handle is None
    assert acquired.plan.required_engine_env_vars is None
    kwargs = acquired.plan.map_batches_kwargs
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
    acquired = GPUAccelerator().build_batch_scheduling_plan(
        BatchSchedulingRequest(
            accelerator_type="A100",
            accelerator_config=GPUConfig(),
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            executor_backend="uni",
            placement_group_config={
                "bundle_per_worker": {"GPU": 1, "CPU": 2, "custom": 1},
            },
            concurrency=1,
        )
    )

    kwargs = acquired.plan.map_batches_kwargs
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

    acquired = GPUAccelerator().build_batch_scheduling_plan(
        BatchSchedulingRequest(
            accelerator_type="A100",
            accelerator_config=GPUConfig(),
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            executor_backend="ray",
            placement_group_config={
                "bundle_per_worker": {"GPU": 1, "CPU": 2},
                "strategy": "STRICT_PACK",
            },
            concurrency=1,
        )
    )

    remote_args = acquired.plan.map_batches_kwargs["ray_remote_args_fn"]()
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

    acquired = GPUAccelerator().build_batch_scheduling_plan(
        BatchSchedulingRequest(
            accelerator_type="A100",
            accelerator_config=GPUConfig(),
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            executor_backend="ray",
            placement_group_config={"bundles": None, "strategy": "PACK"},
            concurrency=1,
        )
    )
    acquired.plan.map_batches_kwargs["ray_remote_args_fn"]()
    assert captured["bundles"] == []
