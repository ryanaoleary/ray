"""Hermetic unit tests for topology-backed TPU Batch scheduling.

All tests run on CPU CI without TPU hardware. Core chips_per_vm / SlicePG
packing and label invariants live in ``python/ray/tests/test_tpu.py``.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

import ray
from ray.data.llm import build_processor, vLLMEngineProcessorConfig
from ray.llm._internal.batch.processor.base import Processor
from ray.llm._internal.common.accelerators import (
    DEFAULT_PG_READY_TIMEOUT_S,
    DEFAULT_USER_CPU_PER_HOST,
    PARENT_ACTOR_CPU_RESERVE,
    RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR,
    TPU_ENGINE_ENV_VARS,
    GPUAccelerator,
    GPUConfig,
    TPUAccelerator,
    TPUConfig,
    _vllm_tp_multiplier,
    get_accelerator_backend,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

_ACCEL = "ray.llm._internal.common.accelerators"


def _tpu_options(
    backend: TPUAccelerator,
    *,
    accelerator_type: str,
    tensor_parallel_size: int,
    executor_backend: str = "ray",
    placement_group_config: Optional[Dict[str, Any]] = None,
    runtime_env: Optional[Dict[str, Any]] = None,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
) -> Tuple[Dict[str, Any], Optional[Callable[[], None]]]:
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
    )


class FakePlacementGroup:
    def __init__(self, pg_id: str = "pg-fake-123"):
        self.id = pg_id

    def ready(self):
        return None


class FakeSlicePlacementGroupHandle:
    def __init__(self, topology: str = "4x4", num_hosts: int = 4, num_bundles: int = 4):
        self.topology = topology
        self.num_hosts = num_hosts
        self.num_bundles = num_bundles
        self.placement_group = FakePlacementGroup()
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
    def _slice(*args, **kwargs):
        if on_slice is not None:
            on_slice(*args, **kwargs)
        fake_handle.topology = kwargs.get("topology", fake_handle.topology)
        return fake_handle

    def _wait(pg, timeout_s):
        if on_wait is not None:
            on_wait(pg, timeout_s)

    monkeypatch.setattr(f"{_ACCEL}.slice_placement_group", _slice)
    monkeypatch.setattr(f"{_ACCEL}._wait_for_placement_group", _wait)
    return fake_handle


@pytest.fixture
def mock_tpu_slice_environment(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    fake_handle = FakeSlicePlacementGroupHandle(
        topology="4x4", num_hosts=4, num_bundles=4
    )
    _install_tpu_slice_fakes(monkeypatch, fake_handle)
    monkeypatch.setattr(
        "ray.llm._internal.batch.processor.vllm_engine_proc.download_model_files",
        lambda *args, **kwargs: "/tmp/mock-model",
    )
    return fake_handle


def test_config_requires_tpu_config_and_concurrency_one():
    cfg = vLLMEngineProcessorConfig(
        model_source="m",
        accelerator_type="TPU-V6E",
        accelerator_config={"kind": "tpu", "topology": "4x4"},
    )
    assert isinstance(cfg.accelerator_config, TPUConfig)
    assert cfg.accelerator_config.topology == "4x4"
    assert cfg.concurrency == 1

    # Topology is optional at config time (single-host may omit it); kind='tpu' is not.
    cfg_no_topo = vLLMEngineProcessorConfig(
        model_source="m",
        accelerator_type="TPU-V6E",
        accelerator_config={"kind": "tpu"},
    )
    assert cfg_no_topo.accelerator_config.topology is None

    with pytest.raises(ValueError, match="kind='tpu'"):
        vLLMEngineProcessorConfig(model_source="m", accelerator_type="TPU-V6E")
    with pytest.raises(ValueError, match="concurrency=1"):
        vLLMEngineProcessorConfig(
            model_source="m",
            accelerator_type="TPU-V6E",
            accelerator_config={"kind": "tpu", "topology": "4x4"},
            concurrency=2,
        )


def test_batch_requires_topology_for_slice_placement(monkeypatch):
    """Topology may be omitted on the config, but SlicePG Batch still needs it."""
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    backend = TPUAccelerator(TPUConfig(kind="tpu"))
    with pytest.raises(ValueError, match="topology"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=8,
        )


def test_omitted_accelerator_config_stays_none_for_gpu():
    cfg = vLLMEngineProcessorConfig(model_source="m")
    assert cfg.accelerator_config is None
    assert isinstance(
        get_accelerator_backend(cfg.accelerator_config or GPUConfig()), GPUAccelerator
    )


def test_builder_returns_managed_processor_and_defaults_executor(
    mock_tpu_slice_environment,
):
    config = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config={"kind": "tpu", "topology": "4x4"},
        engine_kwargs={"tensor_parallel_size": 16},
    )
    processor = build_processor(config)
    assert isinstance(processor, Processor)
    assert processor._close_fn is not None
    stage = processor.stages["vLLMEngineStage"]
    assert (
        stage.map_batches_kwargs["scheduling_strategy"].placement_group_bundle_index
        == 0
    )
    assert (
        stage.map_batches_kwargs[
            "scheduling_strategy"
        ].placement_group_capture_child_tasks
        is True
    )
    assert (
        stage.fn_constructor_kwargs["engine_kwargs"]["distributed_executor_backend"]
        == "ray"
    )
    assert "distributed_executor_backend" not in config.engine_kwargs
    processor.close()
    assert mock_tpu_slice_environment.shutdown_calls == 1


def test_builder_returns_ordinary_processor_for_gpu(monkeypatch):
    monkeypatch.setattr(
        "ray.llm._internal.batch.processor.vllm_engine_proc.download_model_files",
        lambda *args, **kwargs: "/tmp/mock-model",
    )
    processor = build_processor(vLLMEngineProcessorConfig(model_source="m"))
    assert type(processor) is Processor
    assert hasattr(processor, "close")
    processor.close()


@pytest.mark.parametrize(
    "topology, accelerator_type, tp, chips_per_vm, tpu_per_bundle, expected_strategy",
    [
        ("4x4", "TPU-V6E", 16, None, None, "PACK"),
        ("4x4", "TPU-V6E", 16, None, 1, "PACK"),
        ("2x4", "TPU-V6E", 8, None, None, "PACK"),
        ("2x4", "TPU-V6E", 8, None, 1, "STRICT_PACK"),
        ("2x4", "TPU-V6E", 8, 4, None, "PACK"),
        ("2x4", "TPU-V6E", 8, 4, 1, "PACK"),
        ("2x2x2", "TPU-V7X", 16, None, None, "PACK"),
        ("2x2x2", "TPU-V7X", 16, None, 1, "PACK"),
        ("2x2x1", "TPU-V7X", 8, None, 1, "STRICT_PACK"),
    ],
)
def test_topology_bundle_and_strategy_forwarding(
    monkeypatch,
    topology,
    accelerator_type,
    tp,
    chips_per_vm,
    tpu_per_bundle,
    expected_strategy,
):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    fake = FakeSlicePlacementGroupHandle()
    slice_kwargs: List[Dict[str, Any]] = []

    def on_slice(*args, **kwargs):
        slice_kwargs.append(kwargs)

    _install_tpu_slice_fakes(monkeypatch, fake, on_slice=on_slice)
    pg_config = None
    if tpu_per_bundle is not None:
        pg_config = {"bundle_per_worker": {"TPU": tpu_per_bundle}}

    backend = TPUAccelerator(TPUConfig(topology=topology, chips_per_vm=chips_per_vm))
    kwargs, close_fn = _tpu_options(
        backend,
        accelerator_type=accelerator_type,
        tensor_parallel_size=tp,
        placement_group_config=pg_config,
    )
    assert isinstance(kwargs["scheduling_strategy"], PlacementGroupSchedulingStrategy)
    assert kwargs["scheduling_strategy"].placement_group_bundle_index == 0
    assert kwargs["num_cpus"] == PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST
    assert slice_kwargs[0]["strategy"] == expected_strategy
    assert slice_kwargs[0]["topology"] == topology
    resources = slice_kwargs[0]["resources_per_bundle"]
    if tpu_per_bundle is None:
        assert "TPU" not in resources
    else:
        assert resources["TPU"] == float(tpu_per_bundle)
    if chips_per_vm is None:
        assert "chips_per_vm" not in slice_kwargs[0]
    else:
        assert slice_kwargs[0]["chips_per_vm"] == chips_per_vm
    close_fn()
    assert fake.shutdown_calls == 1


def test_batch_strategy_rejects_invalid_layouts(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    fake = FakeSlicePlacementGroupHandle()
    _install_tpu_slice_fakes(monkeypatch, fake)
    backend = TPUAccelerator(TPUConfig(topology="2x4"))
    with pytest.raises(ValueError, match="PACK/STRICT_PACK"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=8,
            placement_group_config={
                "bundle_per_worker": {"TPU": 1},
                "strategy": "SPREAD",
            },
        )

    backend = TPUAccelerator(TPUConfig(topology="4x4"))
    with pytest.raises(ValueError, match="STRICT_PACK cannot represent"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            placement_group_config={
                "bundle_per_worker": {"TPU": 4},
                "strategy": "STRICT_PACK",
            },
        )


def test_eager_acquisition_ordering_and_timeout_cleanup(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    fake = FakeSlicePlacementGroupHandle(topology="4x4", num_hosts=4, num_bundles=4)
    events: List[str] = []

    def on_slice(*args, **kwargs):
        events.append("slice")

    def on_wait(pg, timeout_s):
        events.append(f"wait:{timeout_s}")
        raise ray.exceptions.GetTimeoutError("timed out")

    _install_tpu_slice_fakes(monkeypatch, fake, on_slice=on_slice, on_wait=on_wait)
    backend = TPUAccelerator(TPUConfig(topology="4x4"))
    with pytest.raises(TimeoutError, match="Timed out"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
        )
    assert events == ["slice", f"wait:{DEFAULT_PG_READY_TIMEOUT_S}"]
    assert fake.shutdown_calls == 1
    assert fake.released_head_pgs == 0


def test_construction_failure_cleanup_and_explicit_close(
    mock_tpu_slice_environment, monkeypatch
):
    real_stage = __import__(
        "ray.llm._internal.batch.stages.vllm_engine_stage", fromlist=["vLLMEngineStage"]
    ).vLLMEngineStage

    def boom(*args, **kwargs):
        raise RuntimeError("stage boom")

    monkeypatch.setattr(
        "ray.llm._internal.batch.processor.vllm_engine_proc.vLLMEngineStage",
        boom,
    )
    config = vLLMEngineProcessorConfig(
        model_source="test-model",
        accelerator_type="TPU-V6E",
        accelerator_config={"kind": "tpu", "topology": "4x4"},
        engine_kwargs={"tensor_parallel_size": 16},
        tokenize=False,
        detokenize=False,
        apply_chat_template=False,
    )
    with pytest.raises(RuntimeError, match="stage boom"):
        build_processor(config)
    assert mock_tpu_slice_environment.shutdown_calls == 1

    monkeypatch.setattr(
        "ray.llm._internal.batch.processor.vllm_engine_proc.vLLMEngineStage",
        real_stage,
    )
    processor = build_processor(config)
    processor.close()
    processor.close()  # idempotent
    assert mock_tpu_slice_environment.shutdown_calls == 2
    with pytest.raises(RuntimeError, match="closed"):
        processor(ray.data.from_items([{"prompt": "x"}]))


def test_slice_kwargs_head_release_and_runtime_env_merge(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    fake = FakeSlicePlacementGroupHandle()
    slice_kwargs: List[Dict[str, Any]] = []
    _install_tpu_slice_fakes(
        monkeypatch, fake, on_slice=lambda *a, **k: slice_kwargs.append(k)
    )
    backend = TPUAccelerator(TPUConfig(topology="4x4"))
    kwargs, close_fn = _tpu_options(
        backend,
        accelerator_type="TPU-V6E",
        tensor_parallel_size=16,
        runtime_env={"env_vars": {"USER_FLAG": "1"}},
    )
    assert fake.released_head_pgs == 1
    assert slice_kwargs[0]["resources_per_bundle"]["CPU"] == float(
        PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST
    )
    env = kwargs["runtime_env"]["env_vars"]
    assert env["USER_FLAG"] == "1"
    for name, value in TPU_ENGINE_ENV_VARS.items():
        assert env[name] == value
    close_fn()


def test_rejects_invalid_executor_pp_dp_and_templates(monkeypatch):
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    fake = FakeSlicePlacementGroupHandle()
    _install_tpu_slice_fakes(monkeypatch, fake)
    backend = TPUAccelerator(TPUConfig(topology="4x4"))

    with pytest.raises(ValueError, match="distributed_executor_backend"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            executor_backend="uni",
        )
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            pipeline_parallel_size=2,
        )
    with pytest.raises(ValueError, match="data_parallel_size"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            data_parallel_size=2,
        )
    with pytest.raises(ValueError, match="GPU resources are not supported"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            placement_group_config={"bundle_per_worker": {"GPU": 1, "TPU": 1}},
        )
    with pytest.raises(ValueError, match="must specify bundle_per_worker or bundles"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V6E",
            tensor_parallel_size=16,
            placement_group_config={"strategy": "PACK"},
        )


def test_vllm_tp_multiplier_and_v7x_tp_admission(monkeypatch):
    assert _vllm_tp_multiplier("v6e") == 1
    assert _vllm_tp_multiplier("v7x") == 2
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    fake = FakeSlicePlacementGroupHandle()
    _install_tpu_slice_fakes(monkeypatch, fake)
    backend = TPUAccelerator(TPUConfig(topology="2x2x2"))
    with pytest.raises(ValueError, match="tensor_parallel_size must be 16"):
        _tpu_options(
            backend,
            accelerator_type="TPU-V7X",
            tensor_parallel_size=8,
        )
