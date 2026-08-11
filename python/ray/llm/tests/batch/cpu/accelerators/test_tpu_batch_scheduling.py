"""CPU-CI tests for topology-backed TPU Batch scheduling.

SlicePG packing / labels are covered in ``python/ray/tests/test_tpu.py``.
These tests stub only ``slice_placement_group`` (needs TPU hardware); config,
validation, and processor wiring go through the real Batch APIs.
"""

from typing import Any, Dict, Optional
from unittest.mock import MagicMock

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
_DOWNLOAD = "ray.llm._internal.batch.processor.vllm_engine_proc.download_model_files"


def _options(
    backend: TPUAccelerator,
    *,
    accelerator_type: str = "TPU-V6E",
    tensor_parallel_size: int,
    placement_group_config: Optional[Dict[str, Any]] = None,
    runtime_env: Optional[Dict[str, Any]] = None,
    **engine_overrides,
):
    engine_kwargs = {
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "distributed_executor_backend": "ray",
        **engine_overrides,
    }
    return backend.build_batch_scheduling_options(
        accelerator_type=accelerator_type,
        engine_kwargs=engine_kwargs,
        placement_group_config=placement_group_config,
        runtime_env=runtime_env,
    )


@pytest.fixture
def stub_slice_pg(monkeypatch):
    """Stub SlicePG create/wait; leave Batch validation and processor code real."""
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    handle = MagicMock(name="slice_pg_handle")
    handle.placement_group = MagicMock(name="placement_group")
    handle.num_hosts = 4
    handle.num_bundles = 4
    create = MagicMock(return_value=handle)
    monkeypatch.setattr(f"{_ACCEL}.slice_placement_group", create)
    monkeypatch.setattr(f"{_ACCEL}._wait_for_placement_group", MagicMock())
    monkeypatch.setattr(_DOWNLOAD, lambda *a, **k: "/tmp/mock-model")
    return handle, create


def test_config_requires_tpu_kind_and_concurrency_one():
    cfg = vLLMEngineProcessorConfig(
        model_source="m",
        accelerator_type="TPU-V6E",
        accelerator_config={"kind": "tpu", "topology": "4x4"},
    )
    assert isinstance(cfg.accelerator_config, TPUConfig)
    assert cfg.concurrency == 1

    # Topology is optional at config time; SlicePG Batch still requires it later.
    assert (
        vLLMEngineProcessorConfig(
            model_source="m",
            accelerator_type="TPU-V6E",
            accelerator_config={"kind": "tpu"},
        ).accelerator_config.topology
        is None
    )

    with pytest.raises(ValueError, match="kind='tpu'"):
        vLLMEngineProcessorConfig(model_source="m", accelerator_type="TPU-V6E")
    with pytest.raises(ValueError, match="concurrency=1"):
        vLLMEngineProcessorConfig(
            model_source="m",
            accelerator_type="TPU-V6E",
            accelerator_config={"kind": "tpu", "topology": "4x4"},
            concurrency=2,
        )


def test_omitted_accelerator_config_defaults_to_gpu_backend():
    cfg = vLLMEngineProcessorConfig(model_source="m")
    assert cfg.accelerator_config is None
    assert isinstance(
        get_accelerator_backend(cfg.accelerator_config or GPUConfig()), GPUAccelerator
    )


def test_builder_pins_bundle_zero_and_close_releases_slice(stub_slice_pg):
    handle, _ = stub_slice_pg
    processor = build_processor(
        vLLMEngineProcessorConfig(
            model_source="test-model",
            accelerator_type="TPU-V6E",
            accelerator_config={"kind": "tpu", "topology": "4x4"},
            engine_kwargs={"tensor_parallel_size": 16},
        )
    )
    assert isinstance(processor, Processor)
    assert processor._close_fn is not None
    stage = processor.get_stage_by_name("vLLMEngineStage")
    strategy = stage.map_batches_kwargs["scheduling_strategy"]
    assert isinstance(strategy, PlacementGroupSchedulingStrategy)
    assert strategy.placement_group_bundle_index == 0
    assert strategy.placement_group_capture_child_tasks is True
    assert (
        stage.fn_constructor_kwargs["engine_kwargs"]["distributed_executor_backend"]
        == "ray"
    )
    processor.close()
    handle.shutdown.assert_called_once()


def test_builder_gpu_path_needs_no_slice(monkeypatch):
    monkeypatch.setattr(_DOWNLOAD, lambda *a, **k: "/tmp/mock-model")
    processor = build_processor(vLLMEngineProcessorConfig(model_source="m"))
    assert type(processor) is Processor
    processor.close()


@pytest.mark.parametrize(
    "topology, accelerator_type, tp, chips_per_vm, tpu_per_bundle, strategy",
    [
        ("4x4", "TPU-V6E", 16, None, None, None),
        ("4x4", "TPU-V6E", 16, None, 1, None),
        ("2x4", "TPU-V6E", 8, None, None, None),
        ("2x4", "TPU-V6E", 8, None, 1, None),
        ("2x4", "TPU-V6E", 8, 4, None, None),
        ("2x4", "TPU-V6E", 8, 4, 1, "SPREAD"),
        ("2x2x2", "TPU-V7X", 16, None, None, None),
        ("2x2x2", "TPU-V7X", 16, None, 1, None),
        ("2x2x1", "TPU-V7X", 8, None, 1, None),
    ],
)
def test_topology_bundle_and_strategy_forwarding(
    stub_slice_pg,
    topology,
    accelerator_type,
    tp,
    chips_per_vm,
    tpu_per_bundle,
    strategy,
):
    handle, create = stub_slice_pg
    pg_config = None
    if tpu_per_bundle is not None or strategy is not None:
        pg_config = {}
        if tpu_per_bundle is not None:
            pg_config["bundle_per_worker"] = {"TPU": tpu_per_bundle}
        if strategy is not None:
            pg_config["strategy"] = strategy
    kwargs, close_fn = _options(
        TPUAccelerator(TPUConfig(topology=topology, chips_per_vm=chips_per_vm)),
        accelerator_type=accelerator_type,
        tensor_parallel_size=tp,
        placement_group_config=pg_config,
    )
    assert isinstance(kwargs["scheduling_strategy"], PlacementGroupSchedulingStrategy)
    assert kwargs["scheduling_strategy"].placement_group_bundle_index == 0
    assert kwargs["num_cpus"] == PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST

    slice_kwargs = create.call_args.kwargs
    assert slice_kwargs["strategy"] == (strategy or "PACK")
    assert slice_kwargs["topology"] == topology
    resources = slice_kwargs["resources_per_bundle"]
    if tpu_per_bundle is None:
        assert "TPU" not in resources
    else:
        assert resources["TPU"] == float(tpu_per_bundle)
    if chips_per_vm is None:
        assert "chips_per_vm" not in slice_kwargs
    else:
        assert slice_kwargs["chips_per_vm"] == chips_per_vm

    close_fn()
    handle.shutdown.assert_called_once()


def test_eager_timeout_cleans_up_before_head_release(stub_slice_pg, monkeypatch):
    handle, create = stub_slice_pg

    def _timeout(pg, timeout_s):
        assert timeout_s == DEFAULT_PG_READY_TIMEOUT_S
        raise ray.exceptions.GetTimeoutError("timed out")

    monkeypatch.setattr(f"{_ACCEL}._wait_for_placement_group", _timeout)
    with pytest.raises(TimeoutError, match="Timed out"):
        _options(TPUAccelerator(TPUConfig(topology="4x4")), tensor_parallel_size=16)
    create.assert_called_once()
    handle.shutdown.assert_called_once()
    handle.release_head_pgs.assert_not_called()


def test_builder_failure_and_close_lifecycle(stub_slice_pg, monkeypatch):
    handle, _ = stub_slice_pg
    real_stage = __import__(
        "ray.llm._internal.batch.stages.vllm_engine_stage", fromlist=["vLLMEngineStage"]
    ).vLLMEngineStage
    monkeypatch.setattr(
        "ray.llm._internal.batch.processor.vllm_engine_proc.vLLMEngineStage",
        MagicMock(side_effect=RuntimeError("stage boom")),
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
    handle.shutdown.assert_called_once()

    monkeypatch.setattr(
        "ray.llm._internal.batch.processor.vllm_engine_proc.vLLMEngineStage",
        real_stage,
    )
    processor = build_processor(config)
    processor.close()
    processor.close()
    assert handle.shutdown.call_count == 2
    with pytest.raises(RuntimeError, match="closed"):
        processor(ray.data.from_items([{"prompt": "x"}]))


def test_head_release_cpu_floor_and_runtime_env_merge(stub_slice_pg):
    handle, create = stub_slice_pg
    kwargs, close_fn = _options(
        TPUAccelerator(TPUConfig(topology="4x4")),
        tensor_parallel_size=16,
        runtime_env={"env_vars": {"USER_FLAG": "1"}},
    )
    handle.release_head_pgs.assert_called_once()
    assert create.call_args.kwargs["resources_per_bundle"]["CPU"] == float(
        PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST
    )
    env = kwargs["runtime_env"]["env_vars"]
    assert env["USER_FLAG"] == "1"
    for name, value in TPU_ENGINE_ENV_VARS.items():
        assert env[name] == value
    close_fn()


@pytest.mark.parametrize(
    "backend_kwargs, option_kwargs, match",
    [
        (
            {"topology": "4x4"},
            {"tensor_parallel_size": 16, "distributed_executor_backend": "uni"},
            "distributed_executor_backend",
        ),
        (
            {"topology": "4x4"},
            {"tensor_parallel_size": 16, "pipeline_parallel_size": 2},
            "pipeline_parallel_size",
        ),
        (
            {"topology": "4x4"},
            {"tensor_parallel_size": 16, "data_parallel_size": 2},
            "data_parallel_size",
        ),
        (
            {"topology": "4x4"},
            {
                "tensor_parallel_size": 16,
                "placement_group_config": {"bundle_per_worker": {"GPU": 1, "TPU": 1}},
            },
            "GPU resources are not supported",
        ),
        (
            {"topology": "4x4"},
            {
                "tensor_parallel_size": 16,
                "placement_group_config": {"strategy": "PACK"},
            },
            "must specify bundle_per_worker or bundles",
        ),
        ({}, {"tensor_parallel_size": 8}, "topology"),
        (
            {"topology": "2x2x2"},
            {
                "accelerator_type": "TPU-V7X",
                "tensor_parallel_size": 8,
            },
            "tensor_parallel_size must be 16",
        ),
    ],
)
def test_rejects_invalid_batch_inputs(
    stub_slice_pg, backend_kwargs, option_kwargs, match
):
    assert _vllm_tp_multiplier("v6e") == 1
    assert _vllm_tp_multiplier("v7x") == 2
    with pytest.raises(ValueError, match=match):
        _options(TPUAccelerator(TPUConfig(**backend_kwargs)), **option_kwargs)
    stub_slice_pg[1].assert_not_called()
