"""Hermetic tests for TPU Batch scheduling (no TPU hardware required).

Stub ``slice_placement_group`` / PG wait only. Physical SlicePG packing stays in
``python/ray/tests/test_tpu.py``; these tests cover Batch config validation,
backend selection, SlicePG kwargs forwarding, and processor lifecycle.
"""

from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

import ray
import ray.llm._internal.common.accelerators as accelerators
from ray.data.llm import build_processor, vLLMEngineProcessorConfig
from ray.llm._internal.batch.processor.base import Processor
from ray.llm._internal.batch.processor import vllm_engine_proc
from ray.llm._internal.batch.processor.vllm_engine_proc import (
    _default_batch_accelerator_config,
)
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

_CPU_FLOOR = PARENT_ACTOR_CPU_RESERVE + DEFAULT_USER_CPU_PER_HOST


def _schedule(
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
    """Stub SlicePG create/wait; keep Batch validation and builder real."""
    monkeypatch.setenv(RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR, "1")
    handle = MagicMock(name="slice_pg_handle")
    handle.placement_group = MagicMock(name="placement_group")
    handle.num_hosts = 4
    handle.num_bundles = 4
    create = MagicMock(return_value=handle)
    monkeypatch.setattr(accelerators, "slice_placement_group", create)
    monkeypatch.setattr(accelerators, "_wait_for_placement_group", MagicMock())
    monkeypatch.setattr(
        vllm_engine_proc, "download_model_files", lambda *a, **k: "/tmp/mock-model"
    )
    return handle, create


def _topo_config(**kwargs):
    return vLLMEngineProcessorConfig(
        model_source="m",
        accelerator_type="TPU-V6E",
        accelerator_config={"kind": "tpu", "topology": "4x4"},
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs, expect_topology, expect_error",
    [
        ({"accelerator_type": "TPU-V6E"}, None, None),
        (
            {
                "accelerator_type": "TPU-V6E",
                "accelerator_config": {"kind": "tpu", "topology": "4x4"},
            },
            "4x4",
            None,
        ),
        ({"accelerator_type": "TPU-V6E", "concurrency": 2}, None, None),
        (
            {
                "accelerator_type": "TPU-V6E",
                "accelerator_config": {"kind": "tpu", "topology": "4x4"},
                "concurrency": 2,
            },
            None,
            "concurrency=1",
        ),
    ],
)
def test_processor_config_tpu_rules(kwargs, expect_topology, expect_error):
    if expect_error:
        with pytest.raises(ValueError, match=expect_error):
            vLLMEngineProcessorConfig(model_source="m", **kwargs)
        return
    cfg = vLLMEngineProcessorConfig(model_source="m", **kwargs)
    if expect_topology is None:
        assert cfg.accelerator_config is None or cfg.accelerator_config.topology is None
    else:
        assert isinstance(cfg.accelerator_config, TPUConfig)
        assert cfg.accelerator_config.topology == expect_topology


@pytest.mark.parametrize(
    "accelerator_type, expected_config, expected_backend",
    [
        (None, GPUConfig, GPUAccelerator),
        ("TPU-V6E", TPUConfig, TPUAccelerator),
    ],
)
def test_omitted_accelerator_config_defaults(
    accelerator_type, expected_config, expected_backend
):
    cfg = vLLMEngineProcessorConfig(
        model_source="m", accelerator_type=accelerator_type
    )
    assert cfg.accelerator_config is None
    resolved = _default_batch_accelerator_config(cfg)
    assert isinstance(resolved, expected_config)
    assert isinstance(get_accelerator_backend(resolved), expected_backend)


@pytest.mark.parametrize(
    "tp, expect_resources, expect_ray_fn",
    [
        (1, {"TPU": 1.0}, False),
        (4, {}, True),
    ],
)
def test_single_host_tpu_skips_slice_pg(
    stub_slice_pg, tp, expect_resources, expect_ray_fn
):
    _, create = stub_slice_pg
    kwargs, close_fn = _schedule(TPUAccelerator(TPUConfig()), tensor_parallel_size=tp)
    assert close_fn is None
    assert kwargs["resources"] == expect_resources
    assert ("ray_remote_args_fn" in kwargs) is expect_ray_fn
    assert "scheduling_strategy" not in kwargs
    create.assert_not_called()


@pytest.mark.parametrize(
    "version, expected",
    [("v6e", 1), ("v7x", 2)],
)
def test_vllm_tp_multiplier(version, expected):
    assert _vllm_tp_multiplier(version) == expected


@pytest.mark.parametrize(
    "topology, accelerator_type, tp, chips_per_vm, tpu_per_bundle, strategy",
    [
        # Default: Ray fills chips-per-VM; strategy defaults to PACK.
        ("4x4", "TPU-V6E", 16, None, None, None),
        ("2x4", "TPU-V6E", 8, None, None, None),
        # Explicit per-chip worker template.
        ("4x4", "TPU-V6E", 16, None, 1, None),
        # chips_per_vm override + optional strategy passthrough.
        ("2x4", "TPU-V6E", 8, 4, 1, "SPREAD"),
        # v7x: TP = chips × 2.
        ("2x2x2", "TPU-V7X", 16, None, None, None),
        ("2x2x1", "TPU-V7X", 8, None, 1, None),
    ],
)
def test_topology_forwards_slice_pg_kwargs(
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

    kwargs, close_fn = _schedule(
        TPUAccelerator(TPUConfig(topology=topology, chips_per_vm=chips_per_vm)),
        accelerator_type=accelerator_type,
        tensor_parallel_size=tp,
        placement_group_config=pg_config,
    )
    strategy_obj = kwargs["scheduling_strategy"]
    assert isinstance(strategy_obj, PlacementGroupSchedulingStrategy)
    assert strategy_obj.placement_group_bundle_index == 0
    assert kwargs["num_cpus"] == _CPU_FLOOR

    slice_kwargs = create.call_args.kwargs
    assert slice_kwargs["topology"] == topology
    assert slice_kwargs["strategy"] == (strategy or "PACK")
    resources = slice_kwargs["resources_per_bundle"]
    if tpu_per_bundle is None:
        assert "TPU" not in resources
    else:
        assert resources["TPU"] == float(tpu_per_bundle)
    assert slice_kwargs.get("chips_per_vm") == chips_per_vm

    close_fn()
    handle.shutdown.assert_called_once()


def test_topology_merges_runtime_env_and_releases_head(stub_slice_pg):
    handle, create = stub_slice_pg
    kwargs, close_fn = _schedule(
        TPUAccelerator(TPUConfig(topology="4x4")),
        tensor_parallel_size=16,
        runtime_env={"env_vars": {"USER_FLAG": "1"}},
    )
    handle.release_head_pgs.assert_called_once()
    assert create.call_args.kwargs["resources_per_bundle"]["CPU"] == float(_CPU_FLOOR)
    env = kwargs["runtime_env"]["env_vars"]
    assert env["USER_FLAG"] == "1"
    assert {k: env[k] for k in TPU_ENGINE_ENV_VARS} == TPU_ENGINE_ENV_VARS
    close_fn()


def test_eager_timeout_shuts_down_before_head_release(stub_slice_pg, monkeypatch):
    handle, create = stub_slice_pg

    def _timeout(pg, timeout_s):
        assert timeout_s == DEFAULT_PG_READY_TIMEOUT_S
        raise ray.exceptions.GetTimeoutError("timed out")

    monkeypatch.setattr(accelerators, "_wait_for_placement_group", _timeout)
    with pytest.raises(TimeoutError, match="Timed out"):
        _schedule(TPUAccelerator(TPUConfig(topology="4x4")), tensor_parallel_size=16)
    create.assert_called_once()
    handle.shutdown.assert_called_once()
    handle.release_head_pgs.assert_not_called()


def test_builder_pins_bundle_zero_and_close_releases(stub_slice_pg):
    handle, _ = stub_slice_pg
    processor = build_processor(
        _topo_config(engine_kwargs={"tensor_parallel_size": 16})
    )
    assert isinstance(processor, Processor)
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


def test_builder_failure_and_idempotent_close(stub_slice_pg, monkeypatch):
    handle, _ = stub_slice_pg
    real_stage = vllm_engine_proc.vLLMEngineStage
    monkeypatch.setattr(
        vllm_engine_proc,
        "vLLMEngineStage",
        MagicMock(side_effect=RuntimeError("stage boom")),
    )
    config = _topo_config(
        engine_kwargs={"tensor_parallel_size": 16},
        tokenize=False,
        detokenize=False,
        apply_chat_template=False,
    )
    with pytest.raises(RuntimeError, match="stage boom"):
        build_processor(config)
    handle.shutdown.assert_called_once()

    monkeypatch.setattr(vllm_engine_proc, "vLLMEngineStage", real_stage)
    processor = build_processor(config)
    processor.close()
    processor.close()
    assert handle.shutdown.call_count == 2
    with pytest.raises(RuntimeError, match="closed"):
        processor(ray.data.from_items([{"prompt": "x"}]))


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
        (
            {"topology": "2x2x2"},
            {"accelerator_type": "TPU-V7X", "tensor_parallel_size": 8},
            "tensor_parallel_size must be 16",
        ),
    ],
)
def test_rejects_invalid_topology_inputs(
    stub_slice_pg, backend_kwargs, option_kwargs, match
):
    with pytest.raises(ValueError, match=match):
        _schedule(TPUAccelerator(TPUConfig(**backend_kwargs)), **option_kwargs)
    stub_slice_pg[1].assert_not_called()
