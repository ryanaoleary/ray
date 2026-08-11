"""Hermetic tests for topology-backed TPU Batch scheduling.

Stub ``slice_placement_group`` / PG wait only. Physical multi-host SlicePG
reservation stays in ``python/ray/tests/test_tpu.py``. These tests cover Batch
config validation, SlicePG kwargs (including single-host labels /
STRICT_PACK), and processor lifecycle.
"""

from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

import ray
import ray.llm._internal.common.accelerators as accelerators
from ray.data.llm import build_processor, vLLMEngineProcessorConfig
from ray.llm._internal.batch.processor import vllm_engine_proc
from ray.llm._internal.batch.processor.base import Processor
from ray.llm._internal.common.accelerators import (
    DEFAULT_PG_READY_TIMEOUT_S,
    DEFAULT_USER_CPU_PER_HOST,
    PARENT_ACTOR_CPU_RESERVE,
    RAY_TPU_RESOURCE_PER_CHIP_ENV_VAR,
    TPU_ENGINE_ENV_VARS,
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
    "kwargs, match",
    [
        ({"accelerator_type": "TPU-V6E"}, "requires accelerator_config with topology"),
        (
            {
                "accelerator_type": "TPU-V6E",
                "accelerator_config": {"kind": "tpu"},
            },
            "requires accelerator_config with topology",
        ),
        (
            {
                "accelerator_type": "TPU-V6E",
                "accelerator_config": {"kind": "tpu", "topology": "4x4"},
                "concurrency": 2,
            },
            "concurrency=1",
        ),
    ],
)
def test_processor_config_rejects_invalid_tpu(kwargs, match):
    with pytest.raises(ValueError, match=match):
        vLLMEngineProcessorConfig(model_source="m", **kwargs)


def test_processor_config_accepts_topology():
    cfg = _topo_config()
    assert isinstance(cfg.accelerator_config, TPUConfig)
    assert cfg.accelerator_config.topology == "4x4"
    assert cfg.concurrency == 1


def test_omitted_accelerator_config_defaults_to_gpu():
    from ray.llm._internal.common.accelerators import GPUAccelerator

    cfg = vLLMEngineProcessorConfig(model_source="m")
    assert cfg.accelerator_config is None
    assert isinstance(
        get_accelerator_backend(cfg.accelerator_config or GPUConfig()), GPUAccelerator
    )


@pytest.mark.parametrize("version, expected", [("v6e", 1), ("v7x", 2)])
def test_vllm_tp_multiplier(version, expected):
    assert _vllm_tp_multiplier(version) == expected


def test_chips_per_vm_requires_topology():
    with pytest.raises(ValueError, match="chips_per_vm requires topology"):
        TPUConfig(chips_per_vm=4)


@pytest.mark.parametrize("bad", [0, -1, True])
def test_chips_per_vm_rejects_invalid_values(bad):
    with pytest.raises(ValueError, match="chips_per_vm must be a positive integer"):
        TPUConfig(topology="2x4", chips_per_vm=bad)


@pytest.mark.parametrize(
    "topology, accelerator_type, tp, chips_per_vm, tpu_per_bundle, strategy, "
    "expect_strategy, expect_labels",
    [
        # Multi-host: SlicePG reservation handles placement; no Batch labels.
        ("4x4", "TPU-V6E", 16, None, None, None, "PACK", False),
        ("4x4", "TPU-V6E", 16, None, 1, None, "PACK", False),
        # chips_per_vm can make an otherwise single-host topology multi-host.
        ("2x4", "TPU-V6E", 8, 4, 1, None, "PACK", False),
        ("2x4", "TPU-V6E", 8, 4, 1, "STRICT_PACK", "STRICT_PACK", False),
        # Single-host default (one host-sized bundle): labels, PACK fine.
        ("2x4", "TPU-V6E", 8, None, None, None, "PACK", True),
        # Single-host multi-bundle: PACK upgrades to STRICT_PACK + labels.
        ("2x4", "TPU-V6E", 8, None, 1, None, "STRICT_PACK", True),
        # v7x single-host multi-bundle.
        ("2x2x1", "TPU-V7X", 8, None, 1, None, "STRICT_PACK", True),
        # v7x multi-host.
        ("2x2x2", "TPU-V7X", 16, None, None, None, "PACK", False),
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
    expect_strategy,
    expect_labels,
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
    assert slice_kwargs["strategy"] == expect_strategy
    resources = slice_kwargs["resources_per_bundle"]
    if tpu_per_bundle is None:
        assert "TPU" not in resources
    else:
        assert resources["TPU"] == float(tpu_per_bundle)
    assert slice_kwargs.get("chips_per_vm") == chips_per_vm

    if expect_labels:
        selectors = slice_kwargs["bundle_label_selector"]
        assert selectors
        for labels in selectors:
            assert labels[ray._raylet.RAY_NODE_TPU_TOPOLOGY_KEY] == topology
            assert ray._raylet.RAY_NODE_TPU_POD_TYPE_KEY in labels
    else:
        assert "bundle_label_selector" not in slice_kwargs

    close_fn()
    handle.shutdown.assert_called_once()


def test_single_host_rejects_spread(stub_slice_pg):
    with pytest.raises(ValueError, match="PACK or STRICT_PACK"):
        _schedule(
            TPUAccelerator(TPUConfig(topology="2x4")),
            tensor_parallel_size=8,
            placement_group_config={
                "bundle_per_worker": {"TPU": 1},
                "strategy": "SPREAD",
            },
        )
    stub_slice_pg[1].assert_not_called()


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
        (
            {},
            {"tensor_parallel_size": 1},
            "requires accelerator_config.topology",
        ),
    ],
)
def test_rejects_invalid_topology_inputs(
    stub_slice_pg, backend_kwargs, option_kwargs, match
):
    with pytest.raises(ValueError, match=match):
        _schedule(TPUAccelerator(TPUConfig(**backend_kwargs)), **option_kwargs)
    stub_slice_pg[1].assert_not_called()
