"""Unit tests for topology-backed TPU Batch scheduling.

Stub ``slice_placement_group`` / PG wait only. Physical multi-host TPU
reservation stays in ``python/ray/tests/test_tpu.py``. These tests cover Batch
config validation, placement kwargs, and processor lifecycle.
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
    """Stub TPU placement create/wait; keep Batch validation and builder real."""
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


@pytest.mark.parametrize(
    "topology, accelerator_type, tp, chips_per_vm, tpu_per_bundle, strategy, "
    "expect_strategy",
    [
        # Topology-backed Batch defaults to SPREAD; explicit strategy wins.
        ("4x4", "TPU-V6E", 16, None, None, None, "SPREAD"),
        ("4x4", "TPU-V6E", 16, None, 1, None, "SPREAD"),
        # Ambiguous v6e 2x4: chips_per_vm=4 selects 2x4-chip VMs over default 1x8.
        ("2x4", "TPU-V6E", 8, 4, 1, None, "SPREAD"),
        ("2x4", "TPU-V6E", 8, 4, 1, "STRICT_PACK", "STRICT_PACK"),
        ("2x4", "TPU-V6E", 8, 4, 1, "PACK", "PACK"),
        ("2x4", "TPU-V6E", 8, None, None, None, "SPREAD"),
        ("2x4", "TPU-V6E", 8, None, 1, None, "SPREAD"),
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
    assert "bundle_label_selector" not in slice_kwargs

    close_fn()
    handle.shutdown.assert_called_once()


def test_chips_per_vm_requires_topology():
    with pytest.raises(ValueError, match="chips_per_vm requires topology"):
        TPUConfig(chips_per_vm=4)


@pytest.mark.parametrize("bad", [0, -1, True])
def test_chips_per_vm_rejects_invalid_values(bad):
    with pytest.raises(ValueError, match="chips_per_vm must be a positive integer"):
        TPUConfig(topology="2x4", chips_per_vm=bad)


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


def test_bundle_per_worker_applies_cpu_floor(stub_slice_pg):
    """Explicit TPU templates without CPU still get the parent-actor floor."""
    _, create = stub_slice_pg
    _, close_fn = _schedule(
        TPUAccelerator(TPUConfig(topology="4x4")),
        tensor_parallel_size=16,
        placement_group_config={"bundle_per_worker": {"TPU": 1}},
    )
    resources = create.call_args.kwargs["resources_per_bundle"]
    assert resources["TPU"] == 1.0
    assert resources["CPU"] == float(_CPU_FLOOR)
    close_fn()


def test_tpu_bundle_per_worker_only_defaults_spread_after_exclude_unset(stub_slice_pg):
    """PlacementGroupConfig defaults strategy=PACK, but exclude_unset omits it.

    Topology-backed Batch must still resolve SPREAD when only bundle_per_worker
    is set on the public config.
    """
    handle, create = stub_slice_pg
    cfg = _topo_config(
        engine_kwargs={"tensor_parallel_size": 16},
        placement_group_config={"bundle_per_worker": {"TPU": 1}},
        tokenize=False,
        detokenize=False,
        apply_chat_template=False,
    )
    assert "strategy" not in cfg.placement_group_config
    assert "bundles" not in cfg.placement_group_config
    processor = build_processor(cfg)
    assert create.call_args.kwargs["strategy"] == "SPREAD"
    assert create.call_args.kwargs["resources_per_bundle"]["TPU"] == 1.0
    assert create.call_args.kwargs["resources_per_bundle"]["CPU"] == float(_CPU_FLOOR)
    processor.close()
    handle.shutdown.assert_called_once()


def test_builder_does_not_mutate_caller_engine_kwargs(stub_slice_pg):
    cfg = _topo_config(
        engine_kwargs={"tensor_parallel_size": 16},
        tokenize=False,
        detokenize=False,
        apply_chat_template=False,
    )
    assert "distributed_executor_backend" not in cfg.engine_kwargs
    processor = build_processor(cfg)
    assert "distributed_executor_backend" not in cfg.engine_kwargs
    assert (
        processor.get_stage_by_name("vLLMEngineStage").fn_constructor_kwargs[
            "engine_kwargs"
        ]["distributed_executor_backend"]
        == "ray"
    )
    processor.close()


def test_processor_context_manager_releases_on_exit_and_exception(stub_slice_pg):
    handle, _ = stub_slice_pg
    cfg = _topo_config(
        engine_kwargs={"tensor_parallel_size": 16},
        tokenize=False,
        detokenize=False,
        apply_chat_template=False,
    )
    with build_processor(cfg) as processor:
        assert isinstance(processor, Processor)
    handle.shutdown.assert_called_once()

    handle.shutdown.reset_mock()
    with pytest.raises(RuntimeError, match="boom"):
        with build_processor(cfg) as processor:
            raise RuntimeError("boom")
    handle.shutdown.assert_called_once()


def test_get_accelerator_backend_rejects_unknown_type():
    with pytest.raises(TypeError, match="Unsupported accelerator config"):
        get_accelerator_backend(object())  # type: ignore[arg-type]


def test_eager_timeout_shuts_down_before_head_release(stub_slice_pg, monkeypatch):
    handle, create = stub_slice_pg

    def _timeout(pg, timeout_s):
        assert timeout_s == DEFAULT_PG_READY_TIMEOUT_S
        raise ray.exceptions.GetTimeoutError("timed out")

    monkeypatch.setattr(accelerators, "_wait_for_placement_group", _timeout)
    with pytest.raises(TimeoutError, match=r"Timed out.*4 hosts, 4 bundles"):
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


def test_close_retries_after_shutdown_failure(stub_slice_pg):
    """Failed shutdown must leave close_fn so a later close() can retry."""
    handle, _ = stub_slice_pg
    handle.shutdown.side_effect = [RuntimeError("shutdown boom"), None]
    processor = build_processor(
        _topo_config(
            engine_kwargs={"tensor_parallel_size": 16},
            tokenize=False,
            detokenize=False,
            apply_chat_template=False,
        )
    )
    with pytest.raises(RuntimeError, match="shutdown boom"):
        processor.close()
    assert handle.shutdown.call_count == 1
    assert processor._close_fn is not None
    with pytest.raises(RuntimeError, match="closed"):
        processor(ray.data.from_items([{"prompt": "x"}]))

    processor.close()
    assert handle.shutdown.call_count == 2
    assert processor._close_fn is None
    processor.close()
    assert handle.shutdown.call_count == 2


def test_close_during_call_does_not_deadlock(stub_slice_pg, monkeypatch):
    """__call__ must not hold the lock across dataset graph construction."""
    import threading

    from ray.data.dataset import Dataset

    processor = build_processor(
        _topo_config(
            engine_kwargs={"tensor_parallel_size": 16},
            tokenize=False,
            detokenize=False,
            apply_chat_template=False,
        )
    )

    map_entered = threading.Event()
    release_map = threading.Event()
    real_map = Dataset.map

    def blocking_map(self, *args, **kwargs):
        map_entered.set()
        assert release_map.wait(timeout=5.0), "map was not released"
        return real_map(self, *args, **kwargs)

    monkeypatch.setattr(Dataset, "map", blocking_map)

    call_error = []

    def run_call():
        try:
            processor(ray.data.from_items([{"prompt": "x"}]))
        except Exception as exc:  # noqa: BLE001 - surface to main thread
            call_error.append(exc)

    caller = threading.Thread(target=run_call)
    caller.start()
    assert map_entered.wait(timeout=5.0), "__call__ never reached Dataset.map"

    # If __call__ still held self._lock across map(), close() would hang here.
    closer = threading.Thread(target=processor.close)
    closer.start()
    closer.join(timeout=5.0)
    assert not closer.is_alive(), "close() deadlocked behind __call__ lock"

    release_map.set()
    caller.join(timeout=5.0)
    assert not caller.is_alive(), "__call__ did not finish after map unblocked"
    assert not call_error, call_error


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
            {"topology": "4x4"},
            {"tensor_parallel_size": 8},
            "tensor_parallel_size must be 16",
        ),
        (
            {"topology": "4x4"},
            {"tensor_parallel_size": True},
            "tensor_parallel_size must be a positive integer",
        ),
        (
            {"topology": "4x4"},
            {"tensor_parallel_size": 16.0},
            "tensor_parallel_size must be a positive integer",
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
    _, create = stub_slice_pg
    with pytest.raises(ValueError, match=match):
        _schedule(TPUAccelerator(TPUConfig(**backend_kwargs)), **option_kwargs)
    create.assert_not_called()
