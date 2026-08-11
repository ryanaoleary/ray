"""Hermetic unit tests for GPU Batch scheduling defaults."""

from unittest.mock import MagicMock

import pytest

from ray.llm._internal.common.accelerators import GPUAccelerator
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@pytest.mark.parametrize(
    "tp, pp, expect_backend, expect_num_gpus, expect_ray_fn",
    [
        (1, 1, "uni", 1, False),
        (2, 1, "ray", 0, True),
        (2, 2, "ray", 0, True),
    ],
)
def test_gpu_batch_executor_defaults(
    tp, pp, expect_backend, expect_num_gpus, expect_ray_fn
):
    engine_kwargs = {"tensor_parallel_size": tp, "pipeline_parallel_size": pp}
    kwargs, close_fn = GPUAccelerator().build_batch_scheduling_options(
        accelerator_type="A100",
        engine_kwargs=engine_kwargs,
        placement_group_config=None,
        runtime_env=None,
    )
    assert close_fn is None
    assert engine_kwargs.get("distributed_executor_backend", "uni") == expect_backend
    assert kwargs["num_gpus"] == expect_num_gpus
    assert kwargs["accelerator_type"] == "A100"
    assert ("ray_remote_args_fn" in kwargs) is expect_ray_fn


def test_gpu_ray_executor_builds_pack_placement_group(monkeypatch):
    """Coverage moved out of the stage: default ray executor still PACK-schedules TP×PP GPU bundles."""
    import ray.llm._internal.common.accelerators as accelerators_mod

    created = []

    def fake_placement_group(bundles, strategy="PACK", name="", **kwargs):
        created.append({"bundles": bundles, "strategy": strategy, "name": name})
        return MagicMock(name="placement_group")

    monkeypatch.setattr(accelerators_mod, "placement_group", fake_placement_group)

    engine_kwargs = {"tensor_parallel_size": 2, "pipeline_parallel_size": 2}
    kwargs, close_fn = GPUAccelerator().build_batch_scheduling_options(
        accelerator_type="A100",
        engine_kwargs=engine_kwargs,
        placement_group_config=None,
        runtime_env=None,
    )
    assert close_fn is None
    strategy = kwargs["ray_remote_args_fn"]()["scheduling_strategy"]
    assert isinstance(strategy, PlacementGroupSchedulingStrategy)
    assert len(created) == 1
    assert created[0]["strategy"] == "PACK"
    assert len(created[0]["bundles"]) == 4
    for bundle in created[0]["bundles"]:
        assert bundle["GPU"] == 1
        assert bundle["CPU"] == 1
        assert bundle["accelerator_type:A100"] == 0.001
