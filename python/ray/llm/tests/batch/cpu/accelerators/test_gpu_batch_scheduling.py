"""Hermetic unit tests for GPU Batch scheduling defaults."""

import pytest

from ray.llm._internal.common.accelerators import GPUAccelerator


@pytest.mark.parametrize(
    "tp, expect_backend, expect_num_gpus, expect_ray_fn",
    [
        (1, "uni", 1, False),
        (2, "ray", 0, True),
    ],
)
def test_gpu_batch_executor_defaults(tp, expect_backend, expect_num_gpus, expect_ray_fn):
    engine_kwargs = {"tensor_parallel_size": tp, "pipeline_parallel_size": 1}
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
