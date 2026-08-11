"""Hermetic unit tests for GPU Batch scheduling defaults."""

from ray.llm._internal.common.accelerators import GPUAccelerator


def test_gpu_batch_defaults_uni_and_ray_executor():
    kwargs, close_fn = GPUAccelerator().build_batch_scheduling_options(
        accelerator_type="A100",
        engine_kwargs={"tensor_parallel_size": 1, "pipeline_parallel_size": 1},
        placement_group_config=None,
        runtime_env=None,
    )
    assert close_fn is None
    assert kwargs["num_gpus"] == 1
    assert kwargs["accelerator_type"] == "A100"

    engine_kwargs = {"tensor_parallel_size": 2, "pipeline_parallel_size": 1}
    kwargs, close_fn = GPUAccelerator().build_batch_scheduling_options(
        accelerator_type="A100",
        engine_kwargs=engine_kwargs,
        placement_group_config=None,
        runtime_env=None,
    )
    assert close_fn is None
    assert engine_kwargs["distributed_executor_backend"] == "ray"
    assert "ray_remote_args_fn" in kwargs
    assert kwargs["num_gpus"] == 0
