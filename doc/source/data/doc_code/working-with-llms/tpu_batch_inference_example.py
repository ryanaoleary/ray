# __tpu_batch_inference_example_start__
"""Multi-host TPU batch inference with Ray Data LLM.

Requires a TPU slice and a vLLM-TPU / tpu_inference image. Set
TPU_MULTIHOST_BACKEND=ray in runtime_env so the engine orchestrates workers
through Ray (Ray Data does not inject this). Materialize derived Datasets
inside the context manager so the slice placement group is released on exit.
"""

import ray
from ray.data.llm import TPUConfig, build_processor, vLLMEngineProcessorConfig

config = vLLMEngineProcessorConfig(
    model_source="Qwen/Qwen3-4B-Instruct-2507",
    accelerator_type="TPU-V6E",
    accelerator_config=TPUConfig(kind="tpu", topology="4x4"),
    concurrency=1,
    engine_kwargs={"tensor_parallel_size": 16},
    # Required for multi-host TPU: the engine orchestrates its workers through Ray.
    runtime_env={"env_vars": {"TPU_MULTIHOST_BACKEND": "ray"}},
)

with build_processor(config) as processor:
    ds = processor(ray.data.range(32))
    ds.write_parquet("/tmp/tpu-batch-out")
# __tpu_batch_inference_example_end__
