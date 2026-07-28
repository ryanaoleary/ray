import pytest
import ray
from ray.data.llm import build_processor, vLLMEngineProcessorConfig


def test_tpu_vllm_batch_inference():
    """Test vLLM batch inference on TPU."""
    try:
        import jax

        if not jax.devices("tpu"):
            pytest.skip("No TPU devices found")
    except Exception:
        pytest.skip("JAX not installed or no TPU devices found")

    processor_config = vLLMEngineProcessorConfig(
        model_source="unsloth/Llama-3.2-1B-Instruct",
        engine_kwargs=dict(
            max_model_len=16384,
            enable_chunked_prefill=True,
            max_num_batched_tokens=2048,
            tensor_parallel_size=4,
        ),
        accelerator_type="TPU-V6E",
        batch_size=16,
        concurrency=1,
    )

    processor = build_processor(
        processor_config,
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": "You are a calculator"},
                {"role": "user", "content": f"{row['id']} ** 3 = ?"},
            ],
            sampling_params=dict(
                temperature=0.3,
                max_tokens=50,
            ),
        ),
        postprocess=lambda row: {
            "resp": row["generated_text"],
        },
    )

    ds = ray.data.range(20)
    ds = processor(ds)
    ds = ds.materialize()
    outs = ds.take_all()
    assert len(outs) == 20
    assert all("resp" in out for out in outs)
