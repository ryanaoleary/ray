"""The vLLM engine processor."""

import hashlib
import logging
import threading
from typing import Any, Dict, List, Optional

import transformers
from pydantic import Field, TypeAdapter, field_validator, model_validator

import ray
from ray.data import Dataset
from ray.data.block import UserDefinedFunction
from ray.llm._internal.batch.constants import TypeVLLMTaskType, vLLMTaskType
from ray.llm._internal.batch.observability.usage_telemetry.usage import (
    BatchModelTelemetry,
    TelemetryAgent,
    get_or_create_telemetry_agent,
)
from ray.llm._internal.batch.processor.base import (
    OfflineProcessorConfig,
    Processor,
    ProcessorBuilder,
    ProcessorConfig,
)
from ray.llm._internal.batch.processor.utils import (
    build_cpu_stage_map_kwargs,
    get_value_or_fallback,
)
from ray.llm._internal.batch.stages import (
    ChatTemplateStage,
    DetokenizeStage,
    PrepareMultimodalStage,
    StatefulStage,
    TokenizeStage,
    vLLMEngineStage,
)
from ray.llm._internal.batch.stages.configs import (
    ChatTemplateStageConfig,
    DetokenizeStageConfig,
    PrepareMultimodalStageConfig,
    TokenizerStageConfig,
    resolve_stage_config,
)
from ray.llm._internal.common.accelerators import (
    CPU_ACCELERATOR_TYPE_LITERAL,
    TPU_ACCELERATOR_VALUES,
    AnyAcceleratorConfig,
    BatchResourceHandle,
    BatchSchedulingRequest,
    CPUConfig,
    GPUConfig,
    TPUConfig,
    get_accelerator_backend,
    normalize_tpu_accelerator_type,
)
from ray.llm._internal.common.observability.telemetry_utils import DEFAULT_GPU_TYPE
from ray.llm._internal.common.placement import PlacementGroupConfig
from ray.llm._internal.common.utils.download_utils import (
    STREAMING_LOAD_FORMATS,
    NodeModelDownloadable,
    download_model_files,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ARCHITECTURE = "UNKNOWN_MODEL_ARCHITECTURE"

_ACCEL_ADAPTER: TypeAdapter[AnyAcceleratorConfig] = TypeAdapter(AnyAcceleratorConfig)


def _looks_like_tpu(accelerator_type: Any, accelerator_config: Any) -> bool:
    """Detect a TPU request from raw, not-yet-validated config input."""
    if isinstance(accelerator_type, str) and normalize_tpu_accelerator_type(
        accelerator_type
    ).startswith("TPU"):
        return True
    if isinstance(accelerator_config, TPUConfig):
        return True
    return (
        isinstance(accelerator_config, dict) and accelerator_config.get("kind") == "tpu"
    )


class vLLMEngineProcessorConfig(OfflineProcessorConfig):
    """The configuration for the vLLM engine processor."""

    # vLLM stage configurations.
    engine_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="The kwargs to pass to the vLLM engine. See "
        "https://docs.vllm.ai/en/latest/serving/engine_args.html "
        "for more details.",
    )
    task_type: TypeVLLMTaskType = Field(
        default=vLLMTaskType.GENERATE,
        description="The task type to use. If not specified, will use "
        "'generate' by default.",
    )
    log_engine_metrics: bool = Field(
        default=True,
        description="Enable vLLM engine metrics export via Ray's Prometheus endpoint. "
        "When enabled, metrics like prefix cache hit rate, TTFT, TPOT, KV cache "
        "utilization, and scheduler state are available at Ray's metrics endpoint. "
        "Requires Ray to be initialized with _metrics_export_port "
        "(e.g., ray.init(_metrics_export_port=8080)).",
    )
    # LoRA configurations.
    dynamic_lora_loading_path: Optional[str] = Field(
        default=None,
        description="The path to the dynamic LoRA adapter. It is expected "
        "to hold subfolders each for a different lora checkpoint. If not "
        "specified and LoRA is enabled, then the 'model' in LoRA "
        "requests will be interpreted as model ID used by HF transformers.",
    )
    # Custom placement group config for TP/PP.
    placement_group_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Ray placement group configuration for scheduling GPU vLLM engine "
            "workers. Topology-backed TPU batch inference manages placement "
            "internally and does not accept this field. "
            "Can specify either 'bundle_per_worker' (auto-replicated by tp*pp) or "
            "'bundles' (full list of resource dicts). Optionally include 'strategy' "
            "key ('PACK', 'STRICT_PACK', 'SPREAD', or 'STRICT_SPREAD'). "
            "Example with bundle_per_worker: "
            "{'bundle_per_worker': {'CPU': 1, 'GPU': 1}, 'strategy': 'SPREAD'}. "
            "Example with bundles: "
            "{'bundles': [{'CPU': 1, 'GPU': 1}] * 4, 'strategy': 'SPREAD'}."
        ),
    )
    accelerator_config: Optional[AnyAcceleratorConfig] = Field(
        default=None,
        description=(
            "Accelerator configuration for the LLM stage. For TPU batch inference, "
            "pass a mapping such as {'kind': 'tpu', 'topology': '4x4'}. An omitted "
            "accelerator type preserves GPU batch behavior for this processor."
        ),
    )

    @field_validator("accelerator_config", mode="before")
    @classmethod
    def _coerce_accelerator_config(cls, v):
        """Convert a raw mapping into the typed accelerator config."""
        if v is None or isinstance(v, (CPUConfig, GPUConfig, TPUConfig)):
            return v
        return _ACCEL_ADAPTER.validate_python(v)

    @model_validator(mode="before")
    @classmethod
    def _reject_coercible_tpu_concurrency(cls, data: Any) -> Any:
        """Reject TPU concurrency values that Pydantic would silently coerce to 1.

        ``concurrency`` is typed ``Union[int, Tuple[int, int]]``, so ``True``, ``1.0``,
        and ``"1"`` all become ``1`` during field validation. TPU supports exactly one
        replica, and accepting those spellings would hide a misconfiguration, so they
        have to be caught before coercion. Everything else about the accelerator is
        validated after the model is built.
        """
        if not isinstance(data, dict) or not _looks_like_tpu(
            data.get("accelerator_type"), data.get("accelerator_config")
        ):
            return data

        if "concurrency" not in data:
            return {**data, "concurrency": 1}

        raw_concurrency = data["concurrency"]
        if type(raw_concurrency) is not int or raw_concurrency != 1:
            raise ValueError(
                f"TPU batch inference requires concurrency=1; got {raw_concurrency!r} "
                f"of type {type(raw_concurrency).__name__}."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_task_type(cls, values):
        task_type = values.get("task_type", vLLMTaskType.GENERATE)
        if task_type not in vLLMTaskType.values():
            raise ValueError(f"Invalid task type: {task_type}")

        engine_kwargs = values.get("engine_kwargs", {})
        engine_kwargs_task_type = engine_kwargs.get("task_type", "")
        if engine_kwargs_task_type != task_type:
            if engine_kwargs_task_type:
                logger.warning(
                    "The task_type set in engine kwargs (%s) is different from the "
                    "config (%s). Overriding the task_type in engine kwargs to %s.",
                    engine_kwargs_task_type,
                    task_type,
                    task_type,
                )
            engine_kwargs["task_type"] = task_type
        values["engine_kwargs"] = engine_kwargs
        return values

    @model_validator(mode="after")
    def _validate_accelerator(self) -> "vLLMEngineProcessorConfig":
        """Canonicalize the accelerator type and pair it with an accelerator config.

        ``ProcessorConfig`` sets ``validate_assignment``, so plain attribute assignment
        here would re-enter this validator and recurse. Write through
        ``object.__setattr__`` instead.
        """
        if self.accelerator_type is not None:
            normalized_type = normalize_tpu_accelerator_type(self.accelerator_type)
            if normalized_type == CPU_ACCELERATOR_TYPE_LITERAL:
                raise ValueError(
                    "Explicit 'CPU' accelerator type is not supported for vLLM batch inference."
                )
            if normalized_type.startswith("TPU"):
                if normalized_type not in TPU_ACCELERATOR_VALUES:
                    raise ValueError(
                        f"Unknown or unsupported TPU accelerator type: {self.accelerator_type!r}. "
                        f"Supported TPU types: {sorted(TPU_ACCELERATOR_VALUES)}."
                    )
                object.__setattr__(self, "accelerator_type", normalized_type)

        if isinstance(self.accelerator_config, CPUConfig):
            raise ValueError("CPUConfig is not supported for vLLM batch inference.")

        is_tpu_type = (
            self.accelerator_type is not None
            and self.accelerator_type.startswith("TPU")
        )

        if is_tpu_type:
            if isinstance(self.accelerator_config, GPUConfig):
                raise ValueError(
                    f"GPUConfig cannot be used with TPU accelerator_type {self.accelerator_type!r}."
                )
            if self.accelerator_config is None:
                object.__setattr__(self, "accelerator_config", TPUConfig())
        elif isinstance(self.accelerator_config, TPUConfig):
            raise ValueError(
                f"TPUConfig requires a TPU accelerator_type; got {self.accelerator_type!r}."
            )
        elif self.accelerator_config is None:
            object.__setattr__(self, "accelerator_config", GPUConfig())

        if isinstance(self.accelerator_config, TPUConfig):
            if not self.accelerator_config.topology:
                raise ValueError(
                    "TPU batch inference requires accelerator_config with "
                    "kind='tpu' and topology=..., for example "
                    "{'kind': 'tpu', 'topology': '4x4'}."
                )
            if self.placement_group_config is not None:
                raise ValueError(
                    "placement_group_config is not supported for topology-backed TPU "
                    "batch inference. The TPU slice bundle layout and placement are "
                    "managed by the accelerator backend."
                )

        return self

    @field_validator("placement_group_config")
    @classmethod
    def validate_placement_group_config(cls, value):
        if value is None:
            return None
        validated = PlacementGroupConfig(**value)
        return validated.model_dump()


class _ManagedVLLMProcessor(Processor):
    """A context-managed Processor that owns driver-side batch resources (e.g. TPU slice placement group).

    Lifecycle:
    The processor owns the slice placement group for its explicit lifetime. It exposes an idempotent,
    thread-safe close()/shutdown() method and context-manager support (`with build_processor(config) as p:`).
    Users must not close until all datasets derived from this processor have completed. Closing makes future
    processor(dataset) calls fail immediately.
    """

    def __init__(
        self,
        config: ProcessorConfig,
        stages: List[StatefulStage],
        preprocess: Optional[UserDefinedFunction] = None,
        postprocess: Optional[UserDefinedFunction] = None,
        preprocess_map_kwargs: Optional[Dict[str, Any]] = None,
        postprocess_map_kwargs: Optional[Dict[str, Any]] = None,
        close_handle: Optional[BatchResourceHandle] = None,
    ):
        super().__init__(
            config=config,
            stages=stages,
            preprocess=preprocess,
            postprocess=postprocess,
            preprocess_map_kwargs=preprocess_map_kwargs,
            postprocess_map_kwargs=postprocess_map_kwargs,
        )
        self._close_handle = close_handle
        self._closed = False
        self._lock = threading.Lock()

    def close(self) -> None:
        """Idempotently release driver-owned batch resources.

        Marks the processor closed, then requests release of any owned handle.
        If a custom handle's ``shutdown()`` raises, the handle is retained so a
        later ``close()`` can retry. Production ``SlicePlacementGroup.shutdown()``
        typically logs placement-group removal failures and returns successfully,
        so driver exit remains the fallback fate-sharing boundary for those cases.
        """
        with self._lock:
            self._closed = True
            if self._close_handle is None:
                return

            self._close_handle.shutdown()
            self._close_handle = None

    def __call__(self, dataset: Dataset) -> Dataset:
        # Hold the lock across the closed check and Dataset construction so close()
        # cannot interleave. Base Processor.__call__ must not re-enter this method.
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    "Processor is closed. Cannot execute new datasets on a closed processor."
                )
            return super().__call__(dataset)

    def __reduce__(self):
        raise TypeError(
            f"{self.__class__.__name__} owns driver-local lifecycle resources and cannot be serialized or pickled."
        )


def build_vllm_engine_processor(
    config: vLLMEngineProcessorConfig,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    preprocess: Optional[UserDefinedFunction] = None,
    postprocess: Optional[UserDefinedFunction] = None,
    preprocess_map_kwargs: Optional[Dict[str, Any]] = None,
    postprocess_map_kwargs: Optional[Dict[str, Any]] = None,
    telemetry_agent: Optional[TelemetryAgent] = None,
) -> Processor:
    """Construct a Processor and configure stages.

    Args:
        config: The configuration for the processor.
        chat_template_kwargs: The optional kwargs to pass to apply_chat_template.
        preprocess: An optional lambda function that takes a row (dict) as input
            and returns a preprocessed row (dict). The output row must contain the
            required fields for the following processing stages.
        postprocess: An optional lambda function that takes a row (dict) as input
            and returns a postprocessed row (dict).
        preprocess_map_kwargs: Optional kwargs to pass to Dataset.map() for the
            preprocess stage (e.g., num_cpus, memory, concurrency).
        postprocess_map_kwargs: Optional kwargs to pass to Dataset.map() for the
            postprocess stage (e.g., num_cpus, memory, concurrency).
        telemetry_agent: An optional telemetry agent for collecting usage telemetry.

    Returns:
        The constructed processor.
    """
    ray.init(runtime_env=config.runtime_env, ignore_reinit_error=True)

    # Finish downloads and telemetry before acquiring accelerator slices so a
    # later failure does not leave a reserved TPU slice stranded.
    # We download the config files here so that we can report the underlying
    # architecture to the telemetry system. This should be a lightweight operation.
    # Use EXCLUDE_SAFETENSORS for streaming formats or trust_remote_code models,
    # since custom model architectures require Python config files to be downloaded.
    trust_remote_code = config.engine_kwargs.get("trust_remote_code", False)
    if config.engine_kwargs.get(
        "load_format", None
    ) in STREAMING_LOAD_FORMATS or config.engine_kwargs.get("trust_remote_code", False):
        download_model_mode = NodeModelDownloadable.EXCLUDE_SAFETENSORS
    else:
        download_model_mode = NodeModelDownloadable.TOKENIZER_ONLY
    model_path = download_model_files(
        model_id=config.model_source,
        mirror_config=None,
        download_model=download_model_mode,
        download_extra_files=False,
    )

    try:
        hf_config = transformers.AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=config.engine_kwargs.get("trust_remote_code", False),
        )
    except Exception:
        # Failed to retrieve HuggingFace config for telemetry purposes.
        # This is non-fatal: we fall back to DEFAULT_MODEL_ARCHITECTURE for telemetry.
        # The actual model loading happens later in vLLM, which may support models
        # that aren't available via HuggingFace's AutoConfig.
        logger.warning(
            f"Failed to retrieve HuggingFace config for {config.model_source}"
        )
        hf_config = None

    architectures = getattr(hf_config, "architectures", [])
    architecture = architectures[0] if architectures else DEFAULT_MODEL_ARCHITECTURE

    # Copy engine_kwargs so defaults such as distributed_executor_backend do not
    # mutate the caller's configuration object.
    engine_kwargs = dict(config.engine_kwargs)
    tp_size = engine_kwargs.get("tensor_parallel_size", 1)
    pp_size = engine_kwargs.get("pipeline_parallel_size", 1)
    dp_size = engine_kwargs.get("data_parallel_size", 1)
    if "distributed_executor_backend" in engine_kwargs:
        executor_backend = engine_kwargs["distributed_executor_backend"]
    elif isinstance(config.accelerator_config, TPUConfig):
        # Topology-backed TPU always uses the Ray executor so one SlicePG owns
        # both the parent actor and every TPU worker, including single-VM shapes.
        executor_backend = "ray"
    else:
        # Preserve exact legacy GPU behavior.
        executor_backend = "uni" if tp_size * pp_size == 1 else "ray"
    engine_kwargs["distributed_executor_backend"] = executor_backend

    telemetry_agent = get_or_create_telemetry_agent()
    telemetry_agent.push_telemetry_report(
        BatchModelTelemetry(
            model_id_hash=hashlib.sha256(
                config.model_source.encode("utf-8")
            ).hexdigest(),
            processor_config_name=type(config).__name__,
            model_architecture=architecture,
            batch_size=config.batch_size,
            accelerator_type=config.accelerator_type or DEFAULT_GPU_TYPE,
            concurrency=config.concurrency,
            task_type=config.task_type,
            pipeline_parallel_size=pp_size,
            tensor_parallel_size=tp_size,
            data_parallel_size=dp_size,
        )
    )

    backend = get_accelerator_backend(config.accelerator_config)
    request = BatchSchedulingRequest(
        accelerator_type=config.accelerator_type,
        accelerator_config=config.accelerator_config,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        data_parallel_size=dp_size,
        executor_backend=executor_backend,
        placement_group_config=config.placement_group_config,
        runtime_env=config.runtime_env,
        concurrency=config.concurrency,
    )

    # Resolve all stage configs before acquiring expensive accelerator resources
    # (e.g. a TPU slice) so invalid stage configuration fails cheaply.
    processor_defaults = {
        "batch_size": config.batch_size,
        "concurrency": config.concurrency,
        "runtime_env": config.runtime_env,
        "model_source": config.model_source,
    }
    prepare_multimodal_stage_cfg = resolve_stage_config(
        config.prepare_multimodal_stage,
        PrepareMultimodalStageConfig,
        processor_defaults,
    )
    chat_template_stage_cfg = resolve_stage_config(
        getattr(config, "chat_template_stage", config.apply_chat_template),
        ChatTemplateStageConfig,
        processor_defaults,
    )
    tokenize_stage_cfg = resolve_stage_config(
        getattr(config, "tokenize_stage", config.tokenize),
        TokenizerStageConfig,
        processor_defaults,
    )
    detokenize_stage_cfg = resolve_stage_config(
        getattr(config, "detokenize_stage", config.detokenize),
        DetokenizeStageConfig,
        processor_defaults,
    )

    acquired = backend.build_batch_scheduling_plan(request)

    stages: List[StatefulStage] = []
    try:
        if prepare_multimodal_stage_cfg.enabled:
            base_model_config_kwargs = (
                prepare_multimodal_stage_cfg.model_config_kwargs or {}
            )
            model_config_kwargs = {
                **base_model_config_kwargs,
                "model": processor_defaults.get("model_source"),
            }
            stages.append(
                PrepareMultimodalStage(
                    fn_constructor_kwargs=dict(
                        model_config_kwargs=model_config_kwargs,
                        chat_template_content_format=prepare_multimodal_stage_cfg.chat_template_content_format,
                        apply_sys_msg_formatting=prepare_multimodal_stage_cfg.apply_sys_msg_formatting,
                    ),
                    map_batches_kwargs=build_cpu_stage_map_kwargs(
                        prepare_multimodal_stage_cfg
                    ),
                )
            )

        if chat_template_stage_cfg.enabled:
            stages.append(
                ChatTemplateStage(
                    fn_constructor_kwargs=dict(
                        model=chat_template_stage_cfg.model_source,
                        chat_template=get_value_or_fallback(
                            chat_template_stage_cfg.chat_template, config.chat_template
                        ),
                        chat_template_kwargs=get_value_or_fallback(
                            chat_template_stage_cfg.chat_template_kwargs,
                            chat_template_kwargs,
                        ),
                        trust_remote_code=trust_remote_code,
                    ),
                    map_batches_kwargs=build_cpu_stage_map_kwargs(
                        chat_template_stage_cfg
                    ),
                )
            )

        if tokenize_stage_cfg.enabled:
            stages.append(
                TokenizeStage(
                    fn_constructor_kwargs=dict(
                        model=tokenize_stage_cfg.model_source,
                        trust_remote_code=trust_remote_code,
                    ),
                    map_batches_kwargs=build_cpu_stage_map_kwargs(tokenize_stage_cfg),
                )
            )

        # TPU pins concurrency to 1, which yields a single-actor pool through the
        # same ActorPoolStrategy path as GPU.
        compute = ray.data.ActorPoolStrategy(
            **config.get_concurrency(autoscaling_enabled=True),
            max_tasks_in_flight_per_actor=config.max_tasks_in_flight_per_actor,
        )

        vllm_map_batches_kwargs = dict(
            zero_copy_batch=True,
            compute=compute,
            max_concurrency=config.max_concurrent_batches,
            **acquired.plan.map_batches_kwargs,
        )

        stages.append(
            vLLMEngineStage(
                fn_constructor_kwargs=dict(
                    batch_size=config.batch_size,
                    max_concurrent_batches=config.max_concurrent_batches,
                    model=config.model_source,
                    engine_kwargs=engine_kwargs,
                    task_type=config.task_type,
                    max_pending_requests=config.max_pending_requests,
                    dynamic_lora_loading_path=config.dynamic_lora_loading_path,
                    should_continue_on_error=config.should_continue_on_error,
                    log_engine_metrics=config.log_engine_metrics,
                    required_env_vars=acquired.plan.required_engine_env_vars,
                ),
                map_batches_kwargs=vllm_map_batches_kwargs,
            )
        )

        if detokenize_stage_cfg.enabled:
            stages.append(
                DetokenizeStage(
                    fn_constructor_kwargs=dict(
                        model=detokenize_stage_cfg.model_source,
                        trust_remote_code=trust_remote_code,
                    ),
                    map_batches_kwargs=build_cpu_stage_map_kwargs(detokenize_stage_cfg),
                )
            )

        if acquired.close_handle is None:
            return Processor(
                config=config,
                stages=stages,
                preprocess=preprocess,
                postprocess=postprocess,
                preprocess_map_kwargs=preprocess_map_kwargs,
                postprocess_map_kwargs=postprocess_map_kwargs,
            )

        return _ManagedVLLMProcessor(
            config=config,
            stages=stages,
            preprocess=preprocess,
            postprocess=postprocess,
            preprocess_map_kwargs=preprocess_map_kwargs,
            postprocess_map_kwargs=postprocess_map_kwargs,
            close_handle=acquired.close_handle,
        )
    except Exception:
        if acquired.close_handle is not None:
            try:
                acquired.close_handle.shutdown()
            except Exception:
                logger.exception(
                    "Failed to release TPU batch resources after processor construction "
                    "failed; the slice may remain allocated until the driver exits."
                )
        raise


ProcessorBuilder.register(vLLMEngineProcessorConfig, build_vllm_engine_processor)
