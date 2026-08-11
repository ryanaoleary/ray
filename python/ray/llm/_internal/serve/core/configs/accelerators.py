"""Compatibility re-exports for accelerator config.

Canonical implementation lives in ``ray.llm._internal.common.accelerators``.
This module preserves the historical Serve import path.
"""

from ray.llm._internal.common.accelerators import (
    TPU_ACCELERATOR_VALUES,
    AcceleratorBackend,
    AcceleratorConfig,
    AcceleratorType,
    AnyAcceleratorConfig,
    CPUAccelerator,
    CPUConfig,
    GPUAccelerator,
    GPUConfig,
    TPUAccelerator,
    TPUConfig,
    format_ray_accelerator_resource,
    get_accelerator_backend,
    infer_hardware_kind_from_bundles,
)

__all__ = [
    "TPU_ACCELERATOR_VALUES",
    "AcceleratorBackend",
    "AcceleratorConfig",
    "AcceleratorType",
    "AnyAcceleratorConfig",
    "CPUAccelerator",
    "CPUConfig",
    "GPUAccelerator",
    "GPUConfig",
    "TPUAccelerator",
    "TPUConfig",
    "format_ray_accelerator_resource",
    "get_accelerator_backend",
    "infer_hardware_kind_from_bundles",
]
