"""Compatibility shim for Serve accelerator imports.

The canonical definitions live in ``ray.llm._internal.common.accelerators``.
This module re-exports them so existing Serve import paths keep working.
"""

from ray.llm._internal.common.accelerators import (
    TPU_ACCELERATOR_VALUES,
    AcceleratorBackend,
    AcceleratorType,
    AnyAcceleratorConfig,
    CPUAccelerator,
    CPUConfig,
    GPUAccelerator,
    GPUConfig,
    TPUAccelerator,
    TPUConfig,
    format_ray_accelerator_resource,
    infer_hardware_kind_from_bundles,
)

__all__ = [
    "AcceleratorBackend",
    "AcceleratorType",
    "AnyAcceleratorConfig",
    "CPUAccelerator",
    "CPUConfig",
    "GPUAccelerator",
    "GPUConfig",
    "TPUAccelerator",
    "TPUConfig",
    "TPU_ACCELERATOR_VALUES",
    "format_ray_accelerator_resource",
    "infer_hardware_kind_from_bundles",
]
