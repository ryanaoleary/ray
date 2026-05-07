from unittest.mock import MagicMock, patch

import pytest

from ray.serve._private.common import CreatePlacementGroupRequest
from ray.serve._private.default_impl import (
    _create_replica_placement_group,
    _ReplicaPlacementGroup,
)
from ray.serve.api import deployment
from ray.serve.config import AcceleratorConfig, TPUSliceSpec


def test_accelerator_config_validation():
    """Test that AcceleratorConfig validates its payload correctly."""

    # Missing tpu spec when type is tpu
    with pytest.raises(ValueError, match="AcceleratorConfig.tpu must be set"):
        AcceleratorConfig(accelerator_type="tpu")

    # Valid config
    tpu_spec = TPUSliceSpec(topology="4x4", accelerator_version="v6e")
    config = AcceleratorConfig(accelerator_type="tpu", tpu=tpu_spec)
    assert config.accelerator_type == "tpu"
    assert config.tpu == tpu_spec


def test_deployment_options_accept_dict_accelerator_config():
    """Test that deployment options accept a dict for accelerator_config."""
    tpu_spec = {"topology": "4x4", "accelerator_version": "v6e"}
    accel_config_dict = {"accelerator_type": "tpu", "tpu": tpu_spec}

    @deployment(accelerator_config=accel_config_dict)
    class MyDeployment:
        pass

    # Verify it resolves into AcceleratorConfig on DeploymentConfig
    assert isinstance(
        MyDeployment._deployment_config.accelerator_config, AcceleratorConfig
    )
    assert MyDeployment._deployment_config.accelerator_config.accelerator_type == "tpu"
    assert MyDeployment._deployment_config.accelerator_config.tpu.topology == "4x4"


def test_deployment_options_accept_typed_accelerator_config():
    """Test that deployment options accept a typed AcceleratorConfig."""
    tpu_spec = TPUSliceSpec(topology="4x4", accelerator_version="v6e")
    accel_config = AcceleratorConfig(accelerator_type="tpu", tpu=tpu_spec)

    @deployment(accelerator_config=accel_config)
    class MyDeployment:
        pass

    assert MyDeployment._deployment_config.accelerator_config == accel_config


def test_create_replica_placement_group_no_accelerator():
    """Test that _create_replica_placement_group fallback works without accelerator."""
    request = CreatePlacementGroupRequest(
        bundles=[{"CPU": 1}], strategy="PACK", target_node_id="", name="test-pg"
    )

    # Mock ray.util.placement_group to avoid real Ray session
    with patch("ray.util.placement_group") as mock_pg:
        mock_pg.return_value = MagicMock()

        result = _create_replica_placement_group(request)

        assert isinstance(result, _ReplicaPlacementGroup)
        assert result._slice_pg is None
        assert result.placement_group is not None
        mock_pg.assert_called_once()


def test_create_replica_placement_group_tpu_dispatch():
    """Test that _create_replica_placement_group dispatches to TPU creation."""
    tpu_spec = TPUSliceSpec(topology="2x2x2", accelerator_version="v4")
    accel_config = AcceleratorConfig(accelerator_type="tpu", tpu=tpu_spec)

    request = CreatePlacementGroupRequest(
        bundles=[{"CPU": 1}],  # This will be ignored by TPU creation
        strategy="SPREAD",
        target_node_id="",
        name="test-pg",
    )

    fake_slice_pg = MagicMock()
    fake_slice_pg.placement_group = MagicMock()

    with patch(
        "ray.serve._private.default_impl.slice_placement_group"
    ) as mock_slice_pg:
        mock_slice_pg.return_value = fake_slice_pg

        result = _create_replica_placement_group(
            request, accelerator_config=accel_config
        )

        assert isinstance(result, _ReplicaPlacementGroup)
        assert result._slice_pg == fake_slice_pg
        assert result.placement_group == fake_slice_pg.placement_group
        mock_slice_pg.assert_called_once()


def test_replica_pg_shutdown_idempotent():
    """Test that _ReplicaPlacementGroup shutdown is idempotent."""
    # Path 1: No accelerator
    mock_pg = MagicMock()
    adapter = _ReplicaPlacementGroup(placement_group=mock_pg)

    with patch("ray.serve._private.default_impl.remove_placement_group") as mock_remove:
        adapter.shutdown()
        mock_remove.assert_called_once_with(mock_pg)

        # Call again, should not raise or call remove again
        adapter.shutdown()
        assert mock_remove.call_count == 1

    # Path 2: With accelerator
    mock_slice_pg = MagicMock()
    adapter_with_accel = _ReplicaPlacementGroup(
        placement_group=mock_pg, _slice_pg=mock_slice_pg
    )

    adapter_with_accel.shutdown()
    mock_slice_pg.shutdown.assert_called_once()
    assert adapter_with_accel._slice_pg is None

    # Call again, should not raise or call shutdown again
    adapter_with_accel.shutdown()
    assert mock_slice_pg.shutdown.call_count == 1
