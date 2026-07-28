import sys

import pytest

from ray.util.placement_group import (
    NODE_ID_LABEL_KEY,
    VALID_PLACEMENT_GROUP_STRATEGIES,
    _validate_bundle_label_selector,
    _validate_bundles,
    validate_placement_group,
)


class TestPlacementGroupValidation:
    def test_strategy_validation(self):
        """Test strategy validation when creating a placement group."""

        # Valid strategies should not raise an exception.
        for strategy in VALID_PLACEMENT_GROUP_STRATEGIES:
            validate_placement_group(bundles=[{"CPU": 1}], strategy=strategy)

        # Any other strategy should raise a ValueError.
        with pytest.raises(ValueError, match="Invalid placement group strategy"):
            validate_placement_group(bundles=[{"CPU": 1}], strategy="invalid")

    def test_topology_strategy_validation(self):
        """Test topology_strategy validation when creating a placement group."""

        valid_topology_strategies = [
            {NODE_ID_LABEL_KEY: "PACK"},
            {NODE_ID_LABEL_KEY: "STRICT_SPREAD"},
            {"ray.io/gpu-domain": "STRICT_PACK"},
            {
                NODE_ID_LABEL_KEY: "SPREAD",
                "ray.io/gpu-domain": "STRICT_PACK",
            },
        ]
        for topology_strategy in valid_topology_strategies:
            validate_placement_group(
                bundles=[{"CPU": 1}], topology_strategy=topology_strategy
            )

        with pytest.raises(
            ValueError, match="strategy` and `topology_strategy` cannot both"
        ):
            validate_placement_group(
                bundles=[{"CPU": 1}],
                strategy="PACK",
                topology_strategy={"ray.io/gpu-domain": "STRICT_PACK"},
            )

        with pytest.raises(ValueError, match="must be a dict"):
            validate_placement_group(
                bundles=[{"CPU": 1}],
                topology_strategy="INVALID",
            )

        with pytest.raises(ValueError, match="keys must be non-empty strings"):
            validate_placement_group(
                bundles=[{"CPU": 1}], topology_strategy={"": "STRICT_PACK"}
            )

        with pytest.raises(ValueError, match="keys must be non-empty strings"):
            validate_placement_group(
                bundles=[{"CPU": 1}], topology_strategy={1: "STRICT_PACK"}
            )

        with pytest.raises(ValueError, match="Invalid topology strategy"):
            validate_placement_group(
                bundles=[{"CPU": 1}],
                topology_strategy={NODE_ID_LABEL_KEY: "invalid"},
            )

        with pytest.raises(ValueError, match="Topology strategy 'PACK' for non-node label 'rack' is not supported"):
            validate_placement_group(
                bundles=[[{"CPU": 1}]],
                topology_strategy=[{"rack": "PACK"}, {NODE_ID_LABEL_KEY: "STRICT_PACK"}],
            )

        with pytest.raises(ValueError, match="Invalid topology strategy"):
            validate_placement_group(
                bundles=[{"CPU": 1}],
                topology_strategy={"ray.io/gpu-domain": "INVALID"},
            )

        with pytest.raises(ValueError, match="at most one topology label"):
            validate_placement_group(
                bundles=[{"CPU": 1}],
                topology_strategy={
                    "ray.io/gpu-domain": "STRICT_PACK",
                    "ray.io/zone": "STRICT_PACK",
                },
            )

    def test_bundle_validation(self):
        """Test _validate_bundle()."""

        # Valid bundles should not raise an exception.
        valid_bundles = [{"CPU": 1, "custom-resource": 2.2}, {"GPU": 0.75}]
        _validate_bundles(valid_bundles)

        # Non-list bundles should raise an exception.
        with pytest.raises(ValueError, match="must be a list"):
            _validate_bundles("not a list")

        # Empty list bundles should raise an exception.
        with pytest.raises(ValueError, match="must be a non-empty list"):
            _validate_bundles([])

        # List that doesn't contain dictionaries should raise an exception.
        with pytest.raises(ValueError, match="resource dictionaries"):
            _validate_bundles([{"CPU": 1}, "not a dict"])

        # List with invalid dictionary entries should raise an exception.
        with pytest.raises(ValueError, match="resource dictionaries"):
            _validate_bundles([{8: 7}, {5: 3.5}])
        with pytest.raises(ValueError, match="resource dictionaries"):
            _validate_bundles([{"CPU": "6"}, {"GPU": "5"}])

        # Bundles with resources that all have 0 values should raise an exception.
        with pytest.raises(ValueError, match="only 0 values"):
            _validate_bundles([{"CPU": 0, "GPU": 0}])

    def test_bundle_label_selector_validation(self):
        """Test _validate_bundle_label_selector()."""

        # Valid label selector list should not raise an exception.
        valid_label_selectors = [
            {"ray.io/market_type": "spot"},
            {"ray.io/accelerator-type": "A100"},
        ]
        _validate_bundle_label_selector(valid_label_selectors)

        # Non-list input should raise an exception.
        with pytest.raises(ValueError, match="must be a list"):
            _validate_bundle_label_selector("not a list")

        # Empty list should not raise (interpreted as no-op).
        _validate_bundle_label_selector([])

        # List with non-dictionary elements should raise an exception.
        with pytest.raises(ValueError, match="must be a list of string dictionary"):
            _validate_bundle_label_selector(["not a dict", {"valid": "label"}])

        # Dictionary with non-string keys or values should raise an exception.
        with pytest.raises(ValueError, match="must be a list of string dictionary"):
            _validate_bundle_label_selector([{1: "value"}, {"key": "val"}])
        with pytest.raises(ValueError, match="must be a list of string dictionary"):
            _validate_bundle_label_selector([{"key": 123}, {"valid": "label"}])

        # Invalid label key or value syntax (delegated to validate_label_selector).
        with pytest.raises(ValueError, match="Invalid label selector provided"):
            _validate_bundle_label_selector([{"INVALID key!": "value"}])

    def test_bundle_label_selector_conflict_validation(self):
        """Test that conflicting label selectors within the same bundle group raise an error."""
        # A single group with conflicting labels for 'ray.io/accelerator'
        hierarchical_bundles = [[{"CPU": 1}, {"CPU": 1}]]
        selectors = [
            {"ray.io/accelerator": "A100"},
            {"ray.io/accelerator": "H100"},
        ]
        # Conflicting labels within a group should just log a warning for STRICT_PACK
        # and not raise an error.
        validate_placement_group(
            bundles=hierarchical_bundles,
            topology_strategy=[{"ray.io/node-id": "STRICT_PACK"}],
            bundle_label_selector=selectors,
        )

        # A valid configuration where labels within the group do not conflict
        valid_selectors = [
            {"ray.io/accelerator": "A100", "ray.io/az": "us-east"},
            {"ray.io/accelerator": "A100"},
        ]
        validate_placement_group(
            bundles=hierarchical_bundles,
            topology_strategy=[{"ray.io/node-id": "STRICT_PACK"}],
            bundle_label_selector=valid_selectors,
        )

        # STRICT_SPREAD allows conflicting labels within a group, no error should be raised.
        validate_placement_group(
            bundles=hierarchical_bundles,
            topology_strategy=[{"ray.io/node-id": "STRICT_SPREAD"}],
            bundle_label_selector=selectors,
        )

    def test_ray_client_guard(self, monkeypatch):
        """Test that Ray Client mode rejects hierarchical placement groups."""
        pg_module = sys.modules["ray.util.placement_group"]
        monkeypatch.setattr(
            pg_module,
            "client_mode_should_convert",
            lambda: True,
        )
        with pytest.raises(
            NotImplementedError,
            match="Hierarchical placement groups and multi-layer topology strategies are not supported via Ray Client.",
        ):
            validate_placement_group(
                bundles=[[{"CPU": 1}]],
                topology_strategy=[{"ray.io/node-id": "STRICT_PACK"}],
            )

