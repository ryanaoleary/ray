import pytest

import ray
from ray.util.placement_group import placement_group


def test_placement_group_fallback_validation(ray_start_cluster):
    """
    Test that the Python API accepts a correctly formatted fallback strategy
    and passes it down to the CoreWorker without throwing ValueError.
    """
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)

    fallback_strategy = [
        {"bundles": [{"CPU": 2}], "bundle_label_selector": [{"region": "us-west1"}]},
        {"bundles": [{"CPU": 1}]},
    ]

    pg = placement_group(
        name="validation_success_pg",
        bundles=[{"CPU": 4}],
        strategy="PACK",
        fallback_strategy=fallback_strategy,
    )

    assert not pg.id.is_nil()


def test_placement_group_fallback_validation_errors(ray_start_cluster):
    """
    Test that the Python API correctly raises ValueError for improperly
    formatted fallback_strategy inputs.
    """
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)

    # Invalid: fallback_strategy is not a list
    with pytest.raises(ValueError, match="fallback_strategy must be a list"):
        placement_group(
            bundles=[{"CPU": 1}], fallback_strategy={"bundles": [{"CPU": 1}]}
        )

    # Invalid: fallback_strategy contains non-dict elements
    with pytest.raises(ValueError, match="fallback_strategy\\[0\\] must be a dict"):
        placement_group(bundles=[{"CPU": 1}], fallback_strategy=["not_a_dict"])

    # Invalid: fallback_strategy option missing 'bundles' key
    with pytest.raises(ValueError, match="must contain 'bundles'"):
        placement_group(
            bundles=[{"CPU": 1}],
            fallback_strategy=[{"bundle_label_selector": [{"region": "us-east"}]}],
        )

    # Invalid: fallback_strategy option has invalid bundles
    with pytest.raises(ValueError, match="Bundles must be a non-empty list"):
        placement_group(bundles=[{"CPU": 1}], fallback_strategy=[{"bundles": []}])

    # Invalid: length mismatch between bundles and bundle_label_selector
    with pytest.raises(
        ValueError,
        match="length of `bundle_label_selector` must equal length of `bundles`",
    ):
        placement_group(
            bundles=[{"CPU": 1}],
            fallback_strategy=[
                {
                    "bundles": [{"CPU": 1}, {"CPU": 1}],
                    "bundle_label_selector": [
                        {"region": "us-east"}
                    ],  # Only 1 selector for 2 bundles
                }
            ],
        )

    # Invalid: invalid keys in fallback strategy
    with pytest.raises(ValueError, match="contains invalid options"):
        placement_group(
            bundles=[{"CPU": 1}],
            fallback_strategy=[{"bundles": [{"CPU": 1}], "invalid_key": "value"}],
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
