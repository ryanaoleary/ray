import os
import sys
import warnings

import pytest

import ray
from ray._private.test_utils import placement_group_assert_no_leak
from ray._private.utils import get_ray_doc_version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


def are_pairwise_unique(g):
    s = set()
    for x in g:
        if x in s:
            return False
        s.add(x)
    return True


def test_placement_ready(ray_start_regular):
    @ray.remote
    class Actor:
        def __init__(self):
            pass

        def v(self):
            return 10

    # kBundle_ResourceLabel is placement group reserved resources and
    # can't be used in bundles
    with pytest.raises(Exception):
        ray.util.placement_group(bundles=[{"bundle": 1}])
    # This test is to test the case that even there all resource in the
    # bundle got allocated, we are still able to return from ready[I
    # since ready use 0 CPU
    pg = ray.util.placement_group(bundles=[{"CPU": 1}])
    ray.get(pg.ready())
    a = Actor.options(
        num_cpus=1,
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg),
    ).remote()
    ray.get(a.v.remote())
    ray.get(pg.ready())

    with pytest.raises(ValueError):
        a = Actor.options(
            resources={"bundle": 1},
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg),
        ).remote()
        ray.get(a.v.remote())

    placement_group_assert_no_leak([pg])


@pytest.mark.skipif(
    ray._private.client_mode_hook.is_client_mode_enabled, reason="Fails w/ Ray Client."
)
def test_placement_group_invalid_resource_request(shutdown_only):
    """
    Make sure exceptions are raised if
    requested resources don't fit any bundles.
    """
    ray.init(resources={"a": 1})
    pg = ray.util.placement_group(bundles=[{"a": 1}])

    #
    # Test an actor with 0 cpu.
    #
    @ray.remote
    class A:
        def ready(self):
            pass

    # The actor cannot be scheduled with the default because
    # it requires 1 cpu for the placement, but the pg doesn't have it.
    with pytest.raises(ValueError):
        a = A.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)
        ).remote()
    # Shouldn't work with 1 CPU because pg doesn't contain CPUs.
    with pytest.raises(ValueError):
        a = A.options(
            num_cpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg),
        ).remote()
    # 0 CPU should work.
    a = A.options(
        num_cpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg),
    ).remote()
    ray.get(a.ready.remote())
    del a

    #
    # Test an actor with non-0 resources.
    #
    @ray.remote(resources={"a": 1})
    class B:
        def ready(self):
            pass

    # When resources are given to the placement group,
    # it automatically adds 1 CPU to resources, so it should fail.
    with pytest.raises(ValueError):
        b = B.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)
        ).remote()
    # If 0 cpu is given, it should work.
    b = B.options(
        num_cpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg),
    ).remote()
    ray.get(b.ready.remote())
    del b
    # If resources are requested too much, it shouldn't work.
    with pytest.raises(ValueError):
        # The actor cannot be scheduled with no resource specified.
        # Note that the default actor has 0 cpu.
        B.options(
            num_cpus=0,
            resources={"a": 2},
            schduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg),
        ).remote()

    #
    # Test a function with 1 CPU.
    #
    @ray.remote
    def f():
        pass

    # 1 CPU shouldn't work because the pg doesn't have CPU bundles.
    with pytest.raises(ValueError):
        f.options(
            schduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)
        ).remote()
    # 0 CPU should work.
    ray.get(
        f.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg),
            num_cpus=0,
        ).remote()
    )

    #
    # Test a function with 0 CPU.
    #
    @ray.remote(num_cpus=0)
    def g():
        pass

    # 0 CPU should work.
    ray.get(
        g.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)
        ).remote()
    )

    placement_group_assert_no_leak([pg])


@pytest.mark.parametrize(
    "ray_start_cluster",
    [
        {
            "include_dashboard": True,
        }
    ],
    indirect=True,
)
def test_placement_group_pack(ray_start_cluster):
    @ray.remote(num_cpus=2)
    class Actor(object):
        def __init__(self):
            self.n = 0

        def value(self):
            return self.n

    cluster = ray_start_cluster
    num_nodes = 2
    for i in range(num_nodes):
        cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)

    placement_group = ray.util.placement_group(
        name="name",
        strategy="PACK",
        bundles=[
            {"CPU": 2, "GPU": 0},  # Test 0 resource spec doesn't break tests.
            {"CPU": 2},
        ],
    )
    ray.get(placement_group.ready())
    actor_1 = Actor.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=placement_group, placement_group_bundle_index=0
        )
    ).remote()
    actor_2 = Actor.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=placement_group, placement_group_bundle_index=1
        )
    ).remote()

    ray.get(actor_1.value.remote())
    ray.get(actor_2.value.remote())

    # Make sure all actors in counter_list are collocated in one node.
    actor_info_1 = ray.util.state.get_actor(id=actor_1._actor_id.hex())
    actor_info_2 = ray.util.state.get_actor(id=actor_2._actor_id.hex())

    assert actor_info_1 and actor_info_2

    node_of_actor_1 = actor_info_1.node_id
    node_of_actor_2 = actor_info_2.node_id
    assert node_of_actor_1 == node_of_actor_2
    placement_group_assert_no_leak([placement_group])


@pytest.mark.parametrize(
    "ray_start_cluster",
    [
        {
            "include_dashboard": True,
        }
    ],
    indirect=True,
)
def test_hierarchical_two_pg_contention(ray_start_cluster):
    @ray.remote
    def f():
        return True

    cluster = ray_start_cluster
    # Create 2 nodes, each with 4 CPUs, same rack
    for _ in range(2):
        cluster.add_node(num_cpus=4, labels={"rack": "rack1"})
    ray.init(address=cluster.address)

    # PG1 requires 1 CPU per bundle, 4 bundles = 4 CPUs total
    # STRICT_PACK outer (rack), STRICT_PACK inner (node)
    # Since nodes have 4 CPUs, one node will fit all 4 bundles.
    # The other node will have 4 CPUs free.
    bundles = [[{"CPU": 1}] * 4]

    pg1 = ray.util.placement_group(
        bundles,
        topology_strategy=[{"rack": "STRICT_PACK"}, {"ray.io/node-id": "STRICT_PACK"}],
    )
    ray.get(pg1.ready())

    # PG2 also requires 4 CPUs. It should fit on the OTHER node.
    # If there was a double subtraction leak, this would fail or hang.
    pg2 = ray.util.placement_group(
        bundles,
        topology_strategy=[{"rack": "STRICT_PACK"}, {"ray.io/node-id": "STRICT_PACK"}],
    )
    ray.get(pg2.ready(), timeout=10.0)


def test_placement_group_strict_pack(ray_start_cluster):
    @ray.remote(num_cpus=2)
    class Actor(object):
        def __init__(self):
            self.n = 0

        def value(self):
            return self.n

    cluster = ray_start_cluster
    num_nodes = 2
    for _ in range(num_nodes):
        cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)

    placement_group = ray.util.placement_group(
        name="name",
        strategy="STRICT_PACK",
        bundles=[
            {
                "memory": 50
                * 1024
                * 1024,  # Test memory resource spec doesn't break tests.
                "CPU": 2,
            },
            {"CPU": 2},
        ],
    )
    ray.get(placement_group.ready())
    actor_1 = Actor.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=placement_group, placement_group_bundle_index=0
        )
    ).remote()
    actor_2 = Actor.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=placement_group, placement_group_bundle_index=1
        )
    ).remote()

    ray.get(actor_1.value.remote())
    ray.get(actor_2.value.remote())

    # Make sure all actors in counter_list are collocated in one node.
    actor_info_1 = ray.util.state.get_actor(id=actor_1._actor_id.hex())
    actor_info_2 = ray.util.state.get_actor(id=actor_2._actor_id.hex())

    assert actor_info_1 and actor_info_2

    node_of_actor_1 = actor_info_1.node_id
    node_of_actor_2 = actor_info_2.node_id
    assert node_of_actor_1 == node_of_actor_2

    placement_group_assert_no_leak([placement_group])


@pytest.mark.parametrize(
    "ray_start_cluster",
    [
        {
            "include_dashboard": True,
        }
    ],
    indirect=True,
)
def test_placement_group_spread(ray_start_cluster):
    @ray.remote
    class Actor(object):
        def __init__(self):
            self.n = 0

        def value(self):
            return self.n

    cluster = ray_start_cluster
    num_nodes = 2
    for i in range(num_nodes):
        cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)

    placement_group = ray.util.placement_group(
        name="name",
        strategy="STRICT_SPREAD",
        bundles=[{"CPU": 2}, {"CPU": 2}],
    )
    ray.get(placement_group.ready())
    actors = [
        Actor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group, placement_group_bundle_index=i
            ),
            num_cpus=2,
        ).remote()
        for i in range(num_nodes)
    ]

    [ray.get(actor.value.remote()) for actor in actors]

    # Make sure all actors in counter_list are located in separate nodes.
    actor_info_objs = [
        ray.util.state.get_actor(id=actor._actor_id.hex()) for actor in actors
    ]
    assert are_pairwise_unique([info_obj.node_id for info_obj in actor_info_objs])

    placement_group_assert_no_leak([placement_group])


@pytest.mark.parametrize(
    "ray_start_cluster",
    [
        {
            "include_dashboard": True,
        }
    ],
    indirect=True,
)
def test_placement_group_strict_spread(ray_start_cluster):
    @ray.remote
    class Actor(object):
        def __init__(self):
            self.n = 0

        def value(self):
            return self.n

    cluster = ray_start_cluster
    num_nodes = 3
    for i in range(num_nodes):
        cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)

    placement_group = ray.util.placement_group(
        name="name",
        strategy="STRICT_SPREAD",
        bundles=[{"CPU": 2}, {"CPU": 2}, {"CPU": 2}],
    )
    ray.get(placement_group.ready())
    actors = [
        Actor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group, placement_group_bundle_index=i
            ),
            num_cpus=1,
        ).remote()
        for i in range(num_nodes)
    ]

    [ray.get(actor.value.remote()) for actor in actors]

    # Make sure all actors in counter_list are located in separate nodes.
    actor_info_objs = [
        ray.util.state.get_actor(id=actor._actor_id.hex()) for actor in actors
    ]
    assert are_pairwise_unique([info_obj.node_id for info_obj in actor_info_objs])

    actors_no_special_bundle = [
        Actor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group
            ),
            num_cpus=1,
        ).remote()
        for _ in range(num_nodes)
    ]
    [ray.get(actor.value.remote()) for actor in actors_no_special_bundle]

    actor_no_resource = Actor.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=placement_group
        ),
        num_cpus=2,
    ).remote()
    with pytest.raises(ray.exceptions.GetTimeoutError):
        ray.get(actor_no_resource.value.remote(), timeout=0.5)

    placement_group_assert_no_leak([placement_group])


def test_placement_group_actor_resource_ids(ray_start_cluster):
    @ray.remote(num_cpus=1)
    class F:
        def f(self):
            return ray.get_runtime_context().get_assigned_resources()

    cluster = ray_start_cluster
    num_nodes = 1
    for _ in range(num_nodes):
        cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)

    g1 = ray.util.placement_group([{"CPU": 2}])
    a1 = F.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=g1)
    ).remote()
    resources = ray.get(a1.f.remote())
    assert resources == {"CPU": 1}
    placement_group_assert_no_leak([g1])


def test_placement_group_task_resource_ids(ray_start_cluster):
    @ray.remote(num_cpus=1)
    def f():
        return ray.get_runtime_context().get_assigned_resources()

    cluster = ray_start_cluster
    num_nodes = 1
    for _ in range(num_nodes):
        cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)

    g1 = ray.util.placement_group([{"CPU": 2}])
    o1 = f.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=g1)
    ).remote()
    resources = ray.get(o1)
    assert resources == {"CPU": 1}

    # Now retry with a bundle index constraint.
    o1 = f.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=g1, placement_group_bundle_index=0
        )
    ).remote()
    resources = ray.get(o1)
    assert resources == {"CPU": 1}

    placement_group_assert_no_leak([g1])


def test_placement_group_hang(ray_start_cluster):
    @ray.remote(num_cpus=1)
    def f():
        return ray.get_runtime_context().get_assigned_resources()

    cluster = ray_start_cluster
    num_nodes = 1
    for _ in range(num_nodes):
        cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)

    # Warm workers up, so that this triggers the hang rice.
    ray.get(f.remote())

    g1 = ray.util.placement_group([{"CPU": 2}])
    # This will start out infeasible. The placement group will then be
    # created and it transitions to feasible.
    o1 = f.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=g1)
    ).remote()

    resources = ray.get(o1)
    assert resources == {"CPU": 1}

    placement_group_assert_no_leak([g1])


def test_placement_group_empty_bundle_error(ray_start_regular):
    with pytest.raises(ValueError):
        ray.util.placement_group([])


def test_placement_group_equal_hash(ray_start_regular):
    from copy import copy

    pg1 = ray.util.placement_group([{"CPU": 1}])
    pg2 = copy(pg1)

    # __eq__
    assert pg1 == pg2

    # __hash__
    s = set()
    s.add(pg1)
    assert pg2 in s

    # Compare in remote task
    @ray.remote(num_cpus=0)
    def same(a, b):
        return a == b and b in {a}

    assert ray.get(same.remote(pg1, pg2))

    # Compare before/after object store
    assert ray.get(ray.put(pg1)) == pg1


@pytest.mark.filterwarnings("default:placement_group parameter is deprecated")
def test_placement_group_scheduling_warning(ray_start_regular):
    @ray.remote
    class Foo:
        def foo():
            pass

    pg = ray.util.placement_group(
        name="bar",
        strategy="PACK",
        bundles=[
            {"CPU": 1, "GPU": 0},
        ],
    )
    ray.get(pg.ready())

    # Warning on using deprecated parameters.
    with warnings.catch_warnings(record=True) as w:
        Foo.options(placement_group=pg, placement_group_bundle_index=0).remote()
    assert any(
        "placement_group parameter is deprecated" in str(warning.message)
        for warning in w
    )
    assert any(
        f"docs.ray.io/en/{get_ray_doc_version()}" in str(warning.message)
        for warning in w
    )

    # Pointing to the same doc version as ray.__version__.
    ray.__version__ = "1.13.0"
    with warnings.catch_warnings(record=True) as w:
        Foo.options(placement_group=pg, placement_group_bundle_index=0).remote()
    assert any(
        "docs.ray.io/en/releases-1.13.0" in str(warning.message) for warning in w
    )

    # No warning when scheduling_strategy is specified.
    with warnings.catch_warnings(record=True) as w:
        Foo.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=0
            ),
        ).remote()
    assert not w


@pytest.mark.skipif(
    ray._private.client_mode_hook.is_client_mode_enabled, reason="Fails w/ Ray Client."
)
@pytest.mark.filterwarnings(
    "default:Setting 'object_store_memory' for actors is deprecated"
)
@pytest.mark.filterwarnings(
    "default:Setting 'object_store_memory' for bundles is deprecated"
)
def test_object_store_memory_deprecation_warning(ray_start_regular):
    with warnings.catch_warnings(record=True) as w:

        @ray.remote(object_store_memory=1)
        class Actor:
            pass

        Actor.remote()
    assert any(
        "Setting 'object_store_memory' for actors is deprecated" in str(warning.message)
        for warning in w
    )

    with warnings.catch_warnings(record=True) as w:
        ray.util.placement_group([{"object_store_memory": 1}], strategy="STRICT_PACK")
    assert any(
        "Setting 'object_store_memory' for bundles is deprecated"
        in str(warning.message)
        for warning in w
    )


def test_get_assigned_resources_in_pg(ray_start_cluster):
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=3)
    ray.init(address=cluster.address)

    @ray.remote
    def get_assigned_resources():
        return ray.get_runtime_context().get_assigned_resources()

    resources = ray.get(get_assigned_resources.options(num_cpus=1).remote())
    assert resources == {"CPU": 1}

    pg = ray.util.placement_group(bundles=[{"CPU": 3, "memory": 500}])
    ray.get(pg.ready())

    resources = ray.get(
        get_assigned_resources.options(
            num_cpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg),
        ).remote()
    )
    assert resources == {"CPU": 1}

    resources = ray.get(
        get_assigned_resources.options(
            num_cpus=1,
            memory=100,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=0
            ),
        ).remote()
    )
    assert resources == {"CPU": 1, "memory": 100}


def test_omp_num_threads_in_pg(ray_start_cluster):
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=3)
    ray.init(address=cluster.address)

    @ray.remote(num_cpus=3)
    def test_omp_num_threads():
        omp_threads = os.environ["OMP_NUM_THREADS"]
        return int(omp_threads)

    assert ray.get(test_omp_num_threads.remote()) == 3

    pg = ray.util.placement_group(bundles=[{"CPU": 3}])
    ray.get(pg.ready())

    ref = test_omp_num_threads.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)
    ).remote()
    assert ray.get(ref) == 3

    ref = test_omp_num_threads.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=0
        )
    ).remote()
    assert ray.get(ref) == 3


def test_hierarchical_pg_validations(ray_start_regular):
    # Verify topology_strategy handles lists properly.
    bundles = [{"CPU": 1}, {"CPU": 1}]
    try:
        ray.util.placement_group(
            bundles,
            topology_strategy=[
                {"ray.io/tpu-slice-name": "STRICT_SPREAD"},
                {"ray.io/node-id": "STRICT_PACK"},
            ],
        )
    except Exception as e:
        pytest.fail(f"topology_strategy list validation failed: {e}")

    # Verify single-layer dict stripping ray.io/node-id does not error.
    try:
        ray.util.placement_group(bundles, strategy="STRICT_PACK")
    except Exception as e:
        pytest.fail(f"empty layer validation failed: {e}")

    # Verify bundle_label_selector length validation works for nested bundles.
    hierarchical_bundles = [[{"CPU": 1}, {"CPU": 1}], [{"CPU": 1}]]
    selectors = [{"a": "b"}, {"c": "d"}, {"e": "f"}]
    try:
        ray.util.placement_group(hierarchical_bundles, bundle_label_selector=selectors)
    except ValueError as e:
        pytest.fail(f"Nested bundle label length validation failed: {e}")

    # Verify empty inner lists raise ValueError.
    with pytest.raises(ValueError, match="Hierarchical bundle groups cannot be empty"):
        ray.util.placement_group([[{"CPU": 1}], [], [{"CPU": 1}]])

    # Verify flat bundles with multi-layer topology_strategy raise ValueError.
    with pytest.raises(
        ValueError, match="Multi-layer `topology_strategy` requires hierarchical"
    ):
        ray.util.placement_group(
            [{"CPU": 1}, {"CPU": 1}],
            topology_strategy=[
                {"ray.io/az": "PACK"},
                {"ray.io/rack": "STRICT_PACK"},
            ],
        )


def test_hierarchical_pg_fault_tolerance_partial_failure(ray_start_cluster):
    cluster = ray_start_cluster

    # We mock a small cluster with a shared topology domain.
    # We will simulate 2 nodes in the same domain.
    head_labels = {
        "ray.io/test-domain": "domain-1",
    }
    worker_labels = {
        "ray.io/test-domain": "domain-1",
    }

    cluster.add_node(
        num_cpus=4,
        labels=head_labels,
    )
    worker_node = cluster.add_node(
        num_cpus=4,
        labels=worker_labels,
    )

    ray.init(address=cluster.address)

    # 1. Create a hierarchical placement group that spans 2 nodes
    # Each bundle group requires 4 CPUs, forcing one group on head_node, one group on worker_node
    # We use topology_strategy to ensure they share the same domain-1 but are STRICT_PACKED on node-level
    pg = ray.util.placement_group(
        bundles=[[{"CPU": 4}], [{"CPU": 4}]],
        topology_strategy=[
            {"ray.io/test-domain": "STRICT_PACK"},
            {"ray.io/node-id": "STRICT_PACK"},
        ],
    )
    ray.get(pg.ready())

    # Verify that the placement group is CREATED
    table = ray.util.placement_group_table(pg)
    assert table["state"] == "CREATED"

    # 2. Kill the worker node
    cluster.remove_node(worker_node)

    # Wait for GCS to detect failure and mark it RESCHEDULING
    from ray._private.test_utils import wait_for_condition

    def check_rescheduling():
        t = ray.util.placement_group_table(pg)
        return t["state"] == "RESCHEDULING"

    wait_for_condition(check_rescheduling, timeout=10)

    # 3. Bring up a new node with the same domain
    cluster.add_node(
        num_cpus=4,
        labels=worker_labels,
    )

    # 4. Verify it recovers
    def check_recovered():
        t = ray.util.placement_group_table(pg)
        return t["state"] == "CREATED"

    wait_for_condition(check_recovered, timeout=15)


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))


def test_hierarchical_pg_strict_spread_scheduling(ray_start_cluster):
    cluster = ray_start_cluster
    # Create 3 nodes in 3 distinct zones, each with 2 CPUs
    for i in range(3):
        cluster.add_node(num_cpus=2, labels={"ray.io/zone": f"zone-{i}"})
    ray.init(address=cluster.address)

    # 1. Success case: request 3 bundle groups, each needs 2 CPUs
    # With STRICT_SPREAD on zone, they should be mapped to the 3 distinct zones.
    pg = ray.util.placement_group(
        bundles=[[{"CPU": 2}], [{"CPU": 2}], [{"CPU": 2}]],
        topology_strategy=[
            {"ray.io/zone": "STRICT_SPREAD"},
            {"ray.io/node-id": "STRICT_PACK"},
        ],
    )
    ray.get(pg.ready())
    assert ray.util.placement_group_table(pg)["state"] == "CREATED"

    # 2. Failure case: request 4 bundle groups, but we only have 3 zones.
    pg2 = ray.util.placement_group(
        bundles=[[{"CPU": 2}], [{"CPU": 2}], [{"CPU": 2}], [{"CPU": 2}]],
        topology_strategy=[
            {"ray.io/zone": "STRICT_SPREAD"},
            {"ray.io/node-id": "STRICT_PACK"},
        ],
    )
    # Should stay PENDING because it's infeasible to strictly spread across 4 zones
    import time

    time.sleep(2)
    assert ray.util.placement_group_table(pg2)["state"] == "PENDING"


def test_hierarchical_pg_strict_pack_scheduling(ray_start_cluster):
    cluster = ray_start_cluster
    # Create 2 nodes in the same zone. Node 1 has 4 CPUs. Node 2 has 2 CPUs.
    cluster.add_node(num_cpus=4, labels={"ray.io/zone": "zone-A"})
    cluster.add_node(num_cpus=2, labels={"ray.io/zone": "zone-A"})

    # Create another zone with one node of 6 CPUs
    cluster.add_node(num_cpus=6, labels={"ray.io/zone": "zone-B"})
    ray.init(address=cluster.address)

    # Success case: request 2 bundle groups of {"CPU": 3} each, STRICT_PACKed in the same zone.
    # zone-A has 4 + 2 = 6 CPUs, but no combination of nodes can fit {"CPU": 3} on one node and {"CPU": 3} on another because nodes are 4 and 2. The second {"CPU": 3} will fail on the 2-CPU node.
    # Therefore, zone-A should FAIL. It should backtrack and try zone-B.
    # zone-B has one node with 6 CPUs, so both {"CPU": 3} bundles can pack on it.
    pg = ray.util.placement_group(
        bundles=[[{"CPU": 3}], [{"CPU": 3}]],
        topology_strategy=[
            {"ray.io/zone": "STRICT_PACK"},
            {"ray.io/node-id": "STRICT_PACK"},
        ],
    )
    ray.get(pg.ready())
    table = ray.util.placement_group_table(pg)
    assert table["state"] == "CREATED"


def test_hierarchical_two_layer_same_label(ray_start_cluster):
    # Tests a PG with two layers that use the SAME topology label.
    cluster = ray_start_cluster

    # 2 AZs, 2 nodes each
    cluster.add_node(num_cpus=4, labels={"ray.io/az": "az-1"})
    cluster.add_node(num_cpus=4, labels={"ray.io/az": "az-1"})
    cluster.add_node(num_cpus=4, labels={"ray.io/az": "az-2"})
    cluster.add_node(num_cpus=4, labels={"ray.io/az": "az-2"})

    ray.init(address=cluster.address)

    # Create 4 bundles of CPU:1.
    # We want them to spread across AZs (so 2 groups across az-1 and az-2),
    # but strictly pack within each AZ (so group 1 is strictly packed onto one node in az-1,
    # group 2 is strictly packed onto one node in az-2).
    bundles = [[{"CPU": 1}, {"CPU": 1}], [{"CPU": 1}, {"CPU": 1}]]

    # Test that the same label can be used at multiple layers.
    # Outer strategy: STRICT_SPREAD across AZ.
    # Inner strategy: STRICT_PACK within AZ and STRICT_PACK within node.
    pg = ray.util.placement_group(
        bundles,
        topology_strategy=[
            {"ray.io/az": "STRICT_SPREAD"},
            {"ray.io/az": "STRICT_PACK", "ray.io/node-id": "STRICT_PACK"},
        ],
    )

    ray.get(pg.ready(), timeout=10)

    # Check that they landed on different AZs but same node within the AZ.
    table = ray.util.state.list_placement_groups()
    pg_data = next(p for p in table if p["placement_group_id"] == pg.id.hex())
    assert pg_data["state"] == "CREATED"

    # Group assignments should show different azs
    # Verify using bundles_to_node_id mapping
    bundles_to_node = ray.util.placement_group_table(pg)["bundles_to_node_id"]
    node1 = bundles_to_node[0]
    node2 = bundles_to_node[2]

    # Assert they are on different nodes
    assert node1 != node2

    # Assert strict pack worked within groups
    assert bundles_to_node[0] == bundles_to_node[1]
    assert bundles_to_node[2] == bundles_to_node[3]

    # Assert they are in different AZs
    node_to_az = {}
    for n in ray.nodes():
        node_to_az[n["NodeID"]] = (
            n["Resources"].get("ray.io/az") or n["NodeManagerAddress"]
        )  # fallback to IP if label not in resources
    # Verify the nodes are in different AZs.
    nodes_info = ray.nodes()
    az1 = None
    az2 = None
    for n in nodes_info:
        if n["NodeID"] == node1:
            az1 = n.get("Labels", {}).get("ray.io/az")
        if n["NodeID"] == node2:
            az2 = n.get("Labels", {}).get("ray.io/az")

    if az1 and az2:
        assert az1 != az2


def test_hierarchical_pg_partial_recovery(ray_start_cluster):
    # Test I2-1: 4-group PG, kill one node of group 2's domain.
    # Assert groups 1/3/4 never destroyed, group 2 recovers.
    cluster = ray_start_cluster

    nodes = []
    for i in range(1, 5):
        nodes.append(cluster.add_node(num_cpus=2, labels={"rack": f"rack-{i}"}))
        nodes.append(cluster.add_node(num_cpus=2, labels={"rack": f"rack-{i}"}))

    ray.init(address=cluster.address)

    # 4 groups, 2 bundles of CPU:1 each.
    # Outer: SPREAD across racks, Inner: PACK within node.
    bundles = [
        [{"CPU": 1}, {"CPU": 1}],
        [{"CPU": 1}, {"CPU": 1}],
        [{"CPU": 1}, {"CPU": 1}],
        [{"CPU": 1}, {"CPU": 1}],
    ]

    pg = ray.util.placement_group(
        bundles,
        topology_strategy=[
            {"rack": "SPREAD"},
            {"ray.io/node-id": "PACK"},
        ],
    )
    ray.get(pg.ready(), timeout=15)

    @ray.remote(num_cpus=0.1)
    class Actor:
        def ping(self):
            return "pong"

    # Schedule an actor in each group
    actors = []
    for i in range(4):
        a = Actor.options(
            scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=i * 2
            )
        ).remote()
        actors.append(a)

    assert ray.get([a.ping.remote() for a in actors]) == [
        "pong",
        "pong",
        "pong",
        "pong",
    ]

    table = ray.util.placement_group_table(pg)
    bundles_to_node = table["bundles_to_node_id"]

    # Group 1 (index 1) bundles are at index 2 and 3.
    g1_node_id = bundles_to_node[2]

    for node in nodes:
        if node.unique_id == g1_node_id:
            cluster.remove_node(node)
            break
    # Groups 0, 2, 3 should still be alive since partial self-heal is used.
    import time

    time.sleep(2)

    assert ray.get(actors[0].ping.remote()) == "pong"
    assert ray.get(actors[2].ping.remote()) == "pong"
    assert ray.get(actors[3].ping.remote()) == "pong"

    # Verify group 1 recovers on the other node in the same rack.
    ray.get(pg.ready(), timeout=15)

    a1 = Actor.options(
        scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=2
        )
    ).remote()
    assert ray.get(a1.ping.remote()) == "pong"


def test_hierarchical_pg_strict_pack_rejoin(ray_start_cluster):
    # F1: outer-STRICT_PACK rejoin-siblings invariant test.
    # We kill a node hosting one group. When it recovers, it must rejoin the same outer domain (rack).
    cluster = ray_start_cluster

    nodes = []
    # 2 racks, 2 nodes each
    for i in range(1, 3):
        nodes.append(cluster.add_node(num_cpus=2, labels={"rack": f"rack-{i}"}))
        nodes.append(cluster.add_node(num_cpus=2, labels={"rack": f"rack-{i}"}))

    ray.init(address=cluster.address)

    # 2 groups. Outer=STRICT_PACK(rack), Inner=PACK(node).
    # Since outer is STRICT_PACK, both groups must land in the SAME rack.
    bundles = [
        [{"CPU": 1}, {"CPU": 1}],
        [{"CPU": 1}, {"CPU": 1}],
    ]

    pg = ray.util.placement_group(
        bundles,
        topology_strategy=[
            {"rack": "STRICT_PACK"},
            {"ray.io/node-id": "PACK"},
        ],
    )
    ray.get(pg.ready(), timeout=15)

    table = ray.util.placement_group_table(pg)
    bundles_to_node = table["bundles_to_node_id"]

    g0_node_id = bundles_to_node[0]
    g1_node_id = bundles_to_node[2]

    # Assert they are in the same rack
    rack0 = None
    rack1 = None
    for n in ray.nodes():
        if n["NodeID"] == g0_node_id:
            rack0 = n.get("Labels", {}).get("rack") or n.get("Resources", {}).get(
                "rack"
            )
        if n["NodeID"] == g1_node_id:
            rack1 = n.get("Labels", {}).get("rack") or n.get("Resources", {}).get(
                "rack"
            )

    assert rack0 == rack1
    assert rack0 is not None

    # Kill the node for group 1
    for node in nodes:
        if node.unique_id == g1_node_id:
            cluster.remove_node(node)
            break

    # Wait for group 1 to recover on the other node in the SAME rack.
    # Group 0's node is still alive.
    ray.get(pg.ready(), timeout=15)

    # Verify it recovered on the same rack
    table = ray.util.placement_group_table(pg)
    bundles_to_node = table["bundles_to_node_id"]
    g1_node_id_new = bundles_to_node[2]

    rack1_new = None
    for n in ray.nodes():
        if n["NodeID"] == g1_node_id_new:
            rack1_new = n.get("Labels", {}).get("rack") or n.get("Resources", {}).get(
                "rack"
            )

    assert rack1_new == rack0


def test_hierarchical_pg_exact_fit_race(ray_start_cluster):
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()

    ray.init(address=cluster.address)

    # We submit two placement groups at the same time, both requesting the exact
    # full capacity of the cluster (2 CPUs).
    # Only one should succeed, the other should remain pending without crashing GCS.
    pg1 = ray.util.placement_group(
        [{"CPU": 2}],
        strategy="PACK",
    )

    pg2 = ray.util.placement_group(
        [{"CPU": 2}],
        strategy="PACK",
    )

    # Wait for one to become CREATED
    ready, unready = ray.wait([pg1.ready(), pg2.ready()], num_returns=1, timeout=5.0)
    assert len(ready) == 1

    # The other one should be PENDING
    assert len(unready) == 1

    # Clean up
    ray.util.remove_placement_group(pg1)
    ray.util.remove_placement_group(pg2)
