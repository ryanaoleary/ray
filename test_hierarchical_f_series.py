import grpc

from ray.core.generated import (
    common_pb2,
    gcs_pb2,
    gcs_service_pb2,
    gcs_service_pb2_grpc,
)


def test_f6_validation_identity():
    channel = grpc.insecure_channel("127.0.0.1:6379")  # Adjust port if needed
    stub = gcs_service_pb2_grpc.PlacementGroupInfoGcsServiceStub(channel)

    pg_spec = common_pb2.PlacementGroupSpec(
        placement_group_id=b"1234567890123456789012345678",
        name="test_f6",
        strategy=common_pb2.PlacementStrategy.STRICT_PACK,
        creator_job_id=b"1234567890123456789012345678",
        creator_job_dead=False,
    )

    # Add a flat bundle
    bundle = pg_spec.bundles.add()
    bundle.bundle_id.bundle_index = 0
    bundle.bundle_id.placement_group_id = pg_spec.placement_group_id
    bundle.unit_resources["CPU"] = 1.0

    # Add >1 topology layer
    layer1 = pg_spec.topology_strategy.add()
    layer1.entries["rack"] = common_pb2.PlacementStrategy.STRICT_PACK
    layer2 = pg_spec.topology_strategy.add()
    layer2.entries["node"] = common_pb2.PlacementStrategy.STRICT_PACK

    request = gcs_service_pb2.CreatePlacementGroupRequest(placement_group_spec=pg_spec)
    try:
        response = stub.AddPlacementGroup(request)
        print("Response status:", response.status)
        assert response.status.code == gcs_pb2.GcsStatus.GCS_STATUS_INVALID
    except Exception as e:
        print("Exception:", e)


if __name__ == "__main__":
    pass
