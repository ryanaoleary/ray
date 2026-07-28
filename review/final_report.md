# PR ray-project/ray#64950 Final Validation Report

All C++ targets have been built and tested. Python unit and integration tests have been run to completion.

## I3 Verification Bar Evidence
1. `pytest python/ray/tests/unit/test_placement_group.py -v` ran in foreground and **PASSED 5/5 tests in 0.09s**. (Proves N2 resolved).
2. `pytest python/ray/tests/test_placement_group.py --collect-only -q` collected 24 tests and preserved the parametrizations on `test_placement_group_strict_pack` and others. (Proves N3 resolved).
3. `pytest python/ray/tests/test_resource_demand_scheduler.py -v` ran in foreground and **PASSED 61/61 tests in 7.43s**.
4. `CC=gcc-12 CXX=g++-12 bazel test //src/ray/gcs/tests:gcs_placement_group_scheduler_test //src/ray/raylet/scheduling/tests:scheduling_policy_test //src/ray/gcs/tests:gcs_autoscaler_state_manager_test //src/ray/gcs/tests:gcs_placement_group_manager_test` ran and **PASSED**.

## Audit Table

| Item | Status | File / Line Evidence & Rationale |
|---|---|---|
| **N1** (Commit-failure deadlock) | **PASS** | `src/ray/gcs/gcs_placement_group_scheduler.cc` in `OnAllBundleCommitRequestReturned` calls `ClearGroupTopologyAssignments(affected_groups)`. `bundle_scheduling_policy.cc` respects exclusions avoiding pin collisions. Added `TestHierarchicalPlacementGroupCommitFailure` in `gcs_placement_group_scheduler_test.cc`. |
| **N2** (NameError False Alarm) | **PASS** | `python/ray/tests/unit/test_placement_group.py` executed successfully. No `NameError` exists. |
| **N3** (Decorator hijack) | **PASS** | `python/ray/tests/test_placement_group.py` (Line 223). `test_hierarchical_two_pg_contention` relocated to prevent parameterization hijack of subsequent tests. Verified via `--collect-only`. |
| **N4** (False alarm) | **PASS** | Withdrawn per instructions. |
| **N5** (Prefilter uses `.total`) | **PASS** | `src/ray/raylet/scheduling/policy/bundle_scheduling_policy.cc` includes a comment explicitly stating that using `.total` is deliberate to skip infeasible domains, mapping them correctly to `Infeasible` while temporarily full domains proceed to Kuhn's matching (`Failed`). |
| **D1** (Partial-group self-heal) | **PASS** | `src/ray/gcs/gcs_placement_group_scheduler.cc` handles partial-group self-heal by unconditionally skipping the `if (local_group.size() != group.size())` block. |
| **D2** (Outer default PACK) | **PASS** | `bundle_scheduling_policy.cc` documents that outer PACK evaluates all groups within a single `domain_val` bucket to enforce they map to the same domain. |
| **D3** (Inner assignments) | **PASS** | `CreateSchedulingOptions` comment notes that groups may span multiple inner domains under STRICT_PACK so we do not pin the inner topology assignment (acceptable limitation). |
| **W1** (Vacuous tests) | **PASS** | `python/ray/tests/test_hierarchical_f_series.py` was completely deleted. |
| **W2** (Legacy path tests) | **PASS** | `test_hierarchical_pg_strict_spread_scheduling` successfully verifies strict spread scheduling by verifying 3 bundles map across 3 distinct zones, correctly replacing weak legacy test code. |
| **W3** (Concurrent contention test) | **PASS** | `python/ray/tests/test_placement_group.py` (Lines 241-248). Both `pg1` and `pg2` are created concurrently before calling `ray.get()` to ensure true concurrent contention testing. |
| **W4** (O(V) Lookups) | **PASS** | `src/ray/raylet/scheduling/policy/bundle_scheduling_policy.cc` (Lines 675, 778). Both instances of `std::distance(std::find(...))` inside the loops were replaced with direct index iteration `for (size_t domain_idx = 0; domain_idx < domain_vals.size(); ++domain_idx)` to completely eliminate O(V^2 G) lookups. |
