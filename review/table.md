# Iteration-2 Verification Table

| Item ID | Status | File:Line Evidence | Notes |
|---------|--------|--------------------|-------|
| **I2-1** (Self-Heal Revert) | PASS | `src/ray/gcs/gcs_placement_group_manager.cc:218-232` | Reverted the manager-level infeasibility reset via commit `ad335b6693`. |
| **plan-F2** (Proto field reservation) | PASS | `src/ray/protobuf/common.proto:729-731`, `src/ray/protobuf/gcs.proto:690-691` | Fields 12 and 18 are properly `reserved` and the new wire types are on 13 and 19. This was done in `68d52d8c59`. |
| **plan-F1** (Per-group topology) | PASS | `src/ray/gcs/gcs_autoscaler_state_manager.cc:270`, `src/ray/gcs/gcs_placement_group.cc:224` | Per-group assignments and downstreams wired correctly in `78760ad3cf`. |
| **R2** (Mixed flat+grouped) | PASS | `src/ray/gcs/gcs_placement_group_manager.cc:382-399` | Reject mixed flat+grouped bundles added in `c03b187b9d`. |
| **R3** (Post-Schedule race/view) | PASS | `python/ray/tests/test_placement_group.py` | Added concurrent `test_hierarchical_pg_exact_fit_race` in `33fce55812`. Replaced `RAY_CHECK` with graceful failure in `src/ray/raylet/scheduling/policy/bundle_scheduling_policy.cc`. |
| **R4** (State API/Dashboard) | PASS | `python/ray/util/state/common.py:108`, `dashboard/client/src/type/placementGroup.ts:16` | Dashboard logic updated in `68d52d8c59`. |
| **R6** (Heterogeneous order) | PASS | `src/ray/raylet/scheduling/policy/tests/scheduling_policy_test.cc:637` | Un-sorting implementation added, along with `HeterogeneousBundleOutputOrderTest` in C++ (`33fce55812`). |
| **R7** (Constraint keys) | PASS | `src/ray/gcs/gcs_autoscaler_state_manager.cc:536` | Constraint keys updated for per-group logic (`1f68779680`). |
| **R8** (Conflict validation) | PASS | `python/ray/util/placement_group.py:270-281` | Python label conflict validation demoted (`1f68779680`). |
| **T1 / T8 / T9** | PASS | `src/ray/raylet/scheduling/policy/tests/scheduling_policy_test.cc` | Kuhn matching multi-layer tests added in `2565e1adc4` and `ccf321c1d6`. |
| **Repo Hygiene** | PASS | `python/ray/tests/test_hierarchical_f_series.py` | Moved `test_hierarchical_f_series.py` from root into the `tests/` directory (`7da3e0f080`). |
