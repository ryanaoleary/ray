// Copyright 2021 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ray/raylet/scheduling/policy/bundle_scheduling_policy.h"

#include <algorithm>
#include <numeric>

namespace ray {
namespace raylet_scheduling_policy {

SchedulingResult SortSchedulingResult(const SchedulingResult &result,
                                      const std::vector<int> &sorted_index) {
  if (result.status.IsSuccess()) {
    std::vector<scheduling::NodeID> sorted_nodes(result.selected_nodes.size());
    for (int i = 0; i < (int)sorted_index.size(); i++) {
      sorted_nodes[sorted_index[i]] = result.selected_nodes[i];
    }
    return SchedulingResult::Success(std::move(sorted_nodes));
  } else {
    return result;
  }
}

bool BundleSchedulingPolicy::IsRequestFeasible(
    const std::vector<const ResourceRequest *> &resource_request_list,
    const absl::flat_hash_set<scheduling::NodeID> &candidate_nodes) const {
  for (const auto &request : resource_request_list) {
    bool bundle_feasible = std::any_of(
        candidate_nodes.begin(), candidate_nodes.end(), [&](const auto &node_id) {
          // Validates both resource and label constraints are feasible.
          return cluster_resource_manager_.HasFeasibleResources(node_id, *request);
        });
    if (!bundle_feasible) {
      return false;
    }
  }
  return true;
}

std::pair<std::vector<int>, std::vector<const ResourceRequest *>>
BundleSchedulingPolicy::SortRequiredResources(
    const std::vector<const ResourceRequest *> &resource_request_list) {
  std::vector<int> sorted_index(resource_request_list.size());
  std::iota(sorted_index.begin(), sorted_index.end(), 0);

  // Here we sort in reverse order:
  // sort(_, _, a < b) would result in the vector [a < b < c]
  // sort(_, _, a > b) would result in the vector [c > b > a] which leads to our desired
  // outcome of having highest priority `ResourceRequest` being scheduled first.

  std::sort(sorted_index.begin(), sorted_index.end(), [&](int b_idx, int a_idx) {
    const auto &a = *resource_request_list[a_idx];
    const auto &b = *resource_request_list[b_idx];

    // TODO (jon-chuang): the exact resource priority defined here needs to be revisted.

    // Notes: This is a comparator for sorting in c++. We return true if a < b based on a
    // resource at the given level of priority. If tied, we attempt to resolve based on
    // the resource at the next level of priority.
    //
    // The order of priority is: `ResourceRequest`s with GPU requirements first, then
    // extra resources, then object store memory, memory and finally CPU requirements. If
    // two `ResourceRequest`s require a resource under consideration, the one requiring
    // more of the resource is prioritized.

    auto gpu = scheduling::ResourceID::GPU();
    if (a.Get(gpu) != b.Get(gpu)) {
      return a.Get(gpu) < b.Get(gpu);
    }

    // Make sure that resources are always sorted in the same order
    std::set<scheduling::ResourceID> extra_resources_set;
    for (const auto &r : a.ResourceIds()) {
      if (!r.IsPredefinedResource()) {
        extra_resources_set.insert(r);
      }
    }
    for (const auto &r : b.ResourceIds()) {
      if (!r.IsPredefinedResource()) {
        extra_resources_set.insert(r);
      }
    }

    for (const auto &r : extra_resources_set) {
      auto a_resource = a.Get(r);
      auto b_resource = b.Get(r);
      if (a_resource != b_resource) {
        return a_resource < b_resource;
      }
    }
    for (auto id : std::vector({scheduling::ResourceID::ObjectStoreMemory(),
                                scheduling::ResourceID::Memory(),
                                scheduling::ResourceID::CPU()})) {
      if (a.Get(id) != b.Get(id)) {
        return a.Get(id) < b.Get(id);
      }
    }
    return false;
  });

  std::vector<const ResourceRequest *> sorted_resource_request_list(
      resource_request_list);
  for (size_t i = 0; i < sorted_index.size(); i++) {
    sorted_resource_request_list[i] = resource_request_list[sorted_index[i]];
  }

  return {std::move(sorted_index), std::move(sorted_resource_request_list)};
}

scheduling::NodeID BundleSchedulingPolicy::GetBestNode(
    const ResourceRequest &required_resources,
    const absl::flat_hash_set<scheduling::NodeID> &candidate_nodes,
    const SchedulingOptions &options) const {
  double best_node_score = -1;
  auto best_node_id = scheduling::NodeID::Nil();

  // Score the nodes.
  for (const auto &node_id : candidate_nodes) {
    const auto &node_resources = cluster_resource_manager_.GetNodeResources(node_id);
    double node_score = node_scorer_->Score(required_resources, node_resources);
    if (best_node_id.IsNil() || best_node_score < node_score) {
      best_node_id = node_id;
      best_node_score = node_score;
    }
  }
  if (!best_node_id.IsNil() && best_node_score >= 0) {
    return best_node_id;
  }
  return scheduling::NodeID::Nil();
}

////////////////////  BundlePackSchedulingPolicy  ///////////////////////////////
SchedulingResult BundlePackSchedulingPolicy::Schedule(
    const std::vector<const ResourceRequest *> &resource_request_list,
    SchedulingOptions options,
    const absl::flat_hash_set<scheduling::NodeID> &candidate_nodes_in) {
  absl::flat_hash_set<scheduling::NodeID> candidate_nodes = candidate_nodes_in;
  RAY_CHECK(!resource_request_list.empty());
  if (candidate_nodes.empty()) {
    RAY_LOG(DEBUG) << "The candidate nodes is empty, return directly.";
    return SchedulingResult::Infeasible();
  }

  // First schedule scarce resources (such as GPU) and large capacity resources to improve
  // the scheduling success rate.
  auto sorted_result = SortRequiredResources(resource_request_list);
  const auto &sorted_index = sorted_result.first;
  const auto &sorted_resource_request_list = sorted_result.second;

  if (!IsRequestFeasible(sorted_resource_request_list, candidate_nodes)) {
    RAY_LOG(DEBUG) << "Request requires labels or resources not present in the cluster.";
    return SchedulingResult::Infeasible();
  }

  std::vector<scheduling::NodeID> result_nodes;
  result_nodes.resize(sorted_resource_request_list.size());
  std::list<std::pair<int, const ResourceRequest *>> required_resources_list_copy;
  int index = 0;
  for (const auto &resource_request : sorted_resource_request_list) {
    required_resources_list_copy.emplace_back(index++, resource_request);
  }

  while (!required_resources_list_copy.empty()) {
    const auto &required_resources_index = required_resources_list_copy.front().first;
    const auto &required_resources = required_resources_list_copy.front().second;
    auto best_node_id = GetBestNode(*required_resources, candidate_nodes, options);
    if (best_node_id.IsNil()) {
      // There is no node to meet the scheduling requirements.
      break;
    }

    RAY_CHECK(cluster_resource_manager_.SubtractNodeAvailableResources(
        best_node_id, *required_resources));
    result_nodes[required_resources_index] = best_node_id;
    required_resources_list_copy.pop_front();

    // We try to schedule more resources on one node.
    for (auto iter = required_resources_list_copy.begin();
         iter != required_resources_list_copy.end();) {
      // If the node has sufficient resources, allocate it.
      if (cluster_resource_manager_.HasAvailableResources(
              best_node_id, *iter->second, false)) {
        RAY_CHECK(cluster_resource_manager_.SubtractNodeAvailableResources(
            best_node_id, *iter->second));
        result_nodes[iter->first] = best_node_id;
        required_resources_list_copy.erase(iter++);
      } else {
        // Otherwise try other node.
        ++iter;
      }
    }
    candidate_nodes.erase(best_node_id);
  }

  // Releasing the resources temporarily deducted from `cluster_resource_manager_`.
  for (size_t res_node_idx = 0; res_node_idx < result_nodes.size(); res_node_idx++) {
    // If `PackSchedule` fails, the id of some nodes may be nil.
    if (!result_nodes[res_node_idx].IsNil()) {
      RAY_CHECK(cluster_resource_manager_.AddNodeAvailableResources(
          result_nodes[res_node_idx],
          (*sorted_resource_request_list[res_node_idx]).GetResourceSet()));
    }
  }

  if (!required_resources_list_copy.empty()) {
    // Can't meet the scheduling requirements temporarily.
    return SchedulingResult::Failed();
  }
  return SortSchedulingResult(SchedulingResult::Success(std::move(result_nodes)),
                              sorted_index);
}

//////////////////////  BundleSpreadSchedulingPolicy  ///////////////////////////
SchedulingResult BundleSpreadSchedulingPolicy::Schedule(
    const std::vector<const ResourceRequest *> &resource_request_list,
    SchedulingOptions options,
    const absl::flat_hash_set<scheduling::NodeID> &candidate_nodes_in) {
  absl::flat_hash_set<scheduling::NodeID> candidate_nodes = candidate_nodes_in;
  RAY_CHECK(!resource_request_list.empty());
  if (candidate_nodes.empty()) {
    RAY_LOG(DEBUG) << "The candidate nodes is empty, return directly.";
    return SchedulingResult::Infeasible();
  }

  // First schedule scarce resources (such as GPU) and large capacity resources to improve
  // the scheduling success rate.
  auto sorted_result = SortRequiredResources(resource_request_list);
  const auto &sorted_index = sorted_result.first;
  const auto &sorted_resource_request_list = sorted_result.second;

  if (!IsRequestFeasible(sorted_resource_request_list, candidate_nodes)) {
    RAY_LOG(DEBUG) << "Request requires labels or resources not present in the cluster.";
    return SchedulingResult::Infeasible();
  }

  std::vector<scheduling::NodeID> result_nodes;
  absl::flat_hash_set<scheduling::NodeID> selected_nodes;
  for (const auto &resource_request : sorted_resource_request_list) {
    // Score and sort nodes.
    auto best_node_id = GetBestNode(*resource_request, candidate_nodes, options);

    // There are nodes to meet the scheduling requirements.
    if (!best_node_id.IsNil()) {
      result_nodes.emplace_back(best_node_id);
      RAY_CHECK(cluster_resource_manager_.SubtractNodeAvailableResources(
          best_node_id, *resource_request));
      candidate_nodes.erase(best_node_id);
      selected_nodes.insert(best_node_id);
    } else {
      // Scheduling from selected nodes.
      best_node_id = GetBestNode(*resource_request, selected_nodes, options);
      if (!best_node_id.IsNil()) {
        result_nodes.emplace_back(best_node_id);
        RAY_CHECK(cluster_resource_manager_.SubtractNodeAvailableResources(
            best_node_id, *resource_request));
      } else {
        break;
      }
    }
  }

  // Releasing the resources temporarily deducted from `cluster_resource_manager_`.
  for (size_t index = 0; index < result_nodes.size(); index++) {
    // If `PackSchedule` fails, the id of some nodes may be nil.
    if (!result_nodes[index].IsNil()) {
      RAY_CHECK(cluster_resource_manager_.AddNodeAvailableResources(
          result_nodes[index], (*sorted_resource_request_list[index]).GetResourceSet()));
    }
  }

  if (result_nodes.size() != sorted_resource_request_list.size()) {
    // Can't meet the scheduling requirements temporarily.
    return SchedulingResult::Failed();
  }
  return SortSchedulingResult(SchedulingResult::Success(std::move(result_nodes)),
                              sorted_index);
}

/////////////////////  BundleStrictPackSchedulingPolicy  //////////////////////////
SchedulingResult BundleStrictPackSchedulingPolicy::Schedule(
    const std::vector<const ResourceRequest *> &resource_request_list,
    SchedulingOptions options,
    const absl::flat_hash_set<scheduling::NodeID> &candidate_nodes) {
  RAY_CHECK(!resource_request_list.empty());
  if (candidate_nodes.empty()) {
    RAY_LOG(DEBUG) << "The candidate nodes is empty, return directly.";
    return SchedulingResult::Infeasible();
  }

  // Aggregate required resources.
  ResourceRequest aggregated_resource_request;
  LabelSelector aggregated_label_selector;
  for (const auto &resource_request : resource_request_list) {
    for (auto &resource_id : resource_request->ResourceIds()) {
      auto value = aggregated_resource_request.Get(resource_id) +
                   resource_request->Get(resource_id);
      aggregated_resource_request.Set(resource_id, value);
    }
    // Aggregate label constraints from all requests. The selected node
    // must satisfy the union of all label constraints.
    const auto &label_selector = resource_request->GetLabelSelector();
    for (const auto &constraint : label_selector.GetConstraints()) {
      aggregated_label_selector.AddConstraint(constraint);
    }
  }
  aggregated_resource_request.SetLabelSelector(std::move(aggregated_label_selector));

  bool is_infeasible = true;
  for (const auto &node_id : candidate_nodes) {
    if (cluster_resource_manager_.HasFeasibleResources(node_id,
                                                       aggregated_resource_request)) {
      is_infeasible = false;
      break;
    }
  }

  if (is_infeasible) {
    RAY_LOG(DEBUG) << "The required resource is bigger than the maximum resource in the "
                      "whole cluster or no node satisfies the label constraints, "
                      "schedule failed.";
    return SchedulingResult::Infeasible();
  }

  auto best_node_id = scheduling::NodeID::Nil();
  if (!options.bundle_strict_pack_soft_target_node_id_.IsNil()) {
    if (candidate_nodes.contains(options.bundle_strict_pack_soft_target_node_id_)) {
      best_node_id = GetBestNode(aggregated_resource_request,
                                 absl::flat_hash_set<scheduling::NodeID>{
                                     options.bundle_strict_pack_soft_target_node_id_},
                                 options);
    }
  }

  if (best_node_id.IsNil()) {
    best_node_id = GetBestNode(aggregated_resource_request, candidate_nodes, options);
  }

  // Select the node with the highest score.
  // `StrictPackSchedule` does not need to consider the scheduling context, because it
  // only schedules to a node and triggers rescheduling when node dead.
  std::vector<scheduling::NodeID> result_nodes;
  if (!best_node_id.IsNil()) {
    result_nodes.resize(resource_request_list.size(), best_node_id);
  }
  if (result_nodes.empty()) {
    // Can't meet the scheduling requirements temporarily.
    return SchedulingResult::Failed();
  }

  return SchedulingResult::Success(std::move(result_nodes));
}

/////////////////////  BundleStrictSpreadSchedulingPolicy  //////////////////////////
void BundleStrictSpreadSchedulingPolicy::ExcludeNodesAlreadyContainingBundles(
    absl::flat_hash_set<scheduling::NodeID> &candidate_nodes,
    const SchedulingContext *context) {
  const BundleSchedulingContext *bundle_scheduling_context =
      dynamic_cast<const BundleSchedulingContext *>(context);
  if (bundle_scheduling_context &&
      bundle_scheduling_context->bundle_locations_.has_value()) {
    const std::shared_ptr<BundleLocations> &bundle_locations =
        bundle_scheduling_context->bundle_locations_.value();
    if (bundle_locations != nullptr) {
      for (auto &bundle : *bundle_locations) {
        candidate_nodes.erase(scheduling::NodeID(bundle.second.first.Binary()));
      }
    }
  }
}

SchedulingResult BundleStrictSpreadSchedulingPolicy::Schedule(
    const std::vector<const ResourceRequest *> &resource_request_list,
    SchedulingOptions options,
    const absl::flat_hash_set<scheduling::NodeID> &candidate_nodes_in) {
  absl::flat_hash_set<scheduling::NodeID> candidate_nodes = candidate_nodes_in;
  RAY_CHECK(!resource_request_list.empty());

  ExcludeNodesAlreadyContainingBundles(candidate_nodes,
                                       options.scheduling_context_.get());

  if (candidate_nodes.empty()) {
    RAY_LOG(DEBUG) << "The candidate nodes is empty, return directly.";
    return SchedulingResult::Infeasible();
  }

  if (resource_request_list.size() > candidate_nodes.size()) {
    RAY_LOG(DEBUG) << "The number of required resources " << resource_request_list.size()
                   << " is greater than the number of candidate nodes "
                   << candidate_nodes.size() << ", scheduling fails.";
    return SchedulingResult::Infeasible();
  }

  // First schedule scarce resources (such as GPU) and large capacity resources to improve
  // the scheduling success rate.
  auto sorted_result = SortRequiredResources(resource_request_list);
  const auto &sorted_index = sorted_result.first;
  const auto &sorted_resource_request_list = sorted_result.second;

  if (!IsRequestFeasible(sorted_resource_request_list, candidate_nodes)) {
    RAY_LOG(DEBUG) << "Request requires labels or resources not present in the cluster.";
    return SchedulingResult::Infeasible();
  }

  std::vector<scheduling::NodeID> result_nodes;
  for (const auto &resource_request : sorted_resource_request_list) {
    // Score and sort nodes.
    auto best_node_id = GetBestNode(*resource_request, candidate_nodes, options);

    // There are nodes to meet the scheduling requirements.
    if (!best_node_id.IsNil()) {
      candidate_nodes.erase(best_node_id);
      result_nodes.emplace_back(best_node_id);
    } else {
      // There is no node to meet the scheduling requirements.
      break;
    }
  }

  if (result_nodes.size() != sorted_resource_request_list.size()) {
    // Can't meet the scheduling requirements temporarily.
    return SchedulingResult::Failed();
  }
  return SortSchedulingResult(SchedulingResult::Success(std::move(result_nodes)),
                              sorted_index);
}

SchedulingResult HierarchicalBundleSchedulingPolicy::Schedule(
    const std::vector<const ResourceRequest *> &resource_request_list,
    SchedulingOptions options,
    absl::flat_hash_set<scheduling::NodeID> candidate_nodes,
    NodeScheduleFn node_schedule_fn) {
  if (options.bundle_group_indices_.empty()) {
    return node_schedule_fn(resource_request_list, options, candidate_nodes);
  }

  std::vector<std::vector<int>> group_indices = std::move(options.bundle_group_indices_);
  options.bundle_group_indices_.clear();

  std::vector<scheduling::NodeID> final_nodes(resource_request_list.size(),
                                              scheduling::NodeID::Nil());

  const auto &target_domain = options.target_topology_assignment_;
  const std::string &label_key = target_domain.first;

  // Filter candidates by label_key and populate domain_buckets
  absl::flat_hash_map<std::string, absl::flat_hash_set<scheduling::NodeID>>
      domain_buckets;
  if (!label_key.empty()) {
    for (const auto &node : candidate_nodes) {
      const auto &labels = cluster_resource_manager_.GetNodeLabels(node);
      auto it = labels.find(label_key);
      if (it != labels.end()) {
        if (!target_domain.second.has_value() ||
            it->second == target_domain.second.value()) {
          domain_buckets[it->second].insert(node);
        }
      }
    }
  } else {
    domain_buckets[""] = candidate_nodes;
  }

  std::vector<std::string> domain_vals;
  for (const auto &kv : domain_buckets) {
    if (!options.previously_occupied_topologies_.contains(kv.first)) {
      domain_vals.push_back(kv.first);
    }
  }
  std::sort(domain_vals.begin(), domain_vals.end());

  int num_groups = group_indices.size();
  int num_domains = domain_vals.size();

  std::vector<std::string> group_to_domain_str(num_groups);

  // Pre-calculate raw resources for each domain and each group to short-circuit
  // infeasible domains. We use `.total` instead of `.available` so that groups
  // that will *never* fit are marked Infeasible and skipped (PENDING), while
  // groups that *might* fit (but are temporarily out of resources) proceed to
  // the matching algorithm, which correctly fails them (transition to Failed).
  std::vector<ResourceSet> domain_raw_resources(num_domains);
  for (int d = 0; d < num_domains; ++d) {
    for (const auto &node_id : domain_buckets[domain_vals[d]]) {
      const auto &node_resources = cluster_resource_manager_.GetNodeResources(node_id);
      for (auto resource_id : node_resources.total.ExplicitResourceIds()) {
        domain_raw_resources[d].Set(resource_id,
                                    domain_raw_resources[d].Get(resource_id) +
                                        node_resources.total.Get(resource_id));
      }
    }
  }

  std::vector<ResourceSet> group_raw_resources(num_groups);
  for (int g = 0; g < num_groups; ++g) {
    for (int idx : group_indices[g]) {
      group_raw_resources[g] += resource_request_list[idx]->GetResourceSet();
    }
  }

  // Helper to schedule a single group and revert resources.
  auto check_feasibility = [&](int group_idx,
                               const std::string &domain_val,
                               int domain_idx) -> SchedulingResult {
    if (!(group_raw_resources[group_idx] <= domain_raw_resources[domain_idx])) {
      return SchedulingResult::Infeasible();
    }
    const auto &indices = group_indices[group_idx];
    std::vector<const ResourceRequest *> sub_list;
    for (int idx : indices) {
      sub_list.push_back(resource_request_list[idx]);
    }

    SchedulingResult bucket_res =
        node_schedule_fn(sub_list, options, domain_buckets[domain_val]);

    return bucket_res;
  };

  if (options.outer_strategy_ == rpc::PlacementStrategy::STRICT_SPREAD) {
    std::vector<std::string> group_shapes(num_groups);
    for (int i = 0; i < num_groups; i++) {
      std::vector<std::string> shapes;
      for (int idx : group_indices[i]) {
        shapes.push_back(resource_request_list[idx]->GetResourceSet().DebugString());
      }
      std::sort(shapes.begin(), shapes.end());
      for (const auto &s : shapes) {
        group_shapes[i] += s + "|";
      }
    }

    std::vector<std::vector<SchedulingResult>> feasible_results(
        num_groups, std::vector<SchedulingResult>(num_domains));
    std::vector<std::vector<int>> adj(num_groups);
    std::vector<std::vector<int>> adj_possible(num_groups);
    absl::flat_hash_map<std::string, SchedulingResult> memo;

    for (int i = 0; i < num_groups; i++) {
      int orig_group_idx = options.original_bundle_group_indices_.empty()
                               ? i
                               : options.original_bundle_group_indices_[i];
      auto pin_it = options.group_topology_pins_.find(orig_group_idx);
      std::string pin =
          pin_it != options.group_topology_pins_.end() ? pin_it->second : "";

      for (int j = 0; j < num_domains; j++) {
        if (!pin.empty() && domain_vals[j] != pin) {
          continue;
        }
        std::string cache_key = group_shapes[i] + ":" + std::to_string(j);
        auto it = memo.find(cache_key);
        if (it != memo.end()) {
          feasible_results[i][j] = it->second;
        } else {
          feasible_results[i][j] = check_feasibility(i, domain_vals[j], j);
          memo[cache_key] = feasible_results[i][j];
        }
        if (feasible_results[i][j].status.IsSuccess()) {
          adj[i].push_back(j);
          adj_possible[i].push_back(j);
        } else if (!feasible_results[i][j].status.IsInfeasible()) {
          adj_possible[i].push_back(j);
        }
      }
    }

    std::vector<int> group_order(num_groups);
    std::iota(group_order.begin(), group_order.end(), 0);
    std::sort(group_order.begin(), group_order.end(), [&](int a, int b) {
      if (adj[a].size() != adj[b].size()) {
        return adj[a].size() < adj[b].size();
      }
      return a < b;
    });

    std::vector<int> match(num_domains, -1);
    std::vector<bool> visited(num_domains, false);

    std::function<bool(int, const std::vector<std::vector<int>> &, std::vector<int> &)>
        dfs =
            [&](int u, const std::vector<std::vector<int>> &graph, std::vector<int> &m) {
              for (int v : graph[u]) {
                if (visited[v]) continue;
                visited[v] = true;
                if (m[v] < 0 || dfs(m[v], graph, m)) {
                  m[v] = u;
                  return true;
                }
              }
              return false;
            };

    int matched_count = 0;
    for (int i : group_order) {
      std::fill(visited.begin(), visited.end(), false);
      if (dfs(i, adj, match)) {
        matched_count++;
      }
    }

    if (matched_count < num_groups) {
      std::vector<int> match_possible(num_domains, -1);
      int matched_possible_count = 0;
      for (int i : group_order) {
        std::fill(visited.begin(), visited.end(), false);
        if (dfs(i, adj_possible, match_possible)) {
          matched_possible_count++;
        }
      }
      bool is_infeasible = (matched_possible_count < num_groups);
      return is_infeasible ? SchedulingResult::Infeasible() : SchedulingResult::Failed();
    }

    std::vector<int> group_to_domain(num_groups, -1);
    for (int j = 0; j < num_domains; j++) {
      if (match[j] != -1) {
        group_to_domain[match[j]] = j;
      }
    }

    for (int i = 0; i < num_groups; i++) {
      int domain_idx = group_to_domain[i];
      const auto &res = feasible_results[i][domain_idx];
      const auto &indices = group_indices[i];
      if (!label_key.empty()) {
        group_to_domain_str[i] = domain_vals[domain_idx];
      }

      bool exact_fit_failure = false;
      for (size_t k = 0; k < indices.size(); k++) {
        final_nodes[indices[k]] = res.selected_nodes[k];
        if (!cluster_resource_manager_.SubtractNodeAvailableResources(
                final_nodes[indices[k]], *resource_request_list[indices[k]])) {
          exact_fit_failure = true;
          final_nodes[indices[k]] = scheduling::NodeID::Nil();
          break;
        }
      }

      if (exact_fit_failure) {
        // Rollback any subtractions that did succeed.
        for (size_t rollback_idx = 0; rollback_idx < final_nodes.size(); rollback_idx++) {
          if (!final_nodes[rollback_idx].IsNil()) {
            if (!cluster_resource_manager_.AddNodeAvailableResources(
                    final_nodes[rollback_idx],
                    resource_request_list[rollback_idx]->GetResourceSet())) {
              RAY_LOG(ERROR) << "Failed to add resources back to node "
                             << final_nodes[rollback_idx];
            }
            final_nodes[rollback_idx] = scheduling::NodeID::Nil();
          }
        }
        return SchedulingResult::Failed();
      }
    }
  } else if (options.outer_strategy_ == rpc::PlacementStrategy::STRICT_PACK ||
             options.outer_strategy_ == rpc::PlacementStrategy::PACK) {
    // For outer-PACK, all bundle groups must map to the SAME domain.
    // This is enforced by evaluating all groups within a single `domain_val` bucket.
    bool overall_success = false;
    bool all_infeasible = true;

    for (size_t domain_idx = 0; domain_idx < domain_vals.size(); ++domain_idx) {
      const auto &domain_val = domain_vals[domain_idx];
      bool domain_success = true;
      bool domain_infeasible = false;
      std::vector<std::pair<scheduling::NodeID, const ResourceRequest *>> rollback_log;

      bool satisfies_pin = true;
      for (int i = 0; i < num_groups; i++) {
        int orig_group_idx = options.original_bundle_group_indices_.empty()
                                 ? i
                                 : options.original_bundle_group_indices_[i];
        auto pin_it = options.group_topology_pins_.find(orig_group_idx);
        if (pin_it != options.group_topology_pins_.end() &&
            pin_it->second != domain_val) {
          satisfies_pin = false;
          break;
        }
      }
      if (!satisfies_pin) continue;

      for (int i = 0; i < num_groups; i++) {
        const auto &indices = group_indices[i];
        std::vector<const ResourceRequest *> sub_list;
        for (int idx : indices) {
          sub_list.push_back(resource_request_list[idx]);
        }
        if (!(group_raw_resources[i] <= domain_raw_resources[domain_idx])) {
          domain_success = false;
          domain_infeasible = true;
          break;
        }

        SchedulingResult bucket_res =
            node_schedule_fn(sub_list, options, domain_buckets[domain_val]);

        if (bucket_res.status.IsSuccess()) {
          bool exact_fit_failure = false;
          for (size_t k = 0; k < indices.size(); k++) {
            final_nodes[indices[k]] = bucket_res.selected_nodes[k];
            if (!cluster_resource_manager_.SubtractNodeAvailableResources(
                    final_nodes[indices[k]], *sub_list[k])) {
              exact_fit_failure = true;
              break;
            }
            rollback_log.push_back({final_nodes[indices[k]], sub_list[k]});
          }
          if (exact_fit_failure) {
            domain_success = false;
            break;
          }
        } else {
          domain_success = false;
          if (bucket_res.status.IsInfeasible()) {
            domain_infeasible = true;
          }
          break;
        }
      }

      if (domain_success) {
        overall_success = true;
        for (int i = 0; i < num_groups; i++) {
          group_to_domain_str[i] = domain_val;
        }
        break;
      } else {
        if (!domain_infeasible) {
          all_infeasible = false;
        }
        for (auto &entry : rollback_log) {
          if (!cluster_resource_manager_.AddNodeAvailableResources(
                  entry.first, entry.second->GetResourceSet())) {
            RAY_LOG(ERROR) << "Failed to add resources back to node " << entry.first;
          }
        }
      }
    }

    if (!overall_success) {
      return all_infeasible ? SchedulingResult::Infeasible() : SchedulingResult::Failed();
    }
  } else if (options.outer_strategy_ == rpc::PlacementStrategy::SPREAD) {
    absl::flat_hash_set<std::string> used_domains;
    bool overall_success = true;
    bool any_group_infeasible = false;

    for (int i = 0; i < num_groups; i++) {
      bool group_success = false;
      bool group_has_failed_domain = false;
      const auto &indices = group_indices[i];
      std::vector<const ResourceRequest *> sub_list;
      for (int idx : indices) {
        sub_list.push_back(resource_request_list[idx]);
      }

      auto try_domains = [&](bool require_unused) -> bool {
        int orig_group_idx = options.original_bundle_group_indices_.empty()
                                 ? i
                                 : options.original_bundle_group_indices_[i];
        auto pin_it = options.group_topology_pins_.find(orig_group_idx);
        std::string pin =
            pin_it != options.group_topology_pins_.end() ? pin_it->second : "";

        for (size_t domain_idx = 0; domain_idx < domain_vals.size(); ++domain_idx) {
          const auto &domain_val = domain_vals[domain_idx];
          if (!pin.empty() && domain_val != pin) continue;
          if (require_unused && used_domains.contains(domain_val)) continue;
          if (!require_unused && !used_domains.contains(domain_val)) continue;

          if (!(group_raw_resources[i] <= domain_raw_resources[domain_idx])) continue;

          SchedulingResult bucket_res =
              node_schedule_fn(sub_list, options, domain_buckets[domain_val]);
          if (bucket_res.status.IsSuccess()) {
            bool exact_fit_failure = false;
            size_t rollback_idx = 0;
            for (size_t k = 0; k < indices.size(); k++) {
              final_nodes[indices[k]] = bucket_res.selected_nodes[k];
              if (!cluster_resource_manager_.SubtractNodeAvailableResources(
                      final_nodes[indices[k]], *sub_list[k])) {
                exact_fit_failure = true;
                break;
              }
              rollback_idx++;
            }

            if (exact_fit_failure) {
              for (size_t k = 0; k < rollback_idx; k++) {
                if (!cluster_resource_manager_.AddNodeAvailableResources(
                        final_nodes[indices[k]], sub_list[k]->GetResourceSet())) {
                  RAY_LOG(ERROR) << "Failed to add resources back to node "
                                 << final_nodes[indices[k]];
                }
              }
              group_has_failed_domain = true;
              continue;
            }

            used_domains.insert(domain_val);
            group_to_domain_str[i] = domain_val;
            return true;
          } else if (!bucket_res.status.IsInfeasible()) {
            group_has_failed_domain = true;
          }
        }
        return false;
      };

      group_success = try_domains(true);
      if (!group_success) {
        group_success = try_domains(false);
      }

      if (!group_success) {
        overall_success = false;
        if (!group_has_failed_domain) {
          any_group_infeasible = true;
        }
        break;
      }
    }

    if (!overall_success) {
      // Revert any successful group assignments
      for (size_t i = 0; i < final_nodes.size(); i++) {
        if (!final_nodes[i].IsNil()) {
          if (!cluster_resource_manager_.AddNodeAvailableResources(
                  final_nodes[i], resource_request_list[i]->GetResourceSet())) {
            RAY_LOG(ERROR) << "Failed to add resources back to node " << final_nodes[i];
          }
          final_nodes[i] = scheduling::NodeID::Nil();
        }
      }
      return any_group_infeasible ? SchedulingResult::Infeasible()
                                  : SchedulingResult::Failed();
    }
  }

  // Restore the temporarily subtracted resources so the caller gets a clean view.
  for (size_t i = 0; i < final_nodes.size(); i++) {
    if (!final_nodes[i].IsNil()) {
      if (!cluster_resource_manager_.AddNodeAvailableResources(
              final_nodes[i], resource_request_list[i]->GetResourceSet())) {
        RAY_LOG(ERROR) << "Failed to add resources back to node " << final_nodes[i];
      }
    }
  }

  auto success_result = SchedulingResult::Success(std::move(final_nodes));
  if (!label_key.empty()) {
    for (int i = 0; i < num_groups; i++) {
      if (!group_to_domain_str[i].empty()) {
        int orig_group_idx = options.original_bundle_group_indices_.empty()
                                 ? i
                                 : options.original_bundle_group_indices_[i];
        success_result.selected_group_assignments.push_back(
            {orig_group_idx, {label_key, group_to_domain_str[i]}});
      }
    }
  }
  return success_result;
}

}  // namespace raylet_scheduling_policy
}  // namespace ray
