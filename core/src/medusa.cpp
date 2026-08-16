#include "medusa.h"
#include <cmath>
#include <algorithm>

namespace medusa {

std::vector<uint8_t> build_tree_attention_mask(const std::vector<TreeNode>& nodes) {
    size_t n = nodes.size();
    std::vector<uint8_t> mask(n * n, 0);

    for (size_t i = 0; i < n; ++i) {
        // node i always sees itself
        mask[i * n + i] = 1;
        // walk up the ancestor chain, marking each ancestor visible
        int parent = nodes[i].parent_index;
        while (parent != -1) {
            mask[i * n + parent] = 1;
            parent = nodes[parent].parent_index;
        }
    }
    return mask;
}

bool typical_acceptance_check(
    const std::vector<float>& target_probs,
    int candidate_token,
    const TypicalAcceptanceConfig& cfg) {

    if (candidate_token < 0 || static_cast<size_t>(candidate_token) >= target_probs.size()) return false;

    double entropy = 0.0;
    for (float p : target_probs) {
        if (p > 1e-12f) entropy -= static_cast<double>(p) * std::log(static_cast<double>(p));
    }

    float dynamic_threshold = static_cast<float>(cfg.delta * std::exp(-entropy));
    float threshold = std::min(cfg.epsilon, dynamic_threshold);

    return target_probs[candidate_token] >= threshold;
}

std::vector<int> accept_longest_chain(
    const std::vector<TreeNode>& nodes,
    const std::vector<std::vector<float>>& target_probs_per_node,
    const TypicalAcceptanceConfig& cfg) {

    std::vector<int> accepted;
    if (nodes.empty()) return accepted;

    // Greedy walk: start at root's children, follow the first child at each
    // level that passes typical acceptance. This mirrors Medusa's approach
    // of verifying one path through the candidate tree per forward pass.
    int current_parent = -1;
    bool progressed = true;
    while (progressed) {
        progressed = false;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].parent_index != current_parent) continue;
            if (typical_acceptance_check(target_probs_per_node[i], nodes[i].token_id, cfg)) {
                accepted.push_back(nodes[i].token_id);
                current_parent = static_cast<int>(i);
                progressed = true;
                break;
            }
        }
    }
    return accepted;
}

} // namespace medusa
