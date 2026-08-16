#pragma once
#include <vector>
#include <cstdint>

namespace medusa {

// A candidate token proposed by one Medusa head at a given tree depth.
struct TreeNode {
    int token_id;
    int parent_index; // index into the flat node list, -1 for root
    float log_prob;
};

// Builds the attention mask for a Medusa candidate tree: node i may attend
// to node j iff j is an ancestor of i (or j == i). This is what lets all
// candidate branches be verified in a single forward pass.
// Returns an N x N mask (1 = visible, 0 = masked), row-major, N = nodes.size().
std::vector<uint8_t> build_tree_attention_mask(const std::vector<TreeNode>& nodes);

// Typical acceptance sampling (Medusa's alternative to strict argmax
// matching): a candidate token is accepted if its probability under the
// target model is within a "typical" band relative to the entropy of the
// distribution, controlled by epsilon/delta.
//   accept if: p(token) >= min(epsilon, delta * exp(-entropy))
struct TypicalAcceptanceConfig {
    float epsilon = 0.8f;
    float delta = 0.5f;
};

bool typical_acceptance_check(
    const std::vector<float>& target_probs, // full distribution from target model
    int candidate_token,
    const TypicalAcceptanceConfig& cfg);

// Walks the tree from the root, accepting the longest chain of candidate
// tokens whose per-step target distribution passes typical_acceptance_check.
// Returns the accepted token sequence (root not included).
std::vector<int> accept_longest_chain(
    const std::vector<TreeNode>& nodes,
    const std::vector<std::vector<float>>& target_probs_per_node, // same order as nodes
    const TypicalAcceptanceConfig& cfg);

} // namespace medusa
