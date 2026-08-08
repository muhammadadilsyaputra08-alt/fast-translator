#include "gbnf.h"
#include <algorithm>
#include <set>
#include <string>

namespace gbnf {

std::string Grammar::signature(const Stack& s) {
    std::string sig;
    sig.reserve(s.size() * 12);
    for (auto& c : s) {
        sig += std::to_string(c.rule_id); sig += ',';
        sig += std::to_string(c.alt_id); sig += ',';
        sig += std::to_string(c.pos); sig += ';';
    }
    return sig;
}

std::vector<Grammar::Stack> Grammar::epsilon_close(Stack stack) const {
    std::string key = signature(stack);
    auto cached = epsilon_cache_.find(key);
    if (cached != epsilon_cache_.end()) return cached->second;

    std::vector<Stack> results;
    std::set<std::string> seen_results;

    auto add_result = [&](Stack&& s) {
        std::string sig = signature(s);
        if (seen_results.insert(sig).second) results.push_back(std::move(s));
    };

    if (stack.empty()) {
        add_result(std::move(stack));
        epsilon_cache_[key] = results;
        return results;
    }

    Cursor top = stack.back();
    const Alternatives& alts = rules_[top.rule_id];
    const Sequence& seq = alts[top.alt_id];

    if (top.pos >= static_cast<int>(seq.size())) {
        Stack popped = stack;
        popped.pop_back();
        for (auto& s : epsilon_close(popped)) add_result(std::move(s));
    } else {
        const Element& el = seq[top.pos];
        if (el.kind == Element::Kind::RuleRef) {
            const Alternatives& sub_alts = rules_[el.rule_id];
            for (int a = 0; a < static_cast<int>(sub_alts.size()); ++a) {
                Stack next = stack;
                next.back().pos += 1;
                next.push_back({el.rule_id, a, 0});
                for (auto& s : epsilon_close(next)) add_result(std::move(s));
            }
        } else {
            add_result(std::move(stack));
        }
    }

    epsilon_cache_[key] = results;
    return results;
}

Grammar::State Grammar::initial_state() const {
    State state;
    Stack root_stack = { {root_rule_id_, 0, 0} };
    // Root may itself have multiple alternatives; seed one stack per
    // top-level alternative of the root rule, then epsilon-close each.
    const Alternatives& root_alts = rules_[root_rule_id_];
    for (int a = 0; a < static_cast<int>(root_alts.size()); ++a) {
        Stack s = { {root_rule_id_, a, 0} };
        for (auto& closed : epsilon_close(s)) state.stacks.push_back(closed);
    }
    return state;
}

bool Grammar::can_accept(const State& state, uint32_t c) const {
    for (const auto& stack : state.stacks) {
        if (stack.empty()) continue; // terminal state can't consume more
        const Cursor& top = stack.back();
        const Element& el = rules_[top.rule_id][top.alt_id][top.pos];
        if (el.kind != Element::Kind::CharRange) continue; // shouldn't happen post-closure
        bool in_range = false;
        for (auto& r : el.ranges) if (c >= r.first && c <= r.second) { in_range = true; break; }
        if (el.negate) in_range = !in_range;
        if (in_range) return true;
    }
    return false;
}

Grammar::State Grammar::advance(const State& state, uint32_t c) const {
    State next;
    for (const auto& stack : state.stacks) {
        if (stack.empty()) continue;
        Cursor top = stack.back();
        const Element& el = rules_[top.rule_id][top.alt_id][top.pos];
        if (el.kind != Element::Kind::CharRange) continue;
        bool in_range = false;
        for (auto& r : el.ranges) if (c >= r.first && c <= r.second) { in_range = true; break; }
        if (el.negate) in_range = !in_range;
        if (!in_range) continue;

        Stack advanced = stack;
        advanced.back().pos += 1;
        for (auto& closed : epsilon_close(advanced)) next.stacks.push_back(std::move(closed));
    }
    return next;
}

bool Grammar::can_terminate(const State& state) const {
    for (const auto& stack : state.stacks) {
        if (stack.empty()) return true;
    }
    return false;
}

} // namespace gbnf
